#!/bin/bash
# SAVRR — one entry point for everything.
#
#   ./run.sh data                     download AQuA-RAT, filter to clean rationales, build c0/c30/c50 sets
#   ./run.sh train <method> <rate>    train a NEW run              (method: opsd | answer_only | binary | savrr)
#   ./run.sh eval  <method> <rate>    eval the latest such run     (rate: 0 | 30 | 50)
#   ./run.sh eval  runs/<run>         eval a specific run
#   ./run.sh eval  base               eval the untrained base model
#   ./run.sh all   <method> <rate>    train + eval
#
# Every training run gets its own folder, so nothing is ever overwritten:
#   runs/<method>_c<rate>_seed<seed>_<date>-<time>/
#       checkpoint-*/  adapter_model.safetensors   the model
#       train.log  eval.log                        logs
#       eval/                                      per-checkpoint outputs (every response)
#       summary.md  summary.csv  curves.png        performance summary (written by eval)
# Training first runs the unit tests in tests/ (if present) and stops if any fail.
#
# Options (env vars):  GPUS=0 (GPUS=0,1 for 2 GPUs)   SEED=42   MODEL=Qwen/Qwen3-0.6B
#   e.g.  GPUS=1 ./run.sh all opsd 30
set -euo pipefail
cd "$(dirname "$0")"

# use the `opsd` conda env without needing `conda activate`
ENV_BIN="$(conda info --base 2>/dev/null)/envs/${CONDA_ENV:-opsd}/bin"
[ -d "$ENV_BIN" ] && export PATH="$ENV_BIN:$PATH"

GPUS=${GPUS:-0}
SEED=${SEED:-42}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
BASE_DIR=runs/base_$(basename "$MODEL")
export CUDA_VISIBLE_DEVICES=$GPUS TRL_EXPERIMENTAL_SILENCE=1
mkdir -p runs

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
say() { echo "[$(date '+%F %T')] $*"; }

check_args() {
    case "${1:-}" in opsd|answer_only|binary|savrr) ;; *) echo "method must be opsd|answer_only|binary|savrr"; usage;; esac
    case "${2:-}" in 0|30|50) ;; *) echo "rate must be 0|30|50"; usage;; esac
}

data() {
    if [ ! -d data/aqua_rat_raw ]; then
        python -c "from datasets import load_dataset; load_dataset('deepmind/aqua_rat', 'raw').save_to_disk('data/aqua_rat_raw')"
    fi
    python src/build_clean_pool.py --data data/aqua_rat_raw --out_dir data/savrr
    python src/build_corruption_sets.py --pool data/savrr/clean_pool.jsonl --raw data/aqua_rat_raw --out_dir data/savrr/sets
}

train() {   # sets RUN_DIR
    check_args "$@"
    if [ -d tests ]; then
        say "running unit tests"
        python -m pytest tests -q -p no:cacheprovider > /tmp/savrr_tests_$$.log 2>&1 \
            || { cat /tmp/savrr_tests_$$.log; echo "unit tests failed — not training"; exit 1; }
        rm -f /tmp/savrr_tests_$$.log
    fi
    local run=$1_c$2_seed${SEED}_$(date +%Y%m%d-%H%M%S)
    RUN_DIR=runs/$run
    mkdir -p "$RUN_DIR"
    local grad_acc=$((8 / NUM_GPUS))   # effective batch 32 = 4 per GPU x NUM_GPUS x grad-acc
    say "training $run on GPU(s) $GPUS  (log: $RUN_DIR/train.log)"
    accelerate launch \
        --config_file accelerate.yaml \
        --num_processes "$NUM_GPUS" \
        --gradient_accumulation_steps "$grad_acc" \
        --main_process_port $((12000 + RANDOM % 1000)) \
        src/train.py \
        --model_name_or_path "$MODEL" \
        --train_file "data/savrr/sets/train_c$2.jsonl" \
        --supervision "$1" \
        --output_dir "$RUN_DIR" \
        --run_config "$run" \
        --seed "$SEED" \
        --learning_rate 5e-6 \
        --max_grad_norm 0.1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps "$grad_acc" \
        --gradient_checkpointing \
        --num_train_epochs 3 \
        --max_completion_length 1024 \
        --max_length 4096 \
        --save_steps 25 \
        --logging_steps 2 \
        --attn_implementation sdpa \
        --torch_dtype bfloat16 \
        --beta 0 \
        --lmbda 1 \
        --temperature 1.1 \
        --top_p 0.95 \
        --top_k 20 \
        --use_vllm \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.3 \
        --vllm_tensor_parallel_size 1 \
        --use_peft \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --fixed_teacher \
        --jsd_token_clip 0.05 \
        --report_to "${REPORT_TO:-none}" \
        ${EXTRA_ARGS:-} \
        > "$RUN_DIR/train.log" 2>&1 \
        || { say "TRAINING FAILED — see $RUN_DIR/train.log"; exit 1; }
    say "trained -> $RUN_DIR"
}

# eval: score every checkpoint + final model on validation and test, then write the summary
run_eval() {   # $1 = run dir ("$BASE_DIR" for the base model)
    local dir=${1%/} ckpts=()
    mkdir -p "$dir/eval"
    if [ "$dir" != "$BASE_DIR" ]; then
        [ -f "$dir/adapter_model.safetensors" ] || { echo "$dir has no trained model"; exit 1; }
        # the summary compares against the base model, so make sure it has been scored
        [ -f "$BASE_DIR/eval/base__aqua_test_eval.json" ] || run_eval "$BASE_DIR"
        ln -sfn . "$dir/final"   # the final model lives in the run dir itself
        ckpts=(--checkpoint_dir "$dir"/checkpoint-* "$dir/final")
    fi
    say "evaluating $dir  (log: $dir/eval.log)"
    CUDA_VISIBLE_DEVICES=${GPUS%%,*} python src/evaluate.py \
        --base_model "$MODEL" "${ckpts[@]}" \
        --split validation test --n_samples 4 --temperature 0.6 \
        --output_dir "$dir/eval" --skip_existing \
        >> "$dir/eval.log" 2>&1 \
        || { say "EVAL FAILED — see $dir/eval.log"; exit 1; }
    python src/summarize.py "$dir" --base "$BASE_DIR" > /dev/null
    say "done -> $dir/summary.md"
    awk '/^\| Model/,/^$/' "$dir/summary.md"; grep '^Best-val checkpoint' "$dir/summary.md" || true
}

eval_target() {   # resolve "<method> <rate>" | "runs/<run>" | "base" to a run dir
    if [ "${1:-}" = base ]; then echo "$BASE_DIR"
    elif [ -d "${1:-}" ]; then echo "${1%/}"
    elif [ -d "runs/${1:-}" ]; then echo "runs/$1"
    else
        check_args "$@" >&2
        local latest
        latest=$(ls -d runs/$1_c$2_seed${SEED}_* 2>/dev/null | sort | tail -1)
        [ -n "$latest" ] || { echo "no run found for $1 c$2 seed $SEED — train it first" >&2; exit 1; }
        echo "$latest"
    fi
}

case "${1:-}" in
    data)  data ;;
    train) shift; train "$@" ;;
    eval)  shift; dir=$(eval_target "$@"); run_eval "$dir" ;;
    all)   shift; train "$@"; run_eval "$RUN_DIR" ;;
    *)     usage ;;
esac
