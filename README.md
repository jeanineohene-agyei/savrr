# SAVRR — Soft Arithmetic-Verified Reference Routing

Class project (UC Davis). Extends On-Policy Self-Distillation (OPSD) by scoring each
AQuA-RAT reference rationale with an arithmetic checker (reliability `q`) and training
the student against `q · p_full + (1 − q) · p_answer_only`. Model: Qwen3-0.6B + LoRA.
See the proposal PDF for details.

## Quick start

```bash
conda env create -f environment.yml     # once; creates the `opsd` env (run.sh uses it automatically)

./run.sh data                 # download AQuA-RAT, filter to clean rationales, build c0/c30/c50 sets
./run.sh all savrr 30         # train a new run, then evaluate it
```

| Command | What it does |
|---|---|
| `./run.sh train <method> <rate>` | trains a **new** run (unit tests run first) |
| `./run.sh eval <method> <rate>` | evaluates the latest such run (or `./run.sh eval runs/<run>`) |
| `./run.sh eval base` | evaluates the untrained base model (done automatically before the first run eval) |
| `./run.sh all <method> <rate>` | train + eval |

Methods: `opsd` (q = 1), `answer_only` (q = 0), `binary` (q = 1 iff no error found), `savrr` (soft q).
Corruption rates: `0`, `30`, `50`. Options are env vars: `GPUS=1`, `GPUS=0,1` (two GPUs, same effective
batch of 32), `SEED=43`. One condition takes ~2 h training + ~50 min eval on one A100; for long runs use tmux:
`tmux new -s savrr './run.sh all opsd 30'`.

## Outputs: one folder per run, nothing is overwritten

```
runs/<method>_c<rate>_seed<seed>_<date>-<time>/
    summary.md  curves.png  summary.csv     performance summary (best-val checkpoint on test, vs. base model)
    train.log  eval.log
    eval/                                   every model response, per checkpoint and split
    checkpoint-*/  adapter_model.safetensors
runs/base_Qwen3-0.6B/                       base-model reference (same files, no checkpoints)
```

## Layout

| Path | What |
|---|---|
| `run.sh` | the only command you need |
| `src/arithmetic_checker.py` | the checker: `check_rationale`, `reliability_score` |
| `src/build_clean_pool.py`, `src/build_corruption_sets.py` | data pipeline (run by `./run.sh data`) |
| `src/validate_detection.py` | measures how often the checker catches injected errors |
| `src/prompts.py` | student / full-reference teacher / answer-only teacher prompts |
| `src/trainer.py` | `SAVRRTrainer`: the teacher mixture and loss |
| `src/train.py` | training entry point (`--supervision opsd\|answer_only\|binary\|savrr`) |
| `src/evaluate.py` | AQuA accuracy (greedy + sampled) with vLLM |
| `src/summarize.py` | writes a run's summary.md / curves.png (called by eval) |
| `src/opsd_trainer.py`, `src/data_collator.py` | copied unchanged from the official OPSD repo |
| `tests/` | local unit tests (not in git; run before every training if present) |
| `data/savrr/sets/` | `train_c{0,30,50}.jsonl`, `validation.jsonl`, `test.jsonl` (committed; `./run.sh data` rebuilds them identically) |
