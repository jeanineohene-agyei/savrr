"""SAVRR / OPSD training entry point on the AQuA-RAT corruption sets.

One script for all four methods of the study:

  --supervision opsd         q = 1          (standard OPSD: always full reference)
  --supervision answer_only  q = 0          (teacher sees only the correct answer)
  --supervision binary       q = q_binary   (binary AVRR: full ref iff checker finds no error)
  --supervision savrr        q = q          (SAVRR: soft reliability score)

Example (single GPU):
  accelerate launch --num_processes 1 src/train.py \
      --model_name_or_path Qwen/Qwen3-0.6B --train_file data/savrr/sets/train_c30.jsonl \
      --supervision savrr --output_dir runs --run_config savrr_c30 ...   (see run.sh)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import ModelConfig, ScriptArguments, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.gold import GOLDConfig

from trainer import SAVRRTrainer

SUPERVISION_TO_COLUMN = {"opsd": None, "answer_only": None, "binary": "q_binary", "savrr": "q"}


@dataclass
class SAVRRScriptArguments(ScriptArguments):
    train_file: str = field(default=None, metadata={"help": "jsonl from src/build_corruption_sets.py"})
    supervision: str = field(default="savrr", metadata={"help": "opsd | answer_only | binary | savrr"})
    max_train_samples: int = field(default=None, metadata={"help": "debug: truncate the training set"})
    fixed_teacher: bool = field(default=True, metadata={"help": "teacher = base model (LoRA disabled)"})
    run_config: str = field(default=None, metadata={"help": "run name (appended to output_dir, wandb name)"})
    top_k_loss: int = field(default=0, metadata={"help": "restrict divergence to teacher top-k (0 = full vocab)"})
    jsd_token_clip: float = field(default=0.05, metadata={"help": "point-wise divergence clip (0 = none)"})
    use_ema_teacher: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    student_thinking: bool = field(default=False)
    teacher_thinking: bool = field(default=True)
    presence_penalty: float = field(default=0.0)


def load_train_dataset(path, supervision, max_samples=None, seed=0):
    if supervision not in SUPERVISION_TO_COLUMN:
        raise ValueError(f"--supervision must be one of {list(SUPERVISION_TO_COLUMN)}")
    ds = load_dataset("json", data_files=path, split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    col = SUPERVISION_TO_COLUMN[supervision]

    def set_q(ex):
        if supervision == "opsd":
            ex["q"] = 1.0
        elif supervision == "answer_only":
            ex["q"] = 0.0
        else:
            ex["q"] = float(ex[col])
        return ex

    keep = ["id", "question", "options", "correct", "answer_text", "rationale", "q", "q_binary", "corrupted"]
    ds = ds.map(set_q)
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    return ds


def main():
    parser = TrlParser((SAVRRScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher requires --use_peft (the teacher is the model with LoRA disabled)")

    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    eff_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    run_name = script_args.run_config or (
        f"{script_args.supervision}_{Path(script_args.train_file).stem}_"
        f"{Path(model_args.model_name_or_path).name}_lr{training_args.learning_rate:.0e}_bs{eff_bs}")
    if not training_args.output_dir.rstrip("/").endswith(run_name):
        training_args.output_dir = str(Path(training_args.output_dir) / run_name)
    training_args.run_name = run_name

    # our collator needs the raw columns
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {**(training_args.dataset_kwargs or {}), "skip_prepare_dataset": True}
    training_args.presence_penalty = script_args.presence_penalty

    dt = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None) or "bfloat16"
    dtype = dt if isinstance(dt, torch.dtype) else {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16,
        "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32,
    }.get(str(dt).lower(), torch.bfloat16)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "sdpa",
        torch_dtype=dtype,
        use_cache=not training_args.gradient_checkpointing,
    )
    qconf = get_quantization_config(model_args)
    if qconf is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = qconf
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_train_dataset(script_args.train_file, script_args.supervision,
                                       script_args.max_train_samples)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        qs = train_dataset["q"]
        print(f"\n{'=' * 80}\nSAVRR run: {run_name}\n  supervision={script_args.supervision}  "
              f"train_file={script_args.train_file}  n={len(train_dataset)}\n"
              f"  q: mean={sum(qs) / len(qs):.4f}  q==1: {sum(q == 1 for q in qs)}  "
              f"q==0: {sum(q == 0 for q in qs)}  0<q<1: {sum(0 < q < 1 for q in qs)}\n"
              f"  output_dir={training_args.output_dir}\n{'=' * 80}\n")

    trainer = SAVRRTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
