"""Answer-choice accuracy on AQuA-RAT (official test/validation splits).

Uses exactly the student prompt from training (prompts.student_user_message),
so the evaluated model sees the same format it was distilled on.

Examples:
  # base model
  python src/evaluate.py --base_model Qwen/Qwen3-0.6B --split test
  # trained LoRA checkpoints (any number, one vLLM load), both splits
  python src/evaluate.py --base_model Qwen/Qwen3-0.6B --split validation test \
      --checkpoint_dir runs/savrr_c30_seed42/checkpoint-*
  # average accuracy over 8 samples at T=0.6 (in addition to greedy)
  python src/evaluate.py ... --n_samples 8 --temperature 0.6
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import student_user_message  # noqa: E402

LETTERS = "ABCDE"


def extract_boxed(text: str):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += text[j] == "{"
        depth -= text[j] == "}"
        if depth == 0:
            return text[i + 1: j].strip()
    return None


def extract_letter(text: str, options: dict):
    """Predicted option letter, or None.

    1. \\boxed{...}: a letter (A / (A) / A) / \\text{A}), or a value matching an option
    2. fallback: last "answer is (X)" / "Answer: X" in the response
    """
    # drop any thinking block
    if "</think>" in text:
        text = text.split("</think>")[-1]
    box = extract_boxed(text)
    if box is not None:
        b = re.sub(r"\\text\{([^}]*)\}", r"\1", box).strip()
        m = re.fullmatch(r"\(?\s*([A-E])\s*\)?\s*[.:)]?(?:\s.*)?", b)
        if m:
            return m.group(1)
        norm = lambda s: re.sub(r"[\s$,]|\\%|%", "", s).lower()
        for k, v in options.items():
            if norm(v) and norm(v) == norm(b):
                return k
    hits = re.findall(r"(?:answer|option)\s*(?:is)?[\s:*]*\(?\s*([A-E])\b", text, flags=re.I)
    hits = [h for h in hits if h.isupper()]
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--checkpoint_dir", nargs="*", default=None,
                    help="one or more LoRA checkpoint dirs; all are scored with one vLLM instance. "
                         "Omit to score the base model.")
    ap.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "../data/savrr/sets"))
    ap.add_argument("--split", nargs="+", default=["test"], choices=["test", "validation"])
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--n_samples", type=int, default=0, help="extra sampled generations per problem (0 = greedy only)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode (training used off)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default=None,
                    help="where to write results (default: inside each checkpoint dir; cwd for the base model)")
    ap.add_argument("--skip_existing", action="store_true", help="skip (checkpoint, split) pairs already scored")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.base_model)
    ckpts = [Path(c) for c in (args.checkpoint_dir or [])]
    for c in ckpts:
        if not any((c / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin")):
            raise FileNotFoundError(f"no LoRA adapter in {c}")
    targets = ckpts or [None]

    def out_path(ckpt, split):
        name = f"aqua_{split}_eval.json"
        if args.output_dir:
            tag = "base" if ckpt is None else ckpt.name
            return Path(args.output_dir) / f"{tag}__{name}"
        return (ckpt / name) if ckpt is not None else Path(f"aqua_{split}_{Path(args.base_model).name}_eval.json")

    todo = [(c, s) for c in targets for s in args.split
            if not (args.skip_existing and out_path(c, s).exists())]
    if not todo:
        print("nothing to do"); return

    llm_kwargs = dict(model=args.base_model, tensor_parallel_size=args.tensor_parallel_size,
                      gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
                      max_model_len=4096 + args.max_tokens)
    if ckpts:
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank, max_loras=1)
    llm = LLM(**llm_kwargs)
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    for lora_id, (ckpt, split) in enumerate(todo, start=1):
        rows = [json.loads(l) for l in open(Path(args.data_dir) / f"{split}.jsonl")]
        prompts = [tok.apply_chat_template([{"role": "user", "content": student_user_message(r)}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=args.thinking) for r in rows]
        lora = LoRARequest(f"ckpt{lora_id}", lora_id, str(ckpt)) if ckpt is not None else None
        outs = llm.generate(prompts, greedy, lora_request=lora)
        sampled = None
        if args.n_samples > 0:
            sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, n=args.n_samples,
                                max_tokens=args.max_tokens, seed=args.seed)
            sampled = llm.generate(prompts, sp, lora_request=lora)

        results, n_correct, n_unparsed, avg_n = [], 0, 0, []
        for i, r in enumerate(rows):
            text = outs[i].outputs[0].text
            pred = extract_letter(text, r["options"])
            ok = pred == r["correct"]
            n_correct += ok
            n_unparsed += pred is None
            rec = dict(id=r["id"], correct=r["correct"], pred=pred, is_correct=ok, response=text,
                       n_tokens=len(outs[i].outputs[0].token_ids), finish_reason=outs[i].outputs[0].finish_reason)
            if sampled is not None:
                preds = [extract_letter(o.text, r["options"]) for o in sampled[i].outputs]
                rec["sampled_preds"] = preds
                rec["sampled_n_tokens"] = [len(o.token_ids) for o in sampled[i].outputs]
                rec["sampled_finish_reasons"] = [o.finish_reason for o in sampled[i].outputs]
                rec["sampled_acc"] = sum(p == r["correct"] for p in preds) / len(preds)
                avg_n.append(rec["sampled_acc"])
            results.append(rec)

        summary = dict(base_model=args.base_model, checkpoint=str(ckpt) if ckpt else None, split=split,
                       n=len(rows), greedy_accuracy=n_correct / len(rows), unparsed=n_unparsed,
                       max_tokens=args.max_tokens,
                       hit_limit=sum(r["finish_reason"] == "length" for r in results))
        if avg_n:
            summary["n_samples"] = args.n_samples
            summary["temperature"] = args.temperature
            summary["sampled_unparsed"] = sum(p is None for r in results for p in r["sampled_preds"])
            summary["sampled_hit_limit"] = sum(f == "length" for r in results for f in r["sampled_finish_reasons"])
            summary["sampled_accuracy"] = sum(avg_n) / len(avg_n)
        print(json.dumps(summary, indent=2))
        out = out_path(ckpt, split)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(summary=summary, results=results), f, indent=2)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
