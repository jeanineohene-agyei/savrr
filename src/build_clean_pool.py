"""Filter AQuA-RAT train into a pool of rationales with verified-clean arithmetic.

An example is kept only if ALL of the following hold:
  1. 5 well-formed options A-E, and `correct` is one of them
  2. the answer letter stated in the rationale (if any) matches `correct`
  3. the question is not a duplicate of an earlier train question, and does
     not appear in the official validation or test sets
  4. the arithmetic checker finds >= 2 checkable steps, 0 contradicted
  5. no scrape/layout artifacts (tabs, MathJax duplication, lost superscripts)
  6. at least one checked step has a bare-number result that can be corrupted
     without touching the final answer

Outputs (in --out_dir):
  clean_pool.jsonl     one example per line, with checker spans
  filter_stats.json    how many examples each criterion removed
  inspect_clean.txt    random clean examples rendered for manual review
  inspect_flagged.txt  random examples rejected by the checker (k > 0)

Usage:
  python src/build_clean_pool.py --data data/aqua_rat_raw --out_dir data/savrr
"""

import argparse
import collections
import json
import os
import random
import re
import sys

from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arithmetic_checker import check_rationale, clean_rationale, tokenize  # noqa: E402

OPTION_RE = re.compile(r"^\s*\(?([A-E])\s*\)\s*(.*)$", re.S)
ANSWER_RES = [
    re.compile(r"(?:answer|ans|option|correct\s+choice)\s*(?:is|will\s+be|should\s+be)?\s*[:\-=.]*\s*"
               r"(?:option\s*)?[:\-]?\s*\(?\s*([A-E])\b(?![a-z])", re.I),
    re.compile(r"(?:^|\n)\s*\(?([A-E])\)?\s*[.)]?\s*$"),
    re.compile(r"(?:hence|so|thus|therefore)\s*,?\s*\(?([A-E])\)?\s*[.!]?\s*$", re.I),
]


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_rationale(q).lower()).strip()


def parse_options(options):
    parsed = {}
    for o in options:
        m = OPTION_RE.match(clean_rationale(o))
        if not m:
            return None
        parsed[m.group(1)] = m.group(2).strip()
    return parsed if sorted(parsed) == list("ABCDE") else None


def stated_answer_letter(rationale: str):
    """Last answer letter the rationale commits to, or None if it never says.

    Only uppercase letters count, to avoid matching words like 'a'.
    """
    tail = rationale[-120:]
    hits = []
    for rx in ANSWER_RES:
        for m in rx.finditer(tail):
            if m.group(1).isupper():
                hits.append((m.end(), m.group(1)))
    return max(hits)[1] if hits else None


def option_value(text: str):
    """Numeric value of an option like '$1.40', '37 yrs.', '16/35', '12%'."""
    toks = [t for t in tokenize(text) if t.kind not in ("WS", "CUR")]
    nums = [t for t in toks if t.kind == "NUM"]
    if len(nums) == 1:
        return float(nums[0].text.replace(",", ""))
    if len(nums) == 2 and len(toks) == 3 and toks[1].kind == "OP" and toks[1].op == "/":
        d = float(nums[1].text.replace(",", ""))
        return float(nums[0].text.replace(",", "")) / d if d else None
    return None


def render(ex, report=None):
    lines = [f"### train[{ex['id']}]  correct={ex['correct']}  options={ex['options']}",
             f"Q: {ex['question']}", "RATIONALE:", ex["rationale"], "CHECKS:"]
    checks = ex["checks"] if report is None else [c.as_dict() for c in report.checks]
    for c in checks:
        mark = "ok " if c["ok"] else "BAD"
        fa = "  [final-answer step]" if c.get("is_final_answer_step") else ""
        lines.append(f"  [{mark}] {c['lhs']!r} = {c['rhs']!r}   "
                     f"(correct {c['correct_value']:.6g}, stated {c['stated_value']:.6g}){fa}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aqua_rat_raw")
    ap.add_argument("--out_dir", default="data/savrr")
    ap.add_argument("--max_rationale_chars", type=int, default=2000)
    ap.add_argument("--n_inspect", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_from_disk(args.data)
    heldout = {norm_q(q) for split in ("validation", "test") for q in ds[split]["question"]}

    stats = collections.Counter()
    seen = set()
    clean, flagged = [], []
    for i, ex in enumerate(ds["train"]):
        stats["total"] += 1
        opts = parse_options(ex["options"])
        if opts is None or ex["correct"] not in opts:
            stats["drop_bad_options"] += 1; continue
        nq = norm_q(ex["question"])
        if nq in heldout:
            stats["drop_overlaps_val_test"] += 1; continue
        if nq in seen:
            stats["drop_duplicate_question"] += 1; continue
        seen.add(nq)
        rat = clean_rationale(ex["rationale"])
        if len(rat) > args.max_rationale_chars:
            stats["drop_too_long"] += 1; continue
        letter = stated_answer_letter(rat)
        if letter is None:
            stats["note_no_stated_letter"] += 1
        elif letter != ex["correct"]:
            stats["drop_answer_letter_mismatch"] += 1; continue

        rep = check_rationale(rat)
        if rep.artifacts:
            stats["drop_layout_artifact"] += 1
            for a in rep.artifacts:
                stats[f"  artifact:{a}"] += 1
            continue
        if rep.k > 0:
            stats["drop_arithmetic_error"] += 1
            flagged.append((i, ex, rat, rep))
            continue
        if rep.m < 2:
            stats["drop_fewer_than_2_checks"] += 1; continue

        ans_val = option_value(opts[ex["correct"]])
        checks = []
        for c in rep.checks:
            d = c.as_dict()
            d["is_final_answer_step"] = bool(
                ans_val is not None and math_close(c.stated_value, ans_val))
            d["corruptible"] = c.stated_literal_span is not None and not d["is_final_answer_step"]
            checks.append(d)
        if not any(d["corruptible"] for d in checks):
            stats["drop_no_corruptible_step"] += 1; continue

        stats["kept"] += 1
        clean.append(dict(
            id=i, question=clean_rationale(ex["question"]), options=opts,
            correct=ex["correct"], answer_value=ans_val, rationale=rat,
            stated_letter=letter, m=rep.m, n_corruptible=sum(d["corruptible"] for d in checks),
            checks=checks,
        ))

    with open(os.path.join(args.out_dir, "clean_pool.jsonl"), "w") as f:
        for ex in clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    m_hist = collections.Counter(min(ex["m"], 10) for ex in clean)
    stats_out = dict(stats)
    stats_out["clean_m_histogram(10=10+)"] = dict(sorted(m_hist.items()))
    with open(os.path.join(args.out_dir, "filter_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)

    rng = random.Random(args.seed)
    with open(os.path.join(args.out_dir, "inspect_clean.txt"), "w") as f:
        for ex in rng.sample(clean, min(args.n_inspect, len(clean))):
            f.write(render(ex) + "\n")
    with open(os.path.join(args.out_dir, "inspect_flagged.txt"), "w") as f:
        for i, ex, rat, rep in rng.sample(flagged, min(args.n_inspect, len(flagged))):
            f.write(render(dict(id=i, question=ex["question"], options=ex["options"],
                                correct=ex["correct"], rationale=rat, checks=[]), rep) + "\n")

    for k, v in stats_out.items():
        print(f"{k:35s} {v}")


def math_close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


if __name__ == "__main__":
    main()
