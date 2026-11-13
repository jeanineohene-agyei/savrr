"""Build the SAVRR training sets with controlled, nested rationale corruption.

From data/savrr/clean_pool.jsonl (rationales whose arithmetic is verified
clean), select N examples with >= 2 corruptible steps and write three training
sets that contain the SAME N examples:

  train_c0.jsonl   all rationales clean
  train_c30.jsonl  30% of rationales corrupted
  train_c50.jsonl  50% of rationales corrupted   (superset of the c30 corruptions,
                                                  identical corrupted text)

Corruption of one example:
  * pick c ~ Uniform{1, ..., min(n_corruptible, m - 1)} steps (never all m steps,
    never the step that states the final answer)
  * for each chosen step, rewrite the stated number with a fixed mix:
      - 'relative': multiply by 1 +- U(0.10, 0.50), keep the original precision
      - 'digit'   : change one digit
  * re-run the checker on the corrupted rationale and accept only if exactly
    the c chosen steps are contradicted and the number of checks m is unchanged
    (otherwise redraw).  So the checker detects every corruption by construction.

The question, options and labelled answer are never modified.

Every output example carries the checker's view of the rationale it contains:
  m, k, q (soft reliability, Eq. 2), q_binary (1 if k == 0 else 0)
so the trainer can switch between OPSD (q=1), answer-only (q=0), binary AVRR
(q_binary) and SAVRR (q) without recomputing anything.

Also writes the official AQuA validation/test splits in the same format.

Usage:
  python src/build_corruption_sets.py --pool data/savrr/clean_pool.jsonl \
      --raw data/aqua_rat_raw --out_dir data/savrr/sets --n 2000 --seed 0
"""

import argparse
import collections
import json
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arithmetic_checker import check_rationale, clean_rationale, reliability_score  # noqa: E402

OPTION_RE = re.compile(r"^\s*\(?([A-E])\s*\)\s*(.*)$", re.S)
RATES = (0, 30, 50)


# ---------------------------------------------------------------- perturbations
def _fmt_like(orig: str, val: float) -> str:
    """Format val with the same decimals and thousands-comma style as orig."""
    d = len(orig.split(".")[1]) if "." in orig else 0
    s = f"{val:.{d}f}"
    if "," in orig:
        ip, _, fp = s.partition(".")
        neg = ip.startswith("-")
        ip = f"{int(ip.lstrip('-')):,}"
        s = ("-" if neg else "") + ip + ("." + fp if fp else "")
    if orig.startswith(".") and s.startswith("0."):
        s = s[1:]  # ".75" style
    return s


def perturb(orig: str, kind: str, rng: random.Random):
    """Return a corrupted version of the number string `orig`, or None."""
    if kind == "relative":
        v = float(orig.replace(",", ""))
        if v == 0:
            return None
        f = 1 + rng.choice([-1, 1]) * rng.uniform(0.10, 0.50)
        d = len(orig.split(".")[1]) if "." in orig else 0
        nv = round(v * f, d)
        if nv == v or nv <= 0 < v:
            return None
        return _fmt_like(orig, nv)
    if kind == "digit":
        digs = [i for i, ch in enumerate(orig) if ch.isdigit()]
        i = rng.choice(digs)
        leading = i == digs[0] and len(orig.replace(",", "").split(".")[0]) > 1
        choices = [c for c in "0123456789" if c != orig[i] and not (leading and c == "0")]
        return orig[:i] + rng.choice(choices) + orig[i + 1:]
    raise ValueError(kind)


def corrupt_example(ex, rng, max_tries=50):
    """Corrupt 1..m-1 corruptible steps; return (text, corruptions, report) or None."""
    cands = [c for c in ex["checks"] if c["corruptible"]]
    # a stated number shared by two checks (middle of a chain) would flip both
    span_count = collections.Counter(tuple(c["stated_literal_span"]) for c in cands)
    cands = [c for c in cands if span_count[tuple(c["stated_literal_span"])] == 1]
    hi = min(len(cands), ex["m"] - 1)
    if hi < 1:
        return None
    for _ in range(max_tries):
        n_corr = rng.randint(1, hi)
        chosen = rng.sample(cands, n_corr)
        edits = []
        for c in chosen:
            a, b = c["stated_literal_span"]
            orig = ex["rationale"][a:b]
            kind = rng.choice(["relative", "digit"])
            new = perturb(orig, kind, rng)
            if new is None or new == orig:
                break
            edits.append(dict(span=[a, b], orig=orig, new=new, kind=kind,
                              step=f"{c['lhs']} = {c['rhs']}",
                              correct_value=c["correct_value"]))
        if len(edits) != n_corr:
            continue
        text = ex["rationale"]
        for e in sorted(edits, key=lambda e: -e["span"][0]):
            a, b = e["span"]
            text = text[:a] + e["new"] + text[b:]
        rep = check_rationale(text)
        if rep.m != ex["m"] or rep.k != n_corr or rep.artifacts:
            continue
        flagged = sorted(c.stated_literal_text for c in rep.contradicted)
        if flagged != sorted(e["new"] for e in edits):
            continue
        # record new spans (shifted by earlier edits) for traceability
        shift = 0
        for e in sorted(edits, key=lambda e: e["span"][0]):
            a, b = e["span"]
            e["new_span"] = [a + shift, a + shift + len(e["new"])]
            shift += len(e["new"]) - (b - a)
        return text, edits, rep
    return None


# ---------------------------------------------------------------- formatting
def record(ex, rationale, report, corruptions):
    q = reliability_score(report)
    return dict(
        id=ex["id"],
        question=ex["question"],
        options=ex["options"],
        correct=ex["correct"],
        answer_text=ex["options"][ex["correct"]],
        rationale=rationale,
        corrupted=bool(corruptions),
        n_corrupted=len(corruptions),
        corruptions=corruptions,
        m=report.m,
        k=report.k,
        q=q,
        q_binary=1.0 if report.k == 0 else 0.0,
    )


def eval_records(raw_split):
    out = []
    for i, ex in enumerate(raw_split):
        opts = {}
        for o in ex["options"]:
            m = OPTION_RE.match(clean_rationale(o))
            if m:
                opts[m.group(1)] = m.group(2).strip()
        out.append(dict(id=i, question=clean_rationale(ex["question"]), options=opts,
                        correct=ex["correct"], answer_text=opts.get(ex["correct"], "")))
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def describe(vals):
    if not vals:
        return {}
    qs = sorted(vals)
    return dict(n=len(qs), mean=round(statistics.mean(qs), 4), min=round(qs[0], 4),
                p25=round(qs[len(qs) // 4], 4), median=round(qs[len(qs) // 2], 4),
                p75=round(qs[3 * len(qs) // 4], 4), max=round(qs[-1], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/savrr/clean_pool.jsonl")
    ap.add_argument("--raw", default="data/aqua_rat_raw")
    ap.add_argument("--out_dir", default="data/savrr/sets")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    pool = [json.loads(l) for l in open(args.pool)]
    pool = [ex for ex in pool if ex["n_corruptible"] >= 2]
    rng.shuffle(pool)
    print(f"eligible pool (n_corruptible >= 2): {len(pool)}")

    n_corrupt_max = round(args.n * max(RATES) / 100)       # 1000
    n_corrupt_by_rate = {r: round(args.n * r / 100) for r in RATES}
    corrupted_group, clean_group, n_failed = [], [], 0
    for ex in pool:
        if len(corrupted_group) < n_corrupt_max:
            res = corrupt_example(ex, rng)
            if res is None:
                n_failed += 1
                continue
            corrupted_group.append((ex, res))
        elif len(clean_group) < args.n - n_corrupt_max:
            clean_group.append(ex)
        else:
            break
    assert len(corrupted_group) == n_corrupt_max and len(clean_group) == args.n - n_corrupt_max, \
        "pool too small"

    # corrupted_group order defines nesting: first 600 -> corrupted in c30 and c50
    stats = dict(n=args.n, seed=args.seed, corruption_failures_skipped=n_failed, rates={})
    for rate in RATES:
        n_c = n_corrupt_by_rate[rate]
        rows = []
        for j, (ex, (text, edits, rep)) in enumerate(corrupted_group):
            if j < n_c:
                rows.append(record(ex, text, rep, edits))
            else:
                rows.append(record(ex, ex["rationale"], check_rationale(ex["rationale"]), []))
        for ex in clean_group:
            rows.append(record(ex, ex["rationale"], check_rationale(ex["rationale"]), []))
        random.Random(args.seed + 1).shuffle(rows)   # same order in every set
        write_jsonl(os.path.join(args.out_dir, f"train_c{rate}.jsonl"), rows)

        corr = [r for r in rows if r["corrupted"]]
        stats["rates"][rate] = dict(
            n_corrupted=len(corr),
            frac_corrupted=round(len(corr) / len(rows), 4),
            q_corrupted=describe([r["q"] for r in corr]),
            q_all=describe([r["q"] for r in rows]),
            n_steps_corrupted_hist=dict(sorted(collections.Counter(r["n_corrupted"] for r in corr).items())),
            kind_counts=dict(collections.Counter(e["kind"] for r in corr for e in r["corruptions"])),
            clean_rows_with_q_lt_1=sum(1 for r in rows if not r["corrupted"] and r["q"] < 1),
        )

    # sanity: nesting + identical ids/order
    sets = {r: [json.loads(l) for l in open(os.path.join(args.out_dir, f"train_c{r}.jsonl"))] for r in RATES}
    assert [x["id"] for x in sets[0]] == [x["id"] for x in sets[30]] == [x["id"] for x in sets[50]]
    c30 = {x["id"]: x["rationale"] for x in sets[30] if x["corrupted"]}
    c50 = {x["id"]: x["rationale"] for x in sets[50] if x["corrupted"]}
    assert set(c30) <= set(c50) and all(c50[i] == t for i, t in c30.items())
    assert all(x["k"] == 0 and x["q"] == 1.0 for x in sets[0])
    stats["nesting_verified"] = True

    from datasets import load_from_disk
    raw = load_from_disk(args.raw)
    write_jsonl(os.path.join(args.out_dir, "validation.jsonl"), eval_records(raw["validation"]))
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), eval_records(raw["test"]))

    with open(os.path.join(args.out_dir, "sets_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
