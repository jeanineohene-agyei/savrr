"""Write the performance report for ONE run, from the eval files in <run_dir>/eval/.

Outputs (inside the run dir):
  summary.md    headline test accuracy, comparison with the base model, per-checkpoint table
  summary.csv   one row per checkpoint
  curves.png    validation / test accuracy vs training step (base model as reference line)

Model selection: the checkpoint with the best *validation* sampled accuracy (ties -> later
step) is reported on test, so test is never used for selection.

Called by ./run.sh eval;  manual use:  python src/summarize.py runs/<run> [--base runs/base_Qwen3-0.6B]
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re

METHOD_NAME = {"opsd": "OPSD (full reference)", "answer_only": "Answer-only OPSD",
               "binary": "Binary AVRR", "savrr": "SAVRR"}
# validated categorical slots 1-2 + muted reference ink (dataviz reference palette)
C_SAMPLED, C_GREEDY, C_BASE, C_GRID, C_TEXT = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#3d3c38"


def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)


def fmt(x, n=None):
    if x is None:
        return "–"
    return f"{100 * x:.1f}" + (f" ± {100 * ci95(x, n):.1f}" if n else "")


def step_of(ckpt):
    return 0 if ckpt == "base" else math.inf if ckpt == "final" else int(ckpt.split("-")[1])


def load(run_dir):
    """{ckpt: {split: {"summary": ..., "results": ...}}}"""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "eval", "*__aqua_*_eval.json")):
        m = re.match(r"(.+)__aqua_(validation|test)_eval\.json$", os.path.basename(path))
        with open(path) as f:
            out.setdefault(m.group(1), {})[m.group(2)] = json.load(f)
    return out


def get(d, split, key):
    s = d.get(split, {}).get("summary")
    if s is None:
        return None
    return s.get(key, s["greedy_accuracy"]) if key == "sampled_accuracy" else s.get(key)


def per_problem(d):
    return {r["id"]: r.get("sampled_acc", float(r["is_correct"])) for r in d["test"]["results"]}


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(n_boot))
    return sum(diffs) / len(diffs), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1], \
        sum(x <= 0 for x in boots) / n_boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--base", default=None, help="run dir of the base-model eval, for comparison")
    args = ap.parse_args()
    run = os.path.basename(os.path.normpath(args.run_dir))
    evals = load(args.run_dir)
    if not evals:
        raise SystemExit(f"no eval files in {args.run_dir}/eval")
    base = load(args.base).get("base") if args.base and os.path.isdir(args.base) else None
    ckpts = sorted(evals, key=step_of)
    scored = [c for c in ckpts if "validation" in evals[c] and "test" in evals[c]]
    n = evals[scored[0]]["test"]["summary"]["n"]

    # ---- csv
    keys = [("val_sampled", "validation", "sampled_accuracy"), ("val_greedy", "validation", "greedy_accuracy"),
            ("test_sampled", "test", "sampled_accuracy"), ("test_greedy", "test", "greedy_accuracy"),
            ("test_unparsed", "test", "unparsed")]
    with open(os.path.join(args.run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint"] + [k for k, _, _ in keys])
        for c in ckpts:
            w.writerow([c] + [get(evals[c], s, k) for _, s, k in keys])

    # ---- markdown
    m = re.match(r"(opsd|answer_only|binary|savrr)_c(\d+)_seed(\d+)", run)
    title = f"{METHOD_NAME[m.group(1)]}, {m.group(2)}% corrupted rationales, seed {m.group(3)}" if m \
        else "Base model (no training)"
    best = max(scored, key=lambda c: (get(evals[c], "validation", "sampled_accuracy"), step_of(c)))
    final = "final" if "final" in scored else scored[-1]
    lines = [f"# {run}", "", f"**{title}** — AQuA-RAT, {n} test problems.",
             "Accuracy in %, ± = 95% CI. *Sampled* = mean over 4 samples at T=0.6; *greedy* = T=0.", "",
             "| Model | Val sampled | Test sampled | Test greedy |", "|---|---|---|---|"]
    rows = [("base", base)] if base else []
    if run.startswith("base"):
        rows = []
    rows += [(f"best-val checkpoint ({best})", evals[best])]
    if final != best:
        rows += [(f"final checkpoint ({final})", evals[final])]
    for name, d in rows:
        lines.append(f"| {'Base model' if name == 'base' else name} | {fmt(get(d, 'validation', 'sampled_accuracy'))} | "
                     f"{fmt(get(d, 'test', 'sampled_accuracy'), n)} | {fmt(get(d, 'test', 'greedy_accuracy'), n)} |")
    lines.append("")

    if base and not run.startswith("base"):
        obs, lo, hi, p = paired_bootstrap(per_problem(evals[best]), per_problem(base))
        lines += ["## vs. base model (paired bootstrap, same test problems)", "",
                  f"Best-val checkpoint − base: **{100 * obs:+.1f} pts** test sampled accuracy, "
                  f"95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}], P(Δ ≤ 0) = {p:.3f}.", ""]

    if len(ckpts) > 1:
        lines += ["## Per checkpoint", "", "| Checkpoint | Val sampled | Val greedy | Test sampled | Test greedy |",
                  "|---|---|---|---|---|"]
        for c in ckpts:
            d = evals[c]
            lines.append(f"| {c}{' ← best val' if c == best else ''} | {fmt(get(d, 'validation', 'sampled_accuracy'))} | "
                         f"{fmt(get(d, 'validation', 'greedy_accuracy'))} | {fmt(get(d, 'test', 'sampled_accuracy'))} | "
                         f"{fmt(get(d, 'test', 'greedy_accuracy'))} |")
        lines += ["", "![curves](curves.png)", ""]
    lines.append(f"Note: with {n} test problems, differences of ~2–3 points are within noise for a single seed.")
    with open(os.path.join(args.run_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- curves (only meaningful for trained runs)
    steps = [c for c in ckpts if c not in ("final", "base")]
    if len(steps) < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
                         "xtick.color": C_BASE, "ytick.color": C_BASE, "text.color": C_TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, split in zip(axes, ["validation", "test"]):
        x = [step_of(c) for c in steps]
        for key, color, label in [("sampled_accuracy", C_SAMPLED, "sampled (4 × T=0.6)"),
                                  ("greedy_accuracy", C_GREEDY, "greedy")]:
            y = [100 * get(evals[c], split, key) for c in steps]
            ax.plot(x, y, color=color, lw=2, marker="o", ms=4, label=label)
            ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=C_TEXT)
        if base and split in base:
            ax.axhline(100 * get(base, split, "sampled_accuracy"), color=C_BASE, lw=1, ls=":",
                       label="base model (sampled)")
        b = step_of(best)
        ax.axvline(b, color=C_GRID, lw=1)
        ax.text(b, ax.get_ylim()[1], " best val", va="top", fontsize=8, color=C_BASE)
        ax.set_title(f"AQuA {split}", loc="left", fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(axis="y", color=C_GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.suptitle(f"{run}: {title}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(args.run_dir, "curves.png"), dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
