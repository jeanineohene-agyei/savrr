"""Validate that the checker detects injected corruptions in the clean pool.

For every corruptible step of every clean example, perturb the stated number
four ways (+-1 in the last digit, +-10%, x2, one random digit changed), re-run
the checker on the corrupted rationale and report the detection rate.

Usage: python src/validate_detection.py [data/savrr/clean_pool.jsonl]
"""
import json, sys, random, collections
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arithmetic_checker import check_rationale

def fmt_like(orig, val):
    """Format val with the same number of decimals / thousands commas as orig."""
    d = len(orig.split('.')[1]) if '.' in orig else 0
    s = f"{val:.{d}f}"
    if ',' in orig:
        ip, _, fp = s.partition('.')
        neg = ip.startswith('-'); ip = ip.lstrip('-')
        ip = f"{int(ip):,}"; s = ('-' if neg else '') + ip + ('.' + fp if fp else '')
    return s

def perturb(orig, kind, rng):
    v = float(orig.replace(',', ''))
    d = len(orig.split('.')[1]) if '.' in orig else 0
    u = 10 ** -d
    if kind == 'plus1ulp':   nv = v + u * rng.choice([-1, 1])
    elif kind == 'pct10':    nv = v * (1 + rng.choice([-0.1, 0.1]))
    elif kind == 'x2':       nv = v * 2
    elif kind == 'digit':    # change one digit
        digs = [i for i, ch in enumerate(orig) if ch.isdigit()]
        i = rng.choice(digs); ch = orig[i]
        new = rng.choice([c for c in '0123456789' if c != ch and not (i == 0 and c == '0')])
        return orig[:i] + new + orig[i+1:]
    nv = round(nv, d)
    if nv == v: nv = v + u
    return fmt_like(orig, nv)

rng = random.Random(0)
pool = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'data/savrr/clean_pool.jsonl')]
res = collections.defaultdict(collections.Counter)
misses = []
for ex in pool:
    for ci, c in enumerate(ex['checks']):
        if not c['corruptible']: continue
        a, b = c['stated_literal_span']
        orig = ex['rationale'][a:b]
        for kind in ['plus1ulp', 'pct10', 'x2', 'digit']:
            new = perturb(orig, kind, rng)
            if new == orig: continue
            txt = ex['rationale'][:a] + new + ex['rationale'][b:]
            rep = check_rationale(txt)
            r = res[kind]; r['n'] += 1
            if rep.k == 0:
                r['missed'] += 1
                if len(misses) < 15: misses.append((kind, orig, new, c['lhs'], c['rhs']))
            elif rep.k == 1: r['k=1'] += 1
            else: r['k>1'] += 1
            if rep.m != ex['m']: r['m_changed'] += 1
for k, r in res.items():
    print(f"{k:9s} n={r['n']:6d} detected={1-r['missed']/r['n']:.4f}  k=1:{r['k=1']/r['n']:.3f} k>1:{r['k>1']/r['n']:.3f} m_changed:{r['m_changed']/r['n']:.3f}")
for m in misses: print('MISS', m)
