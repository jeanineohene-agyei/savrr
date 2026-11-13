"""Arithmetic checker for SAVRR.

Extracts explicit arithmetic equalities from a natural-language rationale
(e.g. AQuA-RAT), evaluates them exactly, and reports which ones are
contradicted.  Also computes the rationale reliability score q(r) from the
SAVRR proposal (Eq. 1-2).

Design principle: **precision over recall**.  Anything ambiguous (variables,
operator words like "of"/"times", ratios, unit conversions, pi, ...) is treated
as *uncheckable* rather than guessed at.  Every stated result that *is* checked
comes with character offsets into the cleaned rationale so that controlled
corruption can later edit exactly that number.

Public API
----------
clean_rationale(text)   -> str              (fix mojibake, normalise newlines)
check_rationale(text)   -> RationaleReport  (text should already be cleaned)
reliability_score(rep)  -> float            (q(r) in [0, 1])
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Union

try:  # optional, only needed for clean_rationale
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None

Num = Union[Fraction, float]

EPS = 1e-8          # epsilon in Eq. (1)
MAX_MAGNITUDE = 1e18
REL_TOL = 1e-9

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_rationale(text: str) -> str:
    """Repair mojibake and normalise line endings / odd whitespace.

    The checker's character offsets refer to the string returned here, so this
    cleaned string is what should be stored and used for training.
    """
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("​", "")
    return text.strip()


# --------------------------------------------------------------------------
# Tokeniser
# --------------------------------------------------------------------------

_SUPERSCRIPTS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
                 "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_VULGAR = {"½": Fraction(1, 2), "¼": Fraction(1, 4), "¾": Fraction(3, 4),
           "⅓": Fraction(1, 3), "⅔": Fraction(2, 3), "⅛": Fraction(1, 8)}

_TOKEN_RE = re.compile(
    r"""
     (?P<WS>\s+)
    |(?P<SEP>==>|=>|-{2,}>?|->|<=|>=|=<|!=|‹=›|[⇒⇔→∴∵≠<>;])
    |(?P<COLON>:)
    |(?P<EQ>==|=)
    |(?P<COMB>\d+[CP]\d+(?![\w.]))
    |(?P<NUM>(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|\.\d+)
    |(?P<FRAC>[½¼¾⅓⅔⅛])
    |(?P<SUP>[⁰¹²³⁴⁵⁶⁷⁸⁹]+)
    |(?P<CUR>(?:Rs|INR|USD)\.?(?![A-Za-z])|[$€£₹])
    |(?P<SQRT>√|sqrt(?=\s*\())
    |(?P<OP>\*\*|[+\-*/×÷^−–—∗⁄·%!])
    |(?P<LP>[(\[])
    |(?P<RP>[)\]])
    |(?P<WORD>[A-Za-z]+)
    |(?P<OTHER>.)
    """,
    re.VERBOSE,
)

_OP_CANON = {"×": "*", "∗": "*", "·": "*", "÷": "/", "⁄": "/",
             "−": "-", "–": "-", "—": "-", "**": "^"}

ARITH_KINDS = {"NUM", "COMB", "FRAC", "SUP", "CUR", "SQRT", "OP", "LP", "RP", "WS"}

# Words that, when adjacent to a number, change its meaning (so the number
# alone is NOT the operand of the equality).
DENY_WORDS = {
    "of", "times", "by", "into", "plus", "minus", "less", "more", "than",
    "over", "per", "and", "from", "to", "with", "divided", "multiplied",
    "product", "sum", "difference", "square", "squared", "cube", "cubed",
    "root", "twice", "thrice", "half", "double", "triple", "average", "mean",
    "percent", "percentage", "pc", "out", "lakh", "lakhs", "lac", "lacs",
    "crore", "crores", "thousand", "thousands", "million", "millions",
    "billion", "billions", "hundred", "hundreds", "dozen", "dozens", "th",
    "st", "nd", "rd", "power", "raised", "inverse", "reciprocal", "log", "ln",
    "sin", "cos", "tan", "mod", "remainder", "pi", "e", "x", "y", "z", "n",
    "or", "is", "are", "was", "were",
}
# "is"/"are" are in DENY_WORDS only for the *following* side (e.g. "= 5 is
# the answer" is fine, but we strip them separately); see _FOLLOW_OK below.
_FOLLOW_OK = {"is", "are", "was", "were"}
_PRECEDE_DENY = DENY_WORDS - {"is", "are", "was", "were"}

UNITS = {
    "m", "cm", "mm", "km", "kms", "kg", "kgs", "g", "gm", "gms", "gram",
    "grams", "mg", "l", "lt", "ltr", "ltrs", "litre", "litres", "liter",
    "liters", "ml", "sec", "secs", "second", "seconds", "s", "min", "mins",
    "minute", "minutes", "hr", "hrs", "hour", "hours", "h", "day", "days",
    "week", "weeks", "month", "months", "year", "years", "yr", "yrs",
    "kmph", "kmh", "kmhr", "kmphr", "mph", "mps", "metre", "metres", "meter",
    "meters", "mtr", "mtrs", "feet", "ft", "foot", "inch", "inches", "yard",
    "yards", "mile", "miles", "rupees", "rupee", "rs", "paise", "dollars",
    "dollar", "cents", "cent", "units", "unit", "sq", "cu", "cubic",
    "degrees", "degree", "acres", "acre", "hectares", "hectare", "ton",
    "tons", "tonnes", "pounds", "lbs", "lb", "oz", "ounces", "gallons",
    "gallon", "pages", "people", "persons", "men", "women", "students",
    "boys", "girls", "workers", "items", "ways", "marks", "votes", "balls",
    "km/hr", "km/h", "m/s", "m/sec",
}
_SINGLE_LETTER_UNITS = {"m", "s", "g", "l", "h"}


@dataclass
class Tok:
    kind: str
    text: str
    start: int
    end: int

    @property
    def op(self) -> str:
        return _OP_CANON.get(self.text, self.text)


def tokenize(text: str) -> list[Tok]:
    toks: list[Tok] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        toks.append(Tok(kind, m.group(), m.start(), m.end()))
    _contextualize_x(toks)
    _contextualize_colon(toks)
    return toks


def _contextualize_colon(toks: list[Tok]) -> None:
    """':' between numbers is a ratio (makes the segment uncheckable); else a separator."""
    num_like = ("NUM", "RP", "FRAC", "LP", "SUP", "COMB")
    for i, t in enumerate(toks):
        if t.kind == "COLON":
            p, n = _prev_nonws(toks, i), _next_nonws(toks, i)
            if p >= 0 and n < len(toks) and toks[p].kind in num_like and toks[n].kind in num_like:
                t.kind = "OTHER"
            else:
                t.kind = "SEP"


def _prev_nonws(toks, i):
    j = i - 1
    while j >= 0 and toks[j].kind == "WS":
        j -= 1
    return j


def _next_nonws(toks, i):
    j = i + 1
    while j < len(toks) and toks[j].kind == "WS":
        j += 1
    return j


def _contextualize_x(toks: list[Tok]) -> None:
    """Turn 'x'/'X' into a multiplication operator when it sits between operands.

    If the rationale also uses x as a *variable* (e.g. "let x be", "4x,", "x ="),
    an unspaced "2x2" is more likely 2x^2 with a lost superscript, so only a
    whitespace-delimited " x " is treated as multiplication.
    """
    cands, var_mode = [], False
    for i, t in enumerate(toks):
        if t.kind == "WORD" and t.text in ("x", "X"):
            p, n = _prev_nonws(toks, i), _next_nonws(toks, i)
            pt = toks[p] if p >= 0 else None
            nt = toks[n] if n < len(toks) else None
            left_ok = pt is not None and (pt.kind in ("NUM", "RP", "SUP", "FRAC", "COMB") or (
                pt.kind == "OP" and pt.op in ("%", "!")))
            right_ok = nt is not None and nt.kind in ("NUM", "LP", "SQRT", "CUR", "FRAC", "COMB")
            if left_ok and right_ok:
                spaced = (i > 0 and toks[i - 1].kind == "WS") and (i + 1 < len(toks) and toks[i + 1].kind == "WS")
                cands.append((t, spaced))
            else:
                var_mode = True
    for t, spaced in cands:
        if spaced or not var_mode:
            t.kind, t.text = "OP", "*"


# --------------------------------------------------------------------------
# Clause / segment splitting
# --------------------------------------------------------------------------

def split_clauses(toks: list[Tok]) -> list[list[Tok]]:
    return [c for c, _ in split_clauses_with_starts(toks)]


def split_clauses_with_starts(toks: list[Tok]) -> list[tuple[list[Tok], str]]:
    """Split tokens into clauses at hard separators.

    A newline is a separator unless the next line starts with '=' (or the
    current line ends with '=' / an operator), which is the common
    "continued derivation" layout in AQuA.
    """
    clauses: list[tuple[list[Tok], str]] = []
    cur: list[Tok] = []
    start_reason = "start"

    def flush(reason="sep"):
        nonlocal cur, start_reason
        if cur:
            clauses.append((cur, start_reason))
        cur = []
        start_reason = reason

    n = len(toks)
    depth = 0
    for i, t in enumerate(toks):
        if t.kind == "LP":
            depth += 1
        elif t.kind == "RP":
            depth = max(0, depth - 1)
        if t.kind == "SEP":
            flush(); depth = 0
            continue
        if t.kind == "WS" and "\n" in t.text:
            p, q = _prev_nonws(toks, i), _next_nonws(toks, i)
            cont = (q < n and toks[q].kind == "EQ") or (
                p >= 0 and (toks[p].kind == "EQ" or
                            (toks[p].kind == "OP" and toks[p].op in "+-*/^")))
            if not cont:
                flush()
                continue
        if t.kind == "OTHER" and t.text == "," and depth == 0:
            if i + 1 >= n or toks[i + 1].kind == "WS":
                flush("comma")
                continue
        if t.kind == "OTHER" and t.text in ".?!":
            # sentence end: '.' followed by whitespace + capital / newline / end
            nxt = toks[i + 1] if i + 1 < n else None
            if nxt is None:
                flush()
                continue
            if nxt.kind == "WS":
                q = _next_nonws(toks, i)
                if "\n" in nxt.text or q >= n or (
                        toks[q].kind == "WORD" and toks[q].text[0].isupper()):
                    flush()
                    continue
        cur.append(t)
    flush()
    return clauses


def split_segments(clause: list[Tok]) -> list[list[Tok]]:
    segs: list[list[Tok]] = [[]]
    for t in clause:
        if t.kind == "EQ":
            segs.append([])
        else:
            segs[-1].append(t)
    return segs


# --------------------------------------------------------------------------
# Expression extraction from a segment
# --------------------------------------------------------------------------

@dataclass
class Expr:
    toks: list[Tok]            # arithmetic tokens only (no WS needed)
    unit: str                  # normalised trailing unit ('' if none)
    start: int
    end: int
    literal: Optional[Tok] = None   # NUM token if the expression is a bare literal
    literal_neg: bool = False
    has_percent: bool = False
    has_operation: bool = False
    alt: Optional["Expr"] = None    # alternative reading of the same text

    @property
    def text_span(self):
        return (self.start, self.end)


def _strip_ws(toks):
    a, b = 0, len(toks)
    while a < b and toks[a].kind == "WS":
        a += 1
    while b > a and toks[b - 1].kind == "WS":
        b -= 1
    return toks[a:b]


def _balance(toks: list[Tok], side: str) -> Optional[list[Tok]]:
    """Make a run's parentheses balanced, or return None.

    side='left'  (run ends at '='):    an unmatched leading '(' may be stripped,
                                       e.g. "(3 + 4 = 7)".  An unmatched ')'
                                       means the run is the tail of a
                                       parenthetical, e.g. "f(1, 4) =" -> None.
    side='right' (run starts at '='):  mirror image.
    side='middle':                     must already be balanced.
    """
    toks = list(toks)
    while toks:
        depth, min_depth = 0, 0
        for t in toks:
            depth += t.kind == "LP"
            depth -= t.kind == "RP"
            min_depth = min(min_depth, depth)
        if min_depth == 0 and depth == 0:
            return toks
        if side == "left" and min_depth == 0 and depth > 0 and toks[0].kind == "LP":
            toks = _strip_ws(toks[1:]); continue
        if side == "right" and depth < 0 and toks[-1].kind == "RP" and depth == min_depth:
            toks = _strip_ws(toks[:-1]); continue
        return None
    return None


def _is_label_group(toks: list[Tok], i: int) -> bool:
    """'(1)', '(2)' ... not preceded by an explicit operator: an equation label."""
    if not (i + 2 < len(toks) and toks[i].kind == "LP" and toks[i + 1].kind == "NUM"
            and toks[i + 2].kind == "RP"):
        return False
    txt = toks[i + 1].text
    if not (len(txt) == 1 and txt.isdigit()):
        return False
    j = i - 1
    while j >= 0 and toks[j].kind == "WS":
        j -= 1
    return j < 0 or not (toks[j].kind == "OP" and toks[j].op in ("+", "-", "*", "/", "^"))


def _parse_unit_tail(toks: list[Tok]) -> Optional[str]:
    """Parse a trailing unit like 'km/hr', 'm/s', 'kmph', 'km per hour'.

    Returns normalised unit string ('' for none) or None if toks is not a unit.
    """
    toks = [t for t in toks if t.kind != "WS"]
    if toks and toks[-1].kind == "OTHER" and toks[-1].text == ".":
        toks = toks[:-1]
    if not toks:
        return ""
    words = []
    expect_word = True
    for t in toks:
        if expect_word:
            if t.kind == "WORD" and t.text.lower() in UNITS:
                words.append(t.text.lower()); expect_word = False
            else:
                return None
        else:
            if (t.kind == "OP" and t.op == "/") or (t.kind == "WORD" and t.text.lower() == "per"):
                words.append("/"); expect_word = True
            elif t.kind == "WORD" and t.text.lower() in UNITS:  # "sq m", "cubic cm"
                words.append(t.text.lower())
            else:
                return None
    if expect_word:
        return None
    return "".join(words)


def _make_expr(toks: list[Tok], unit: str, side: str) -> Optional[Expr]:
    toks = _balance(_strip_ws(toks), side)
    if not toks:
        return None
    if any(_is_label_group(toks, i) for i in range(len(toks))):
        return None  # "(1) = (2) + (3)", "9.84(2)"
    core = [t for t in toks if t.kind not in ("WS", "CUR")]
    if not core or not any(t.kind in ("NUM", "COMB", "FRAC") for t in core):
        return None
    e = Expr(toks=core, unit=unit, start=toks[0].start, end=toks[-1].end)
    e.has_percent = any(t.kind == "OP" and t.op == "%" for t in core)
    no_pct = [t for t in core if not (t.kind == "OP" and t.op == "%")]
    if len(no_pct) == 1 and no_pct[0].kind == "NUM":
        e.literal = no_pct[0]
    elif len(no_pct) == 2 and no_pct[0].kind == "OP" and no_pct[0].op == "-" and no_pct[1].kind == "NUM":
        e.literal, e.literal_neg = no_pct[1], True
    e.has_operation = any(
        (t.kind == "OP" and t.op in ("+", "-", "*", "/", "^", "!")) or t.kind in ("SQRT", "SUP", "COMB")
        for t in core) and e.literal is None
    # implicit multiplication like 3/8(1900) also counts as an operation
    if e.literal is None and not e.has_operation and len(no_pct) > 1:
        e.has_operation = True
    return e


def _is_binary_start(t: Tok) -> bool:
    return t.kind == "OP" and t.op in ("+", "*", "/", "^", "%", "!")


def _is_dangling_end(t: Tok) -> bool:
    return (t.kind == "OP" and t.op in ("+", "-", "*", "/", "^")) or t.kind in ("SQRT", "LP", "CUR")


def extract_left(seg: list[Tok], clause_start: str = "start") -> Optional[Expr]:
    """Trailing expression of the first segment of a chain (left of the first '=')."""
    seg = _strip_ws(seg)
    if not seg or seg[-1].kind not in ARITH_KINDS:
        return None  # ends with a word/unit, e.g. "in 26 seconds ="
    i = len(seg)
    while i > 0 and seg[i - 1].kind in ARITH_KINDS:
        i -= 1
    run = _strip_ws(seg[i:])
    if not run:
        return None
    if _is_binary_start(run[0]) or _is_dangling_end(run[-1]):
        return None
    # what precedes the run?
    if i > 0:
        before = seg[i - 1]  # non-arith token (WS is arith, so this is adjacent or after WS)
        adjacent = seg[i].kind != "WS"
        if adjacent:
            return None  # e.g. "R+8", "S.I.=", "a,b"
        if before.kind == "OTHER" and before.text not in "'\"":
            return None  # "2 : 3 + 1 =" (ratio), "& 5 ="
        if before.kind == "WORD" and before.text.lower() in _PRECEDE_DENY:
            return None
        if run[0].kind == "OP" and run[0].op == "-" and before.kind in ("WORD", "OTHER"):
            return None  # "speed - 3 = ..."
        if run[0].kind == "LP" and before.kind == "WORD" and _wraps_whole(run):
            return None  # "75% of D (60) = 45": parenthetical annotation
    e = _make_expr(run, "", "left")
    if e is not None and e.literal is not None and (i > 0 or clause_start == "comma"):
        # A bare number preceded by words ("growth on 1000 = (32-11)",
        # "arranging 4 digits as 2 = 5C4") is a quantity mention, not an operand.
        return None
    return e


def _wraps_whole(run: list[Tok]) -> bool:
    """True if run is '( ... )' with the first '(' closing at the very end."""
    depth = 0
    for k, t in enumerate(run):
        depth += t.kind == "LP"
        depth -= t.kind == "RP"
        if depth == 0:
            return k == len(run) - 1
    return False


def extract_right(seg: list[Tok], following_eq: bool = False) -> Optional[Expr]:
    """Leading expression of the last segment of a chain (right of the last '=')."""
    seg = _strip_ws(seg)
    if not seg:
        return None
    j = 0
    while j < len(seg) and seg[j].kind in ARITH_KINDS:
        j += 1
    run = _strip_ws(seg[:j])
    rest = seg[j:]
    if not run:
        return None
    # "= 385(350+35)", "= 210 (approx)": a stated number followed by a
    # parenthetical is an annotation, not implicit multiplication.
    core_idx = [k for k, t in enumerate(run) if t.kind not in ("WS", "CUR")]
    annotated = None
    if len(core_idx) >= 2 and run[core_idx[0]].kind == "NUM" and run[core_idx[1]].kind == "LP":
        annotated = _make_expr(run[:core_idx[1]], "", "right")
    if _is_dangling_end(run[-1]) or _is_binary_start(run[0]):
        return annotated
    unit = ""
    if rest:
        first = rest[0]
        adjacent = seg[j - 1].kind != "WS"
        if first.kind == "WORD":
            w = first.text.lower()
            if adjacent:
                if w in UNITS and w not in _SINGLE_LETTER_UNITS:
                    unit = _unit_prefix(rest)
                else:
                    return None  # "2S", "4x", "10th", "5k"
            else:
                if w in DENY_WORDS and w not in _FOLLOW_OK:
                    return None
                if len(w) == 1 and w not in _SINGLE_LETTER_UNITS:
                    return None
                if w in UNITS:
                    unit = _unit_prefix(rest)
                    # a unit followed by an operator means the unit letter is a variable
                    k = _unit_len(rest)
                    after = _strip_ws(rest[k:])
                    if after and after[0].kind == "OP":
                        return None
        elif first.kind == "OTHER":
            if first.text not in ".?!'\"":
                return None  # "= 5,6", "= 5°", "= 2 : 3" (ratio)
        # SUP/NUM etc. can't be here (they're arith)
    full = _make_expr(run, unit, "right")
    if annotated is not None:
        if full is None:
            return annotated
        full.alt = annotated  # "= 7(3*4)" (implicit mult) vs "= 385(350+35)" (annotation)
    return full


def _unit_len(toks: list[Tok]) -> int:
    """Number of tokens at the start of toks forming a unit (incl. internal WS)."""
    best = 0
    for k in range(1, min(len(toks), 6) + 1):
        if _parse_unit_tail(toks[:k]) not in (None, ""):
            best = k
    return best


def _unit_prefix(toks: list[Tok]) -> str:
    k = _unit_len(toks)
    return _parse_unit_tail(toks[:k]) or "" if k else ""


def extract_middle(seg: list[Tok]) -> Optional[Expr]:
    """A segment between two '=' must be a pure expression (+ optional unit)."""
    seg = _strip_ws(seg)
    if not seg:
        return None
    j = 0
    while j < len(seg) and seg[j].kind in ARITH_KINDS:
        j += 1
    run, rest = seg[:j], seg[j:]
    if rest and seg[j - 1].kind != "WS" and j > 0:
        # attached word: only multi-letter units allowed ("30cm")
        if not (rest[0].kind == "WORD" and rest[0].text.lower() in UNITS
                and rest[0].text.lower() not in _SINGLE_LETTER_UNITS):
            return None
    unit = _parse_unit_tail(rest)
    if unit is None:
        return None
    run = _strip_ws(run)
    if not run or _is_dangling_end(run[-1]) or _is_binary_start(run[0]):
        return None
    return _make_expr(run, unit, "middle")


# --------------------------------------------------------------------------
# Evaluation (exact where possible)
# --------------------------------------------------------------------------

class EvalError(Exception):
    pass


def _num(text: str) -> Fraction:
    return Fraction(text.replace(",", ""))


def _guard(v: Num) -> Num:
    if isinstance(v, float) and not math.isfinite(v):
        raise EvalError("non-finite")
    if abs(v) > MAX_MAGNITUDE:
        raise EvalError("too large")
    return v


def _pow(a: Num, b: Num) -> Num:
    if isinstance(b, Fraction) and b.denominator == 1 and abs(b) <= 64 and isinstance(a, Fraction):
        if a == 0 and b < 0:
            raise EvalError("0 ** negative")
        if abs(a) > 10 ** 6 and b > 3:
            raise EvalError("too large")
        return _guard(a ** int(b))
    try:
        if a < 0:
            raise EvalError("negative base with non-integer exponent")
        return _guard(float(a) ** float(b))
    except OverflowError:
        raise EvalError("overflow")


def _isqrt_exact(n: int) -> Optional[int]:
    if n < 0:
        return None
    r = math.isqrt(n)
    return r if r * r == n else None


def _sqrt(a: Num) -> Num:
    if a < 0:
        raise EvalError("sqrt of negative")
    if isinstance(a, Fraction):
        p, q = _isqrt_exact(a.numerator), _isqrt_exact(a.denominator)
        if p is not None and q is not None:
            return Fraction(p, q)
    return math.sqrt(float(a))


def _fact(a: Num) -> Num:
    if not (isinstance(a, Fraction) and a.denominator == 1 and 0 <= a <= 20):
        raise EvalError("bad factorial")
    return Fraction(math.factorial(int(a)))


class _Parser:
    """Recursive-descent parser over arithmetic tokens.

    pct_mode: 'strip' -> '%' is ignored;  'pct' -> x% == x/100
    """

    def __init__(self, toks: list[Tok], pct_mode: str):
        self.toks = toks
        self.i = 0
        self.pct = pct_mode

    def peek(self) -> Optional[Tok]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> Tok:
        t = self.peek()
        if t is None:
            raise EvalError("unexpected end")
        self.i += 1
        return t

    def parse(self) -> Num:
        v = self.expr()
        if self.peek() is not None:
            raise EvalError(f"trailing token {self.peek().text!r}")
        return v

    def expr(self) -> Num:
        v = self.term()
        while (t := self.peek()) is not None and t.kind == "OP" and t.op in ("+", "-"):
            self.take()
            r = self.term()
            v = _guard(v + r if t.op == "+" else v - r)
        return v

    def term(self) -> Num:
        v = self.unary()
        while True:
            t = self.peek()
            if t is None:
                return v
            if t.kind == "OP" and t.op in ("*", "/"):
                self.take()
                r = self.unary()
                if t.op == "*":
                    v = _guard(v * r)
                else:
                    if r == 0:
                        raise EvalError("division by zero")
                    v = _guard(v / r) if isinstance(v, float) or isinstance(r, float) else _guard(Fraction(v) / r)
            elif t.kind in ("LP", "SQRT"):
                # implicit multiplication: 3/8(1900), 2(3+4), 2√3
                r = self.unary()
                v = _guard(v * r)
            else:
                return v

    def unary(self) -> Num:
        t = self.peek()
        if t is not None and t.kind == "OP" and t.op in ("+", "-"):
            self.take()
            v = self.unary()
            return -v if t.op == "-" else v
        return self.power()

    def power(self) -> Num:
        base = self.postfix()
        t = self.peek()
        if t is not None and t.kind == "OP" and t.op == "^":
            nx = self.toks[self.i + 1: self.i + 3]
            if len(nx) == 2 and nx[0].kind == "NUM" and nx[1].kind == "OP" and nx[1].op == "/":
                raise EvalError("ambiguous exponent a^b/c")
            self.take()
            exp = self.unary()  # right-assoc
            return _pow(base, exp)
        return base

    def postfix(self) -> Num:
        v = self.atom()
        while (t := self.peek()) is not None:
            if t.kind == "OP" and t.op == "!":
                self.take(); v = _fact(v)
            elif t.kind == "OP" and t.op == "%":
                self.take()
                if self.pct == "pct":
                    v = v / 100 if isinstance(v, float) else Fraction(v) / 100
            elif t.kind == "SUP":
                self.take()
                v = _pow(v, Fraction(int("".join(_SUPERSCRIPTS[c] for c in t.text))))
            else:
                break
        return v

    def atom(self) -> Num:
        t = self.take()
        if t.kind == "NUM":
            nxt = self.peek()
            if nxt is not None and nxt.kind in ("NUM", "FRAC", "COMB"):
                raise EvalError("adjacent numbers")  # "2 1/2", "16 28"
            return _num(t.text)
        if t.kind == "FRAC":
            return _VULGAR[t.text]
        if t.kind == "COMB":
            n, r = re.split(r"[CP]", t.text)
            n, r = int(n), int(r)
            if not (0 <= r <= n <= 1000):
                raise EvalError("bad nCr")
            return Fraction(math.comb(n, r) if "C" in t.text else math.perm(n, r))
        if t.kind == "LP":
            v = self.expr()
            c = self.take()
            if c.kind != "RP":
                raise EvalError("missing )")
            return v
        if t.kind == "SQRT":
            return _sqrt(self.postfix())
        raise EvalError(f"unexpected token {t.text!r}")


def evaluate(expr: Expr, pct_mode: str = "strip", slash_mode: str = "standard") -> Num:
    """Evaluate an expression.

    slash_mode='standard': normal precedence.
    slash_mode='loose':    the expression's single top-level '/' splits it into
                           numerator and denominator, the way people commonly
                           write "48 / 4 x 2" for 48/(4x2) or "3+4+6/12" for
                           (3+4+6)/12.  Raises EvalError if there isn't exactly
                           one top-level '/'.
    """
    for a, b, c in zip(expr.toks, expr.toks[1:], expr.toks[2:]):
        if a.kind == "OP" and a.op == "^" and b.kind == "NUM" and c.kind == "OP" and c.op == "/":
            raise EvalError("ambiguous exponent a^b/c")  # x^1/2: sqrt or (x^1)/2?
    if slash_mode == "standard":
        return _Parser(expr.toks, pct_mode).parse()
    toks = expr.toks
    while len(toks) > 2 and toks[0].kind == "LP" and _wraps_whole(toks):
        toks = toks[1:-1]
    depth, cuts = 0, []
    for k, t in enumerate(toks):
        depth += t.kind == "LP"
        depth -= t.kind == "RP"
        if depth == 0 and t.kind == "OP" and t.op == "/":
            cuts.append(k)
    if len(cuts) != 1:
        raise EvalError("loose slash needs exactly one top-level '/'")
    k = cuts[0]
    d = 0
    for t in toks[k + 1:]:
        d += t.kind == "LP"
        d -= t.kind == "RP"
        if d == 0 and t.kind == "OP" and t.op in ("+", "-"):
            # "600/75 - 1" is never meant as 600/(75-1); only "a/b x c" -> a/(b x c)
            raise EvalError("loose denominator must be multiplicative")
    num = _Parser(toks[:k], pct_mode).parse()
    den = _Parser(toks[k + 1:], pct_mode).parse()
    if den == 0:
        raise EvalError("division by zero")
    if isinstance(num, float) or isinstance(den, float):
        return _guard(float(num) / float(den))
    return _guard(Fraction(num) / den)


def _interpretations(e: Expr):
    """Yield (pct_mode, slash_mode) pairs, standard reading first."""
    pcts = ("strip", "pct") if e.has_percent else ("strip",)
    n_slash = sum(t.kind == "OP" and t.op == "/" for t in e.toks)  # top-level checked in evaluate()
    slashes = ("standard", "loose") if n_slash == 1 and e.literal is None and len(e.toks) > 3 else ("standard",)
    for sl in slashes:
        for pc in pcts:
            yield pc, sl


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _decimals(num_text: str) -> int:
    s = num_text.replace(",", "")
    return len(s.split(".")[1]) if "." in s else 0


def _exact_equal(a: Num, b: Num) -> bool:
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1e-12)


def _consistent(correct: Num, stated: Num, stated_expr: Expr, scale: Fraction) -> bool:
    """Is `stated` an acceptable statement of `correct`?

    Exact match always passes.  If the stated side is a bare literal with d
    decimals, it may be the *rounded* (nearest) or *truncated* value of
    `correct` -- but only when `correct` is NOT itself exactly representable
    with d decimals (so 12*3 = 35 always fails, while 3/8*1900 = 712 passes,
    and 3.478 may be written 3 but not 4).
    """
    if _exact_equal(correct, stated):
        return True
    if stated_expr.literal is None:
        return False
    ulp = Fraction(1, 10 ** _decimals(stated_expr.literal.text)) * scale
    ratio = (Fraction(correct) if isinstance(correct, Fraction) else correct) / ulp
    if isinstance(ratio, Fraction):
        representable = ratio.denominator == 1
    else:
        representable = math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9)
    if representable:
        return False
    c, st, u = float(correct), float(stated), float(ulp)
    nearest = math.floor(c / u + 0.5) * u
    trunc = math.trunc(c / u) * u
    tol = u * 1e-6
    return abs(st - nearest) <= tol or abs(st - trunc) <= tol


@dataclass
class Check:
    lhs: str                    # source text of the left expression
    rhs: str                    # source text of the right expression
    lhs_span: tuple[int, int]
    rhs_span: tuple[int, int]
    stated_side: str            # 'lhs' or 'rhs'
    correct_value: float        # z_i
    stated_value: float         # z~_i
    ok: bool
    error: float                # e_i (0.0 when ok)
    interpretation: Optional[str]   # which reading was used (diagnostics)
    stated_literal_span: Optional[tuple[int, int]]  # span of the stated number (for corruption)
    stated_literal_text: Optional[str]
    clause_index: int
    chain_position: int         # index of this '=' within its chain

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class RationaleReport:
    checks: list[Check] = field(default_factory=list)
    n_equals: int = 0           # total '=' signs seen (coverage diagnostics)
    artifacts: list[str] = field(default_factory=list)  # layout/scrape damage signals

    @property
    def m(self) -> int:
        return len(self.checks)

    @property
    def k(self) -> int:
        return sum(not c.ok for c in self.checks)

    @property
    def contradicted(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def q(self) -> float:
        return reliability_score(self)


def _pct_modes(e: Expr):
    return ("strip", "pct") if e.has_percent else ("strip",)


def compare(left: Expr, right: Expr, text: str, clause_index: int, pos: int) -> Optional[Check]:
    """Build a Check for `left = right`, or None if the pair isn't a checkable step.

    If `right` has an alternative reading (see extract_right), the pair passes
    if either reading is consistent.
    """
    results = []
    for r in ([right, right.alt] if right.alt is not None else [right]):
        chk = _compare_one(left, r, text, clause_index, pos)
        if chk is not None:
            if chk.ok:
                return chk
            results.append(chk)
    return min(results, key=lambda c: c.error) if results else None


def _compare_one(left: Expr, right: Expr, text: str, clause_index: int, pos: int) -> Optional[Check]:
    if not (left.has_operation or right.has_operation):
        return None  # e.g. "20 = 20", "50% = 0.5": not an arithmetic step
    # unit conversion guard: "72 kmph = 72*5/18 (m/s)" -- left carries a unit,
    # right has no/different unit -> skip
    if left.unit and left.unit != right.unit:
        return None
    if text[left.start:left.end].replace(" ", "") == text[right.start:right.end].replace(" ", ""):
        return None  # "(25)x(1829) = (25)x(1829)": duplicated layout, not a step
    # which side states the result?
    if right.literal is not None:
        stated_e, correct_e, side = right, left, "rhs"
    elif left.literal is not None:
        stated_e, correct_e, side = left, right, "lhs"
    else:
        stated_e, correct_e, side = right, left, "rhs"

    # Try every interpretation (standard reading first).  The step is OK if
    # ANY reading is consistent; otherwise e_i is the smallest error over
    # readings evaluated under the *standard* slash reading (falling back to
    # any reading if the standard one is unparseable).
    best_ok, best_err, best_vals, interp = False, math.inf, None, None
    for cm in _interpretations(correct_e):
        for sm in _interpretations(stated_e):
            try:
                c = evaluate(correct_e, *cm)
                s = evaluate(stated_e, *sm)
            except (EvalError, ZeroDivisionError, OverflowError, ValueError):
                continue
            scale = Fraction(1, 100) if (sm[0] == "pct" and stated_e.has_percent) else Fraction(1)
            ok = _consistent(c, s, stated_e, scale)
            signflip = False
            if not ok and c != 0 and _consistent(-c, s, stated_e, scale):
                # "1990 - 1994 = 4", "100 - 112 = 12%": magnitude of a difference
                ok, signflip = True, True
            if ok:
                best_ok, best_err, best_vals = True, 0.0, (c, s)
                interp = f"{cm[0]}/{cm[1]}|{sm[0]}/{sm[1]}" + ("|abs" if signflip else "")
                break
            standard = cm[1] == "standard" and sm[1] == "standard"
            err = min(1.0, abs(float(s) - float(c)) / (abs(float(c)) + EPS))
            # prefer standard-reading errors for e_i
            key = err if standard else err + 10.0
            if key < best_err:
                best_err, best_vals = key, (c, s)
                interp = f"{cm[0]}/{cm[1]}|{sm[0]}/{sm[1]}"
        if best_ok:
            break
    if best_err >= 10.0 and best_err != math.inf:
        best_err -= 10.0
    if best_vals is None:
        return None  # unparseable under every interpretation -> uncheckable
    c, s = best_vals
    lit = stated_e.literal
    return Check(
        lhs=text[left.start:left.end], rhs=text[right.start:right.end],
        lhs_span=left.text_span, rhs_span=right.text_span,
        stated_side=side, correct_value=float(c), stated_value=float(s),
        ok=best_ok, error=best_err, interpretation=interp,
        stated_literal_span=(lit.start, lit.end) if lit is not None else None,
        stated_literal_text=lit.text if lit is not None else None,
        clause_index=clause_index, chain_position=pos,
    )


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def check_rationale(text: str) -> RationaleReport:
    """Extract and verify all checkable arithmetic equalities in `text`.

    `text` should be the output of clean_rationale(); all spans index into it.
    """
    toks = tokenize(text)
    report = RationaleReport()
    report.n_equals = sum(t.kind == "EQ" for t in toks)
    report.artifacts = detect_artifacts(text, toks)
    for ci, (clause, start) in enumerate(split_clauses_with_starts(toks)):
        segs = split_segments(clause)
        if len(segs) < 2:
            continue
        exprs: list[Optional[Expr]] = []
        last = len(segs) - 1
        seg_texts = ["".join(t.text for t in sg if t.kind != "WS") for sg in segs]
        for si in range(last):
            if seg_texts[si] and seg_texts[si] == seg_texts[si + 1] and any(
                    t.kind == "OP" for t in segs[si]) and "duplicated_expression" not in report.artifacts:
                report.artifacts.append("duplicated_expression")
        for si, seg in enumerate(segs):
            if si == 0:
                exprs.append(extract_left(seg, start))
            elif si == last:
                exprs.append(extract_right(seg))
            else:
                exprs.append(extract_middle(seg))
        for si in range(last):
            a, b = exprs[si], exprs[si + 1]
            if a is None or b is None:
                continue
            chk = compare(a, b, text, ci, si)
            if chk is not None:
                report.checks.append(chk)
    return report


_LOST_SUPERSCRIPT_RE = re.compile(r"(?<![A-Za-z0-9])[a-z][2-9](?![0-9])(?=\s*[-+=)*/]|\s*$)")


def detect_artifacts(text: str, toks: Optional[list[Tok]] = None) -> list[str]:
    """Signals that the rationale's math layout was damaged when scraped.

    These rationales are unreliable *as text* even when the checker can't
    point at a specific wrong equality, so the dataset filter excludes them.
    """
    toks = toks if toks is not None else tokenize(text)
    out = []
    if "\t" in text:
        out.append("tab_layout")          # 2-D fraction layout flattened with tabs
    for t in toks:
        if t.kind == "NUM":
            d = t.text.replace(",", "").replace(".", "")
            if len(d) >= 4 and len(d) % 2 == 0 and d[: len(d) // 2] == d[len(d) // 2:]:
                out.append("duplicated_number")   # MathJax "1829=1829" duplication
                break
    if _LOST_SUPERSCRIPT_RE.search(text):
        out.append("lost_superscript")    # "a3 + b2", "x2 - 225"
    if "\ufffd" in text or "Â" in text or "â€" in text:
        out.append("mojibake")
    return out


def reliability_score(report: RationaleReport) -> float:
    """q(r) from Eq. (2) of the SAVRR proposal."""
    m, bad = report.m, report.contradicted
    k = len(bad)
    if k == 0:
        return 1.0
    mean_e = sum(c.error for c in bad) / k
    return 1.0 - 0.5 * (k / m + mean_e)


def analyze(text: str, clean: bool = True) -> RationaleReport:
    return check_rationale(clean_rationale(text) if clean else text)


if __name__ == "__main__":  # quick manual probe
    import sys
    src = sys.stdin.read() if not sys.stdin.isatty() else (
        "Total age = 25 x 11 = 275\nAge of coach = 312-275 = 37 yrs.\n"
        "Total votes = (3408 + 2500 + 5050 + 3400)\n= 61200.")
    rep = analyze(src)
    for c in rep.checks:
        mark = "OK " if c.ok else "BAD"
        print(f"[{mark}] {c.lhs!r} = {c.rhs!r}  correct={c.correct_value:g} stated={c.stated_value:g} e={c.error:.3f}")
    print(f"m={rep.m} k={rep.k} q={rep.q:.3f}")
