"""Final-answer scorer for GSM8K and AIME style problems.

Version ``math-answer-v4``. Pure Python, no third-party imports, so the same
file runs inside the Miles training image, the baseline evaluation image and
on a laptop.

The scorer reads the model's answer section (the text after the last
``</think>``; the whole response for a model that does not think), extracts
the answer the model committed to, normalises it and compares it with the
label. It never searches the reasoning for a matching number. Every decision
is returned as a structured ``ScoreResult`` so audits can count why a
response scored what it scored.

Selection order:

1. ``boxed``      the last ``\\boxed{...}`` / ``\\fbox{...}`` in the answer
                  section decides, whatever else the text says.
2. ``bare_line``  the last non-empty line is nothing but a number
                  (``...takes 204 minutes.\\n\\n204``).
3. the latest of ``stated`` (``Answer: 8``, ``The final answer is 8``,
   ``**Answer:** Kerry is **8** years old``, ``**Answer:**`` + next line),
   ``bold`` (``Kerry is **8** years old``, ``\\mathbf{7}``; headings such as
   ``**Step 2:**`` and ``**Case 1**`` are ignored) and ``last_line`` (the
   last number on a last line that reads as a conclusion or holds a single
   number; trailing verification, note and drawing blocks are skipped). Position in the
   text decides between them because models restate the answer last; on the
   same line an explicit statement beats bold beats a plain number.

A response cut off by the length limit (``truncated=True``) can only score
through ``boxed`` and ``stated``: a number that merely happens to end a
truncated text is not a committed answer. A response whose thinking never
closes has no answer section and scores 0 with reason ``no_answer_section``.

Scores are 0 or 1.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction

VERSION = "math-answer-v4"

_SPECIAL_TOKEN = re.compile(r"<\|[^<>|]{1,40}\|>")
_THINK_CLOSE = "</think>"
_THINK_OPEN = "<think>"

# "\x08oxed": a "\b" that an upstream JSON layer decoded into a backspace.
_BOXED_CMD = re.compile(r"(?:\\(?:boxed|fbox)|\x08oxed)\s*")
_BOXED_BARE = re.compile(r"(?:\\(?:boxed|fbox)|\x08oxed)\s+([^\s${}\\]+)")
# "Answer:" / "Final answer =" must start the line (after list/heading markup);
# "the (final) answer is" may appear mid-line.
_STATED_LINE_START = re.compile(
    r"^[\s>*#\-\d.)]*(?:\*\*)?\s*(?:the\s+)?(?:final\s+)?answer\s*(?:\*\*)?\s*(?::|=|is)\s*(?:\*\*)?\s*(?::\s*)?(.*)$",
    re.IGNORECASE,
)
_STATED_MID = re.compile(r"\b(?:the\s+)?(?:final\s+)?answer\s+is\s*(?:\*\*)?\s*:?\s*(.*)$", re.IGNORECASE)
_BOLD = re.compile(r"\*\*([^*\n]{1,200}?)\*\*|\\(?:mathbf|textbf)\s*{([^{}\n]{1,200})}")
_NUMBER = re.compile(r"[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?")
_CONCLUSION_WORDS = re.compile(
    r"\b(?:therefore|thus|so|hence|total|answer|is|are|be|equals?|gets?|needs?|has|have|will|remainder|value)\b|=",
    re.IGNORECASE,
)
_HEADING_LIKE = re.compile(r"(?::\s*$)|^(?:step|case|part|section)\b", re.IGNORECASE)
_LIST_STEP = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*(?:\*\*)?\s*(?:step|case)?", re.IGNORECASE)
_ENV_LINE = re.compile(r"^\s*\\(?:begin|end)\s*{[a-zA-Z*]+}\s*$")
# Text after the answer that restates the work: verification blocks, notes,
# breakdown tables and drawings. Ignored when an answer precedes them.
_TRAILING_HEADING = re.compile(
    r"^\s*(?:\*\*|#+\s*|\*\(|\()?\s*(?:verification|verify|check(?:ing)?|double-check|breakdown|note|explanation|summary\s+of\s+(?:the\s+)?steps|why\s+this\s+works)\b",
    re.IGNORECASE,
)
_TRAILING_ENV = re.compile(r"^\s*\\begin\s*{(?:tikzpicture|figure|table|tabular|asy)}")
_BARE_LINE = re.compile(
    r"^\s*(?:\*\*|\\\(|\\\[|\$+|\\text\s*{|\\(?:boxed|fbox)\s*{)?\s*"
    r"(?P<num>[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?)"
    r"\s*(?:\*\*|\\\)|\\\]|\$+|})?\s*[.。]?\s*$"
)
_UNIT_TAIL = re.compile(
    r"^(?P<num>[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?)\s*"
    r"(?:%|\\%|°|\^\\circ|\^{\\circ}|\\circ)?\s*"
    r"(?P<unit>[a-zA-Z][a-zA-Z .\-]*)?$"
)


@dataclass
class ScoreResult:
    score: int
    reason: str
    tier: str | None = None
    extracted: str | None = None
    normalized: str | None = None
    label_normalized: str | None = None
    answer_section: bool = False
    thinking_closed: bool | None = None
    truncated: bool = False
    ambiguous: bool = False
    candidates: list[str] = field(default_factory=list)
    version: str = VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def strip_special_tokens(text: str) -> str:
    return _SPECIAL_TOKEN.sub("", text)


def answer_section(response: str, *, thinking_opened_in_prompt: bool = True) -> tuple[str | None, bool | None]:
    """Return (answer text or None, thinking_closed); thinking_closed is None without a think block."""
    text = strip_special_tokens(response or "")
    if _THINK_CLOSE in text:
        return text.rsplit(_THINK_CLOSE, 1)[1], True
    if _THINK_OPEN in text or thinking_opened_in_prompt:
        return None, False
    return text, None


def _balanced_group(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def boxed_candidates(text: str) -> list[str]:
    out: list[str] = []
    for m in _BOXED_CMD.finditer(text):
        inner = _balanced_group(text, m.end())
        if inner is not None:
            out.append(inner.strip())
    for m in _BOXED_BARE.finditer(text):
        out.append(m.group(1).strip())
    return out


def _with_sign(phrase: str, num: str) -> str:
    i = phrase.rfind(num)
    if i > 0 and phrase[i - 1] in "-−" and (i == 1 or not phrase[i - 2].isdigit()):
        return "-" + num
    return num


def _number_in_phrase(phrase: str) -> str | None:
    """The answer number in a short phrase: a bold number inside wins, else the last number."""
    bold = [b for b in (m.group(1) or m.group(2) for m in _BOLD.finditer(phrase)) if _NUMBER.search(b)]
    if bold:
        return _with_sign(bold[-1], _NUMBER.findall(bold[-1])[-1])
    nums = _NUMBER.findall(phrase)
    return _with_sign(phrase, nums[-1]) if nums else None


def _content_lines(text: str) -> list[tuple[int, str]]:
    """(index, line) for non-empty lines that are not LaTeX environment delimiters."""
    return [(i, ln) for i, ln in enumerate(text.splitlines()) if ln.strip() and not _ENV_LINE.match(ln)]


def _stated_tail(lines: list[str], i: int, tail: str) -> str | None:
    if not re.search(r"\d", tail):
        # "**Answer:**" with the answer on the next line; never a list step.
        for nxt in lines[i + 1 : i + 3]:
            if nxt.strip():
                if _LIST_STEP.match(nxt) and not _BARE_LINE.match(nxt):
                    return None
                tail = nxt.strip()
                break
        else:
            return None
        if not re.search(r"\d", tail):
            return None
    tail = re.split(r"(?<=[^\d.])\.(?:\s|$)|;", tail, maxsplit=1)[0].strip()
    if "boxed" in tail or "fbox" in tail:
        cands = boxed_candidates(tail)
        if cands:
            return cands[-1]
    if _BARE_LINE.match(tail):
        return tail
    return _number_in_phrase(tail)


def positioned_candidates(text: str) -> list[tuple[int, int, str, str]]:
    """(line index, rank within line, tier, candidate) for stated, bold and last-line tiers."""
    lines = text.splitlines()
    content = _content_lines(text)
    out: list[tuple[int, int, str, str]] = []
    for i, line in content:
        m = _STATED_LINE_START.match(line) or _STATED_MID.search(line)
        if m:
            cand = _stated_tail(lines, i, m.group(1).strip())
            if cand is not None:
                out.append((i, 2, "stated", cand))
        for bm in _BOLD.finditer(line):
            span = (bm.group(1) or bm.group(2)).strip()
            if _HEADING_LIKE.search(span) or not _NUMBER.search(span):
                continue
            num = _number_in_phrase(span)
            if num is not None:
                out.append((i, 1, "bold", num))
    if content:
        i, line = content[-1]
        one_number = len(_NUMBER.findall(line)) == 1
        if len(line.split()) <= 60 and (_CONCLUSION_WORDS.search(line) or one_number):
            num = _number_in_phrase(line)
            if num is not None:
                out.append((i, 0, "last_line", num))
    return out


def bare_line_candidate(text: str) -> str | None:
    content = _content_lines(text)
    if not content:
        return None
    m = _BARE_LINE.match(content[-1][1])
    return m.group("num") if m else None


def _strip_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\(?:text|textbf|mathrm|mathbf|mathit|operatorname)\s*{([^{}]*)}", r"\1", s)
    s = re.sub(r"\\[dt]?frac\s*{([^{}]*)}\s*{([^{}]*)}", r"\1/\2", s)
    s = re.sub(r"\\(?:left|right|,|;|!|quad|qquad)", "", s)
    s = s.replace("\\$", "").replace("\\%", "%").replace("\\times", "*")
    s = re.sub(r"^\s*(?:\\\(|\\\[|\$+)\s*", "", s)
    s = re.sub(r"\s*(?:\\\)|\\\]|\$+)\s*$", "", s)
    s = s.replace("$", "")
    s = re.sub(r"\^\s*{?\\circ}?|°", "", s)
    s = s.replace("−", "-").replace("–", "-")
    return s.strip()


def normalize_answer(raw: str | int | float | None) -> str | None:
    """Return a canonical string for comparison, or None when empty."""
    if raw is None:
        return None
    s = _strip_latex(str(raw))
    s = s.strip().strip("*").strip()
    s = re.sub(r"^(?:[≈~=:]|\\approx)\s*", "", s)
    s = re.sub(r"[.。!]+$", "", s).strip()
    s = re.sub(r"^\$\s*", "", s)
    if not s:
        return None
    m = _UNIT_TAIL.match(s)
    if m:
        s = m.group("num")
    s = s.replace(",", "").replace(" ", "").replace("−", "-")
    if "/" in s:
        a, _, b = s.partition("/")
        try:
            return str(Fraction(int(a), int(b)))
        except (ValueError, ZeroDivisionError):
            return s.lower()
    try:
        f = Fraction(s)
        return str(f.numerator) if f.denominator == 1 else str(f)
    except (ValueError, ZeroDivisionError):
        return s.lower()


def _answers_match(candidate: str | None, label: str | None) -> bool:
    if candidate is None or label is None:
        return False
    if candidate == label:
        return True
    try:
        return Fraction(candidate) == Fraction(label)
    except (ValueError, ZeroDivisionError):
        return False


def trim_trailing_sections(text: str) -> str:
    """Drop a trailing verification/note/drawing block when an answer precedes it."""
    lines = text.splitlines()
    cut = None
    for i, ln in enumerate(lines):
        if _TRAILING_HEADING.match(ln) or _TRAILING_ENV.match(ln) or ln.strip() == "***":
            if cut is None:
                cut = i
        elif cut is not None and not (_TRAILING_HEADING.match(ln) or ln.startswith((" ", "\t", "*", "-", "$", "\\", "|", "(")) or not ln.strip()):
            cut = None  # ordinary prose resumed; the block was not trailing
    if cut is None or cut == 0:
        return text
    head = "\n".join(lines[:cut])
    if bare_line_candidate(head) is None and not positioned_candidates(head):
        return text
    return head


def extract(text: str, *, truncated: bool = False) -> tuple[str | None, list[str]]:
    """Return (tier, candidates); the last candidate is the chosen one."""
    boxed = boxed_candidates(text)
    if boxed:
        return "boxed", boxed
    text = trim_trailing_sections(text)
    pos = positioned_candidates(text)
    if truncated:
        stated = [c for c in pos if c[2] == "stated"]
        return ("stated", [c[3] for c in stated]) if stated else (None, [])
    bare = bare_line_candidate(text)
    if bare is not None:
        return "bare_line", [bare]
    if not pos:
        return None, []
    pos.sort(key=lambda c: (c[0], c[1]))
    tier = pos[-1][2]
    return tier, [c[3] for c in pos if c[2] == tier]


def score(
    response: str,
    label: str | int | float,
    *,
    truncated: bool = False,
    thinking_opened_in_prompt: bool = True,
) -> ScoreResult:
    """Score one response against its label; see the module docstring for the rules."""
    label_norm = normalize_answer(label)
    section, closed = answer_section(response, thinking_opened_in_prompt=thinking_opened_in_prompt)
    if section is None:
        return ScoreResult(0, "no_answer_section", thinking_closed=closed, truncated=truncated, label_normalized=label_norm)
    section = section.strip()
    if not section:
        return ScoreResult(0, "empty_answer_section", answer_section=True, thinking_closed=closed, truncated=truncated, label_normalized=label_norm)
    tier, cands = extract(section, truncated=truncated)
    if tier is None:
        reason = "truncated_no_commitment" if truncated else "no_candidate"
        return ScoreResult(0, reason, answer_section=True, thinking_closed=closed, truncated=truncated, label_normalized=label_norm)
    normalized = [n for n in (normalize_answer(c) for c in cands) if n is not None]
    if not normalized:
        return ScoreResult(
            0, "unparseable_candidate", tier=tier, extracted=cands[-1], answer_section=True,
            thinking_closed=closed, truncated=truncated, label_normalized=label_norm, candidates=cands[-5:],
        )
    chosen = normalized[-1]
    ok = _answers_match(chosen, label_norm)
    return ScoreResult(
        1 if ok else 0,
        "match" if ok else "mismatch",
        tier=tier,
        extracted=cands[-1],
        normalized=chosen,
        label_normalized=label_norm,
        answer_section=True,
        thinking_closed=closed,
        truncated=truncated,
        ambiguous=len(set(normalized)) > 1,
        candidates=cands[-5:],
    )


# --- Miles reward-model adapters ------------------------------------------
# ``--custom-rm-path <module>.math_scorer.miles_batched_reward`` makes Miles
# call the batched entry point with the whole rollout batch; the per-sample
# entry point serves the multi-LoRA path, which calls ``async_rm`` per sample.


def _sample_flags(sample) -> tuple[bool, bool]:
    md = getattr(sample, "metadata", None)
    md = md if isinstance(md, dict) else {}
    truncated = "truncated" in str(getattr(sample, "status", "")).lower()
    return truncated, bool(md.get("thinking_opened_in_prompt", True))


async def miles_reward(args, sample, **kwargs) -> float:
    truncated, thinking = _sample_flags(sample)
    result = score(sample.response, sample.label, truncated=truncated, thinking_opened_in_prompt=thinking)
    md = getattr(sample, "metadata", None)
    if isinstance(md, dict):
        d = result.to_dict()
        md["scorer"] = {k: d[k] for k in ("reason", "tier", "extracted", "normalized", "ambiguous", "truncated", "version")}
    return float(result.score)


async def miles_batched_reward(args, samples, **kwargs) -> list[float]:
    return [await miles_reward(args, s, **kwargs) for s in samples]
