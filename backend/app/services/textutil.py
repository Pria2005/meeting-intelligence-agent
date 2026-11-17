"""Small text helpers shared by chunking, verification, RAG and the offline LLM."""
from __future__ import annotations

import re
from typing import Iterable, List, Set

STOPWORDS: Set[str] = set("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i
if in into is it its itself just let me more most my no nor not now of off on once only or other our ours out over own
same she should so some such than that the their theirs them then there these they this those through to too under
until up us very was we were what when where which while who whom why will with would you your yours yeah okay ok um uh
like really going get got one think know say said well right yes sure
""".split())

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def tokenize(text: str) -> List[str]:
    return [w.lower().strip("'-") for w in _WORD.findall(text or "")]


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def stem(token: str) -> str:
    """Very light suffix stripping so 'deployed'/'deploying'/'deploy' match."""
    for suf in ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    if token.endswith("e") and len(token) > 4:  # decide / decided -> decid
        return token[:-1]
    return token


def stems(text: str) -> Set[str]:
    return {stem(t) for t in content_tokens(text)}


def coverage(claim: str, evidence: str) -> float:
    """Fraction of the claim's content stems found in the evidence (0..1)."""
    c = stems(claim)
    if not c:
        return 0.0
    return len(c & stems(evidence)) / len(c)


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds or 0)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out
