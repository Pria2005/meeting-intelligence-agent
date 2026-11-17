"""Evaluation metrics from PRD §15: WER, DER (frame-based approximation), extraction P/R/F1, Recall@K, MRR,
and a simple citation-accuracy score. Pure Python, no extra dependencies."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from app.services.textutil import coverage


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words / number of reference words."""
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def diarization_error_rate(reference: Sequence[Tuple[float, float, str]], hypothesis: Sequence[Tuple[float, float, str]],
                           step: float = 0.1) -> float:
    """Frame-based DER (missed + false-alarm + confusion) with a greedy optimal speaker mapping.
    Turns are ``(start, end, speaker)``. An approximation of pyannote.metrics, adequate for small evaluation sets."""
    if not reference:
        return 0.0
    end = max(max(t[1] for t in reference), max((t[1] for t in hypothesis), default=0.0))
    n = int(end / step) + 1

    def label_at(turns, t):
        for s, e, spk in turns:
            if s <= t < e:
                return spk
        return None

    pairs: Dict[Tuple[str, str], int] = {}
    ref_frames, hyp_frames = [], []
    for k in range(n):
        t = k * step
        r, h = label_at(reference, t), label_at(hypothesis, t)
        ref_frames.append(r)
        hyp_frames.append(h)
        if r and h:
            pairs[(r, h)] = pairs.get((r, h), 0) + 1
    mapping: Dict[str, str] = {}
    used: Set[str] = set()
    for (r, h), _ in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if r not in mapping and h not in used:
            mapping[r] = h
            used.add(h)
    total = sum(1 for r in ref_frames if r)
    if total == 0:
        return 0.0
    errors = 0
    for r, h in zip(ref_frames, hyp_frames):
        if r and not h:
            errors += 1  # missed speech
        elif h and not r:
            errors += 1  # false alarm
        elif r and h and mapping.get(r) != h:
            errors += 1  # speaker confusion
    return errors / total


def precision_recall_f1(predicted: Iterable[str], gold: Iterable[str], threshold: float = 0.6) -> Dict[str, float]:
    """Fuzzy set matching: a predicted item matches a gold item if their key-term coverage >= threshold."""
    pred, ref = list(predicted), list(gold)
    matched_gold: Set[int] = set()
    tp = 0
    for p in pred:
        best, best_i = 0.0, -1
        for i, g in enumerate(ref):
            if i in matched_gold:
                continue
            score = max(coverage(p, g), coverage(g, p))
            if score > best:
                best, best_i = score, i
        if best >= threshold:
            tp += 1
            matched_gold.add(best_i)
    precision = tp / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = tp / len(ref) if ref else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "predicted": len(pred), "gold": len(ref)}


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    for i, cid in enumerate(ranked_ids, 1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def citation_accuracy(answer_claims: Sequence[Tuple[str, str]], threshold: float = 0.5) -> float:
    """Share of (claim, cited_text) pairs where the cited text actually covers the claim."""
    if not answer_claims:
        return 0.0
    return sum(1 for claim, cited in answer_claims if coverage(claim, cited) >= threshold) / len(answer_claims)
