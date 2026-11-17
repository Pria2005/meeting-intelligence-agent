"""Rule-based extraction used by the offline ``HeuristicLLM``.

Conservative by design: it only reports what matches explicit linguistic cues, leaves
owners / deadlines empty when they are not stated, and always attaches evidence."""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from app.schemas.extraction import (ActionItemOut, AnswerOut, ClaimJudgement, DecisionOut, RiskOut, SummaryOut,
                                    WindowExtraction)
from app.services.dtypes import ContextChunk, SegmentView
from app.services.textutil import (STOPWORDS, content_tokens, coverage, split_sentences, stem, stems, truncate)

GENERIC_SPEAKER = re.compile(r"^(speaker[ _]?\d*|unknown|spk[ _]?\d+)$", re.I)

DECISION_RE = re.compile(
    r"\b(we(?:'ve| have)? (?:decided|agreed|concluded|settled)|we(?:'ll| will| are going to|'re going to) (?:go with|use|adopt|switch|move|launch|ship|keep|stick)"
    r"|let'?s (?:go with|use|move forward with|stick with|adopt|ship|launch)|the decision is|final decision|decision:|it'?s decided"
    r"|(?:is|was|been) (?:approved|agreed|confirmed)|agreed (?:to|on|that)|we (?:chose|picked|selected)"
    r"|going with|settled on|moving forward with|signed off)\b", re.I)

ACTION_RE = re.compile(
    r"\b(i(?:'ll| will)|i(?:'m| am) going to|can you|could you|will you|please|needs? to|need to|has to|have to|action item|to-?do"
    r"|take care of|responsible for|follow up|reach out|get back to|(?:will|to|should|must) (?:send|prepare|write|draft|review|schedule|update"
    r"|create|build|fix|share|finish|complete|deliver|set up|check|contact|book|publish|test|deploy|investigate|document|submit|present|email|call|plan|design|implement|migrate|confirm|collect|gather|organi[sz]e))\b",
    re.I)

OWNER_SUBJECT_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|so|and|then|great|thanks|alright)[,.]?\s+)*(?P<name>[A-Z][a-z]{1,20}(?:\s+(?:and|&)\s+[A-Z][a-z]{1,20})?)\s+"
    r"(?:will|to|is going to|can|should|needs to|has to|agreed to|volunteered to|is responsible for)\b")
ADDRESS_RE = re.compile(r"^(?:ok(?:ay)?[,.]?\s+|so[,.]?\s+)?(?P<name>[A-Z][a-z]{1,20}),?\s+(?:can|could|will|would) you\b")
NOT_NAMES = {"We", "I", "You", "They", "It", "He", "She", "This", "That", "There", "The", "Someone", "Everyone", "Who",
             "What", "When", "Where", "How", "Why", "Please", "Let", "Okay", "Ok", "So", "And", "But", "Yes", "No", "Team",
             "Next", "Also", "Then", "Once", "If", "Maybe"}

_DAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DUE_RE = re.compile(
    rf"\b(?:by|before|until|due|on|for)\s+((?:next\s+|this\s+)?{_DAY}|tomorrow|tonight|today|(?:the\s+)?end of (?:the\s+)?(?:day|week|month|quarter|year|sprint)"
    rf"|eod|eow|next (?:week|month|quarter|sprint)|this (?:week|month|sprint)"
    rf"|{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}(?:,?\s+\d{{4}})?"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?)\b", re.I)
DUE_BARE_RE = re.compile(r"\b(next week|next month|tomorrow|end of (?:the )?(?:day|week|month)|by eod|by eow)\b", re.I)

RISK_RE = re.compile(
    r"\b(risk(?:y|s)?|blocker|blocked|blocking|stuck|concern(?:ed|s)?|worried|worry|delay(?:ed|s)?|behind schedule|at risk|bottleneck|"
    r"might (?:not|fail|slip|break)|may not|could (?:fail|slip|break)|problem|issue|dependency|dependent on|not enough|shortage|"
    r"over budget|can'?t (?:afford|finish|make)|won'?t (?:be able|make|finish))\b", re.I)
BLOCKER_RE = re.compile(r"\b(blocker|blocked|blocking|stuck|waiting (?:on|for))\b", re.I)
HIGH_RE = re.compile(r"\b(critical|severe|major|serious|urgent|show-?stopper|high risk|huge)\b", re.I)
OPEN_Q_RE = re.compile(
    r"(\?\s*$)|\b(open question|unclear|not sure|don'?t know|haven'?t decided|yet to (?:decide|determine)|tbd|to be determined|"
    r"still need to (?:figure|decide|determine)|undecided|figure out)\b", re.I)
QUESTION_CUE_RE = re.compile(
    r"\b(should we|do we|are we|how (?:do|will|should|are|can) we|who (?:will|is|should|can)|when (?:will|do|should|can)|what (?:about|if|is the plan)|"
    r"which|can we|is there|not sure|unclear|open question|don'?t know|haven'?t decided|yet to|tbd|figure out|undecided)\b", re.I)
PURPOSE_RE = re.compile(r"\b(purpose|goal|agenda|today we|meeting is (?:about|to)|we(?:'re| are) here to|objective|"
                        r"going to (?:talk|discuss|review|go over)|let'?s (?:talk|discuss|review|go over|start))\b", re.I)


ACTION_VERB_RE = re.compile(
    r"\b(send|prepare|write|draft|review|schedule|update|create|build|fix|share|finish|complete|deliver|set up|check|contact|"
    r"book|publish|test|deploy|investigate|document|submit|present|email|call|plan|design|implement|migrate|confirm|collect|"
    r"gather|organi[sz]e|handle|follow up|take care|look into|reach out|arrange|own|lead|run)\b", re.I)
NOT_A_TASK_RE = re.compile(r"\b(give us|tell us|walk us|update us|fill us|share your|quick (?:status|update)|any (?:update|thoughts)|your thoughts)\b", re.I)
FIRST_PERSON_RE = re.compile(r"\bi(?:'ll| will| need to| have to| should| must|'m going to| am going to)\b", re.I)
NEGATED_RISK_RE = re.compile(r"\b(no longer|not (?:a|an|any)\b|isn'?t (?:a|an)\b|aren'?t (?:a|an)?|no (?:more )?(?:risks?|blockers?|concerns?|issues?|problems?)|resolved|fixed now)\b", re.I)


def _is_real_name(label: Optional[str]) -> bool:
    return bool(label) and not GENERIC_SPEAKER.match(label.strip())


def _sentences(seg: SegmentView) -> List[str]:
    return split_sentences(seg.text) or [seg.text]


def _find_due(sentence: str) -> Optional[str]:
    m = DUE_RE.search(sentence)
    if m:
        return m.group(1).strip()
    m = DUE_BARE_RE.search(sentence)
    return m.group(1).strip() if m else None


def _find_owner(sentence: str, seg: SegmentView, prev: Optional[SegmentView]) -> Optional[str]:
    if FIRST_PERSON_RE.search(sentence) and not re.search(r"\b(you|we) (?:need|have|should|will)\b", sentence, re.I):
        return seg.speaker if _is_real_name(seg.speaker) else None
    m = ADDRESS_RE.match(sentence)
    if m and m.group("name") not in NOT_NAMES:
        return m.group("name")
    m = OWNER_SUBJECT_RE.match(sentence)
    if m:
        name = m.group("name")
        if name.split()[0] not in NOT_NAMES:
            return name
    return None


def _clean_task(sentence: str) -> str:
    s = sentence.strip()
    s = re.sub(r"^(?:ok(?:ay)?|so|and|then|alright|great|yes|sure)[,.]?\s+", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s[0].upper() + s[1:] if s else s


def extract(segments: Sequence[SegmentView]) -> WindowExtraction:
    decisions: List[DecisionOut] = []
    actions: List[ActionItemOut] = []
    risks: List[RiskOut] = []
    questions: List[RiskOut] = []
    seen: set = set()

    def _new(kind: str, text: str) -> bool:
        key = (kind, " ".join(sorted(stems(text))))
        if key in seen:
            return False
        seen.add(key)
        return True

    prev: Optional[SegmentView] = None
    for seg in segments:
        for sent in _sentences(seg):
            words = len(sent.split())
            if words < 4:
                continue
            is_q = sent.rstrip().endswith("?")
            if DECISION_RE.search(sent) and not is_q and _new("d", sent):
                decisions.append(DecisionOut(text=_clean_task(sent), evidence_segment_ids=[seg.ref]))
                continue
            is_action = (ACTION_RE.search(sent) and not is_q and not PURPOSE_RE.search(sent)) or (
                is_q and ADDRESS_RE.match(sent) and ACTION_VERB_RE.search(sent) and not NOT_A_TASK_RE.search(sent))
            if is_action:
                if _new("a", sent):
                    actions.append(ActionItemOut(task=_clean_task(sent), owner=_find_owner(sent, seg, prev),
                                                 due_date=_find_due(sent), evidence_segment_ids=[seg.ref]))
                continue
            if RISK_RE.search(sent) and not NEGATED_RISK_RE.search(sent) and not is_q and _new("r", sent):
                kind = "blocker" if BLOCKER_RE.search(sent) else "risk"
                sev = "high" if HIGH_RE.search(sent) else None
                risks.append(RiskOut(text=_clean_task(sent), kind=kind, severity=sev, evidence_segment_ids=[seg.ref]))
                continue
            if OPEN_Q_RE.search(sent) and QUESTION_CUE_RE.search(sent) and _new("q", sent):
                questions.append(RiskOut(text=_clean_task(sent), kind="open_question", evidence_segment_ids=[seg.ref]))
        prev = seg

    summary = _summarise(segments, decisions, actions, risks)
    return WindowExtraction(summary=summary, decisions=decisions[:25], action_items=actions[:40],
                            risks=risks[:25], open_questions=questions[:20])


# ------------------------------------------------------------------ summary
GREETING_RE = re.compile(r"^(?:good (?:morning|afternoon|evening)|hello|hi|hey|welcome|thanks|thank you|perfect|great|okay|ok)\b", re.I)
TOPIC_STOP = {"team", "everyone", "today", "thing", "things", "week", "month", "day", "time", "meeting", "morning", "lot",
              "way", "work", "part", "need", "make", "want", "look", "take", "first", "second", "third", "also", "still",
              "something", "anyone", "someone", "number", "point", "piece", "side", "plan", "folder", "status"}
NUMBERISH_RE = re.compile(r"^(?:\w+th|\w+ty|\w+teen|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|percent|twelve|eleven)$", re.I)


def _rank_sentences(segments: Sequence[SegmentView]) -> List[Tuple[float, int, str]]:
    all_sents: List[Tuple[int, str]] = []
    for seg in segments:
        for s in _sentences(seg):
            if len(s.split()) >= 6 and not s.rstrip().endswith("?") and not GREETING_RE.match(s):
                all_sents.append((len(all_sents), s))
    freq = Counter(stem(t) for _, s in all_sents for t in content_tokens(s))
    ranked = []
    for order, s in all_sents:
        toks = [stem(t) for t in content_tokens(s)]
        if not toks:
            continue
        score = sum(freq[t] for t in toks) / (len(toks) ** 0.7)
        ranked.append((score, order, s))
    return ranked


def _topics(segments: Sequence[SegmentView], limit: int = 6) -> List[str]:
    text = " ".join(s.text for s in segments)
    freq: Counter = Counter()
    display: dict = {}
    toks = content_tokens(text)
    for t in toks:
        if len(t) < 4 or t in TOPIC_STOP or NUMBERISH_RE.match(t):
            continue
        st = stem(t)
        freq[st] += 1
        display.setdefault(st, t)
    # bigrams of adjacent content words give more topical phrases
    bigrams: Counter = Counter()
    for a, b in zip(toks, toks[1:]):
        if len(a) > 3 and len(b) > 3:
            bigrams[f"{a} {b}"] += 1
    topics = [bg.title() for bg, c in bigrams.most_common(4) if c >= 2]
    for st, c in freq.most_common(12):
        w = display[st].title()
        if c >= 2 and not any(w.lower() in t.lower() for t in topics):
            topics.append(w)
        if len(topics) >= limit:
            break
    return topics[:limit]


def _summarise(segments: Sequence[SegmentView], decisions, actions, risks) -> SummaryOut:
    if not segments:
        return SummaryOut(overview="No transcript content was available.")
    ranked = _rank_sentences(segments)
    top = sorted(sorted(ranked, reverse=True)[:5], key=lambda x: x[1])  # best five, back in speaking order
    key_points = [truncate(s, 220) for _, _, s in top]
    purpose = ""
    head = segments[: max(3, len(segments) // 5)]
    for seg in head:
        for s in _sentences(seg):
            if PURPOSE_RE.search(s):
                purpose = truncate(s, 240)
                break
        if purpose:
            break
    best = [s for _, _, s in sorted(sorted(ranked, reverse=True)[:2], key=lambda x: x[1])]
    lead = ([purpose] if purpose else []) + [s for s in best if s != purpose][: 1 if purpose else 2]
    overview = " ".join(truncate(s, 240) for s in (lead or key_points[:2]))
    parts = []
    if decisions:
        parts.append(f"{len(decisions)} decision{'s' if len(decisions) != 1 else ''}")
    if actions:
        parts.append(f"{len(actions)} action item{'s' if len(actions) != 1 else ''}")
    if risks:
        parts.append(f"{len(risks)} risk{'s' if len(risks) != 1 else ''}/blocker{'s' if len(risks) != 1 else ''}")
    outcome = ("The discussion produced " + ", ".join(parts) + ".") if parts else "No explicit decisions or action items were detected."
    return SummaryOut(overview=overview or key_points[0], purpose=purpose, main_topics=_topics(segments),
                      key_points=key_points, outcome=outcome)


def merge_summaries(partials: Sequence[SummaryOut]) -> SummaryOut:
    if not partials:
        return SummaryOut(overview="No summary available.")
    if len(partials) == 1:
        return partials[0]
    overview = " ".join(truncate(p.overview, 300) for p in partials[:3])
    topics: List[str] = []
    points: List[str] = []
    for p in partials:
        for t in p.main_topics:
            if t.lower() not in {x.lower() for x in topics}:
                topics.append(t)
        points.extend(p.key_points[:3])
    outcome = " ".join(p.outcome for p in partials if p.outcome)[:400]
    purpose = next((p.purpose for p in partials if p.purpose), "")
    return SummaryOut(overview=overview, purpose=purpose, main_topics=topics[:8], key_points=points[:10],
                      outcome=outcome)


# ------------------------------------------------------------------ answering
def answer(question: str, contexts: Sequence[ContextChunk]) -> AnswerOut:
    q_stems = stems(question)
    if not q_stems or not contexts:
        return AnswerOut(found=False, answer="", source_ids=[])
    scored: List[Tuple[float, str, ContextChunk]] = []
    for ctx in contexts:
        for s in split_sentences(ctx.text):
            cov = len(q_stems & stems(s)) / len(q_stems)
            if cov > 0:
                scored.append((cov, s, ctx))
    if not scored:
        return AnswerOut(found=False, answer="", source_ids=[])
    scored.sort(key=lambda t: -t[0])
    if scored[0][0] < 0.34 and len(q_stems) > 2:
        return AnswerOut(found=False, answer="", source_ids=[])
    top = scored[0][0]
    picked, seen_text = [], set()
    for cov, sent, ctx in scored:
        key = sent.strip().lower()
        if cov < max(0.34, 0.75 * top) or key in seen_text:
            continue
        seen_text.add(key)
        picked.append((cov, sent, ctx))
        if len(picked) == 3:
            break
    meetings = {c.meeting_id for _, _, c in picked}
    lines = []
    for _, s, c in picked:
        prefix = f"In “{c.meeting_title}”: " if len(meetings) > 1 else ""
        lines.append(prefix + s)
    labels = []
    for _, _, c in picked:
        if c.label not in labels:
            labels.append(c.label)
    return AnswerOut(found=True, answer=" ".join(lines), answer_type="synthesized" if len(meetings) > 1 else "direct",
                     source_ids=labels)


NEGATION_RE = re.compile(r"\b(not|no|never|won'?t|don'?t|doesn'?t|didn'?t|can'?t|cannot|isn'?t|aren'?t|without|rejected|declined)\b", re.I)


def judge(claim: str, evidence: Sequence[SegmentView]) -> ClaimJudgement:
    ev = " ".join(s.text for s in evidence)
    cov = coverage(claim, ev)
    if cov >= 0.5:
        sentences = [x for s in evidence for x in split_sentences(s.text)] or [ev]
        best_sentence = max(sentences, key=lambda x: coverage(claim, x))  # compare polarity where the claim is stated
        if cov >= 0.75 and bool(NEGATION_RE.search(claim)) != bool(NEGATION_RE.search(best_sentence)):
            return ClaimJudgement(verdict="conflicting", reason="Claim and cited lines differ in polarity.")
        return ClaimJudgement(verdict="supported", reason="Key terms of the claim appear in the cited lines.")
    return ClaimJudgement(verdict="unsupported", reason="Cited lines do not cover the claim's key terms.")
