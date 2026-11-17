"""Versioned prompt templates (PRD §13 maintainability). Bump PROMPT_VERSION on any change."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

PROMPT_VERSION = "v1"

EXTRACTION_SYSTEM = """You are a meticulous meeting analyst. You convert a transcript into structured notes.

The transcript is a list of lines formatted as:
[seg-0004 | 00:01:23 | Speaker] text

STRICT RULES
1. Use ONLY information stated in the transcript. Never invent facts, owners, or deadlines.
2. Every decision, action item, risk and open question MUST cite the seg-XXXX ids that support it in "evidence_segment_ids".
3. Decisions are EXPLICIT choices the group made. Ideas, suggestions and unresolved debates are NOT decisions.
4. Action items: "task" is what must be done. "owner" is a person only if the transcript explicitly assigns or volunteers them
   (e.g. "Maria will send the report", or a named speaker saying "I'll do it"). "due_date" only if a deadline is stated
   (copy it as spoken, e.g. "Friday", "end of next week", or ISO date if one is given). Otherwise use null. Do NOT guess.
5. Risks, blockers and open questions are kept SEPARATE from decisions. Use kind "risk", "blocker", or "open_question".
6. Be concise. No duplicated items. Write in the language of the transcript.
7. Respond with ONE JSON object and nothing else (no markdown fences).

JSON SCHEMA
{{
  "summary": {{
    "overview": "2-4 sentence overview",
    "purpose": "why the meeting was held (empty string if not stated)",
    "main_topics": ["topic", "..."],
    "key_points": ["important point", "..."],
    "outcome": "what the meeting concluded"
  }},
  "decisions": [{{"text": "...", "evidence_segment_ids": ["seg-0004"]}}],
  "action_items": [{{"task": "...", "owner": null, "due_date": null, "evidence_segment_ids": ["seg-0007"]}}],
  "risks": [{{"text": "...", "kind": "risk|blocker", "severity": "low|medium|high|null", "evidence_segment_ids": ["seg-0009"]}}],
  "open_questions": [{{"text": "...", "kind": "open_question", "evidence_segment_ids": ["seg-0011"]}}]
}}"""

EXTRACTION_USER = """Meeting title: {title}
{window_note}
TRANSCRIPT
{transcript}

Return the JSON object now."""

MERGE_SUMMARY_SYSTEM = """You combine partial summaries of consecutive parts of ONE meeting into a single coherent summary.
Use only the information given. Respond with ONE JSON object and nothing else:
{{"overview": "...", "purpose": "...", "main_topics": ["..."], "key_points": ["..."], "outcome": "..."}}
Keep overview to 2-5 sentences, at most 8 main_topics and 10 key_points. Remove duplicates."""

MERGE_SUMMARY_USER = """Meeting title: {title}

PARTIAL SUMMARIES (in order)
{partials}

Return the JSON object now."""

ANSWER_SYSTEM = """You answer questions about meetings using ONLY the evidence excerpts provided.

RULES
1. If the excerpts do not contain the answer, respond with found=false and answer="" . Never use outside knowledge.
2. Cite excerpts by their labels (C1, C2, ...) in "source_ids".
3. answer_type is "direct" when the answer is stated in the excerpts, or "synthesized" when you combine or infer a pattern
   across several excerpts/meetings. When synthesizing, say so explicitly in the answer.
4. When different meetings are involved, mention which meeting (by title) each fact comes from.
5. Be concise and specific. Include names, numbers and dates exactly as stated.
6. Respond with ONE JSON object and nothing else:
{{"found": true, "answer": "...", "answer_type": "direct|synthesized", "source_ids": ["C1"]}}"""

ANSWER_USER = """{history}QUESTION: {question}

EVIDENCE EXCERPTS
{context}

Return the JSON object now."""

JUDGE_SYSTEM = """You verify one claim extracted from a meeting against the cited transcript lines.
Verdicts: "supported" (the lines clearly state it), "unsupported" (the lines do not state it), "conflicting" (the lines contradict it).
Respond with ONE JSON object only: {{"verdict": "supported|unsupported|conflicting", "reason": "one short sentence"}}"""

JUDGE_USER = """CLAIM ({kind}): {claim}

CITED TRANSCRIPT LINES
{evidence}

Return the JSON object now."""


def _tmpl(system: str, user: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", system), ("human", user)])


EXTRACTION_PROMPT = _tmpl(EXTRACTION_SYSTEM, EXTRACTION_USER)
MERGE_SUMMARY_PROMPT = _tmpl(MERGE_SUMMARY_SYSTEM, MERGE_SUMMARY_USER)
ANSWER_PROMPT = _tmpl(ANSWER_SYSTEM, ANSWER_USER)
JUDGE_PROMPT = _tmpl(JUDGE_SYSTEM, JUDGE_USER)
