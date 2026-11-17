"""LLM abstraction layer (PRD §8, §13: providers stay replaceable).

* ``ChatLLM``       – any OpenAI-compatible chat endpoint (Groq, OpenAI, Ollama, vLLM ...) through LangChain.
* ``HeuristicLLM``  – dependency-free, offline rule-based fallback so the whole app works without an API key.

Both implement the same four operations, so the rest of the system never depends on a vendor."""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.errors import ProviderError
from app.prompts import ANSWER_PROMPT, EXTRACTION_PROMPT, JUDGE_PROMPT, MERGE_SUMMARY_PROMPT
from app.schemas.extraction import AnswerOut, ClaimJudgement, SummaryOut, WindowExtraction
from app.services import heuristics
from app.services.textutil import fmt_time
from app.services.dtypes import ContextChunk, SegmentView, Usage

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {"groq": "llama-3.3-70b-versatile", "openai": "gpt-4o-mini"}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMService(Protocol):
    provider: str
    model: str
    external: bool
    usage: Usage

    def extract(self, title: str, segments: Sequence[SegmentView], window_note: str = "") -> WindowExtraction: ...

    def merge_summaries(self, title: str, partials: Sequence[SummaryOut]) -> SummaryOut: ...

    def answer(self, question: str, contexts: Sequence[ContextChunk], history: Sequence[tuple] = ()) -> AnswerOut: ...

    def judge_claim(self, kind: str, claim: str, evidence: Sequence[SegmentView]) -> ClaimJudgement: ...


def format_transcript(segments: Sequence[SegmentView]) -> str:
    lines = []
    for s in segments:
        who = s.speaker or "Speaker"
        lines.append(f"[{s.ref} | {fmt_time(s.start)} | {who}] {s.text}")
    return "\n".join(lines)


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating code fences and leading/trailing prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


_TRANSIENT = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "Timeout",
              "ReadTimeout", "ConnectError", "ConnectTimeout", "ServiceUnavailableError"}


def _is_transient(exc: Exception) -> bool:
    if type(exc).__name__ in _TRANSIENT:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


class ChatLLM:
    """LangChain-based client with retries, exponential backoff and schema-validated JSON output."""

    external = True

    def __init__(self, provider: str, model: str, api_key: Optional[str], base_url: Optional[str]):
        from langchain_openai import ChatOpenAI  # lazy: keeps offline mode import-light

        s = get_settings()
        self.provider, self.model = provider, model
        self.usage = Usage()
        self._max_retries = s.llm_max_retries
        self._json_mode = True
        self._chat = ChatOpenAI(model=model, api_key=api_key or "none", base_url=base_url,
                                temperature=s.llm_temperature, timeout=s.llm_timeout_s, max_retries=0)
        self._chat_json = self._chat.bind(response_format={"type": "json_object"})

    # ------------------------------------------------------------ plumbing
    def _call(self, prompt, variables: dict, schema: Type[T]) -> T:
        messages = prompt.format_messages(**variables)
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                model = self._chat_json if self._json_mode else self._chat
                resp = model.invoke(messages)
                self.usage.calls += 1
                meta = getattr(resp, "usage_metadata", None) or {}
                self.usage.input_tokens += int(meta.get("input_tokens", 0) or 0)
                self.usage.output_tokens += int(meta.get("output_tokens", 0) or 0)
                content = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)
                return schema.model_validate(extract_json(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                self.usage.retries += 1
                logger.warning("LLM output failed validation (attempt %d): %s", attempt + 1, type(exc).__name__)
                messages = messages + [
                    ("human", f"Your previous reply was invalid ({type(exc).__name__}). "
                              "Reply again with ONE valid JSON object that follows the schema exactly.")]
            except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary
                last_error = exc
                if self._json_mode and getattr(exc, "status_code", None) == 400:
                    logger.info("Provider rejected JSON mode; retrying without it.")
                    self._json_mode = False
                    self.usage.retries += 1
                    continue
                if not _is_transient(exc):
                    raise ProviderError(f"LLM request failed: {type(exc).__name__}") from exc
                self.usage.retries += 1
                delay = min(30.0, (2 ** attempt) + random.random())
                logger.warning("Transient LLM error (%s); retrying in %.1fs", type(exc).__name__, delay)
                time.sleep(delay)
        raise ProviderError(f"LLM did not return a usable response after retries: "
                            f"{type(last_error).__name__ if last_error else 'unknown'}")

    # ------------------------------------------------------------ operations
    def extract(self, title: str, segments: Sequence[SegmentView], window_note: str = "") -> WindowExtraction:
        return self._call(EXTRACTION_PROMPT, {"title": title, "window_note": window_note,
                                              "transcript": format_transcript(segments)}, WindowExtraction)

    def merge_summaries(self, title: str, partials: Sequence[SummaryOut]) -> SummaryOut:
        text = "\n\n".join(f"Part {i + 1}:\n{json.dumps(p.model_dump(), ensure_ascii=False)}"
                           for i, p in enumerate(partials))
        return self._call(MERGE_SUMMARY_PROMPT, {"title": title, "partials": text}, SummaryOut)

    def answer(self, question: str, contexts: Sequence[ContextChunk], history: Sequence[tuple] = ()) -> AnswerOut:
        ctx = "\n\n".join(
            f"[{c.label}] Meeting: \"{c.meeting_title}\" | {fmt_time(c.start)}-{fmt_time(c.end)}\n{c.text}"
            for c in contexts)
        hist = ""
        if history:
            hist = "CONVERSATION SO FAR\n" + "\n".join(f"{r.upper()}: {t}" for r, t in history[-6:]) + "\n\n"
        return self._call(ANSWER_PROMPT, {"question": question, "context": ctx, "history": hist}, AnswerOut)

    def judge_claim(self, kind: str, claim: str, evidence: Sequence[SegmentView]) -> ClaimJudgement:
        return self._call(JUDGE_PROMPT, {"kind": kind, "claim": claim, "evidence": format_transcript(evidence)},
                          ClaimJudgement)


class HeuristicLLM:
    """Offline rule-based stand-in. Lower quality than a real LLM, but fully local and deterministic."""

    external = False

    def __init__(self) -> None:
        self.provider, self.model = "heuristic", "rules-v1"
        self.usage = Usage()

    def extract(self, title: str, segments: Sequence[SegmentView], window_note: str = "") -> WindowExtraction:
        self.usage.calls += 1
        return heuristics.extract(segments)

    def merge_summaries(self, title: str, partials: Sequence[SummaryOut]) -> SummaryOut:
        self.usage.calls += 1
        return heuristics.merge_summaries(partials)

    def answer(self, question: str, contexts: Sequence[ContextChunk], history: Sequence[tuple] = ()) -> AnswerOut:
        self.usage.calls += 1
        return heuristics.answer(question, contexts)

    def judge_claim(self, kind: str, claim: str, evidence: Sequence[SegmentView]) -> ClaimJudgement:
        self.usage.calls += 1
        return heuristics.judge(claim, evidence)


def get_llm() -> LLMService:
    """Create a fresh LLM client (one per job, so usage counters are job-scoped)."""
    s = get_settings()
    provider = s.resolved_llm_provider
    if provider == "heuristic":
        return HeuristicLLM()
    if not s.allow_external_llm:
        logger.warning("ALLOW_EXTERNAL_LLM=false: using the offline heuristic provider.")
        return HeuristicLLM()
    if provider == "groq":
        if not s.groq_api_key:
            raise ProviderError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        return ChatLLM("groq", s.llm_model or DEFAULT_MODELS["groq"], s.groq_api_key, s.llm_base_url or GROQ_BASE_URL)
    if provider == "openai":
        if not s.openai_api_key:
            raise ProviderError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        return ChatLLM("openai", s.llm_model or DEFAULT_MODELS["openai"], s.openai_api_key, s.llm_base_url)
    if provider == "openai_compatible":
        if not (s.llm_base_url and s.llm_model):
            raise ProviderError("openai_compatible requires LLM_BASE_URL and LLM_MODEL.")
        return ChatLLM("openai_compatible", s.llm_model, s.llm_api_key or "none", s.llm_base_url)
    raise ProviderError(f"Unknown LLM_PROVIDER '{provider}'.")


def describe_llm() -> Dict[str, str]:
    s = get_settings()
    provider = s.resolved_llm_provider
    if provider == "heuristic" or not s.allow_external_llm:
        return {"provider": "heuristic", "model": "rules-v1"}
    model = s.llm_model or DEFAULT_MODELS.get(provider, "unknown")
    return {"provider": provider, "model": model}
