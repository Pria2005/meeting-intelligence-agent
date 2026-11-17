"""Offline evaluation harness for extraction and RAG (PRD §15).

    python -m app.evaluation.run --dataset ../docs/eval_sample.json

Dataset format (JSON):
{
  "meetings": [{
      "title": "...", "segments": [{"speaker": "Priya", "text": "..."}],
      "gold": {"decisions": ["..."], "action_items": ["..."], "risks": ["..."]},
      "questions": [{"question": "...", "relevant_refs": ["seg-0008"], "expect_found": true}]
  }]
}
The harness runs the *configured* LLM / embedding backend and reports the model configuration with the results."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from app.evaluation.metrics import mean, precision_recall_f1, recall_at_k, reciprocal_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    import os

    workdir = tempfile.mkdtemp(prefix="mi_eval_")
    os.environ["DATA_DIR"] = workdir  # isolate from real data
    os.environ.pop("DATABASE_URL", None)

    from app.core.config import get_settings
    from app.db.database import init_db, session_scope
    from app.db import models as m
    from app.services import repository as repo
    from app.services.chunking import chunk_transcript
    from app.services.dtypes import SegmentView
    from app.services.extraction import run_extraction
    from app.services.llm import describe_llm, get_llm
    from app.services.vectorstore import get_vector_store
    from app.services.embeddings import get_embedder
    from app.services.pipeline import _replace_transcript
    from app.services.transcription import RawSegment
    from app.services.indexing import index_meeting
    from app.services.rag import _rank

    init_db()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    s = get_settings()
    llm = get_llm()
    ext_scores = {"decisions": [], "action_items": [], "risks": []}
    recalls, mrrs = [], []
    abstain_ok = []

    with session_scope() as db:
        user = repo.get_or_create_default_user(db)
        for item in data["meetings"]:
            t, raws = 0.0, []
            for seg in item["segments"]:
                dur = max(3.0, len(seg["text"].split()) / 2.6)
                raws.append(RawSegment(t, t + dur, seg["text"], speaker=seg.get("speaker")))
                t += dur + 1
            mt = m.Meeting(user_id=user.id, title=item["title"], source="demo", status="completed")
            db.add(mt)
            db.flush()
            _replace_transcript(db, mt, raws)
            views = repo.segment_views(db, mt.id)
            ex = run_extraction(llm, mt.id, mt.title, views)
            gold = item.get("gold", {})
            pred = {"decisions": [d.text for d in ex.decisions], "action_items": [a.task for a in ex.action_items],
                    "risks": [r.text for r in ex.risks + ex.open_questions]}
            for key in ext_scores:
                if key in gold:
                    ext_scores[key].append(precision_recall_f1(pred[key], gold[key]))
            index_meeting(db, mt)
            db.flush()
            for q in item.get("questions", []):
                hits = get_vector_store().query(q["question"], [mt.id], top_k=args.k * 3)
                ranked = _rank(q["question"], hits, args.k, s.rag_min_score or 0.0)
                if q.get("expect_found", True):
                    relevant = {h.chunk_id for h in hits
                                if set(q["relevant_refs"]) & set(str(h.metadata.get("segment_refs", "")).split(","))}
                    ids = [h.chunk_id for h in ranked]
                    recalls.append(recall_at_k(ids, relevant, args.k))
                    mrrs.append(reciprocal_rank(ids, relevant))
                else:
                    abstain_ok.append(1.0 if not ranked else 0.0)

    report = {
        "configuration": {"llm": describe_llm(), "embedding": get_embedder().name,
                          "rag_min_score": s.rag_min_score, "chunk_chars": s.chunk_chars},
        "extraction": {k: {m_: round(mean(x[m_] for x in v), 4) for m_ in ("precision", "recall", "f1")}
                       for k, v in ext_scores.items() if v},
        "retrieval": {f"recall@{args.k}": round(mean(recalls), 4), "mrr": round(mean(mrrs), 4),
                      "abstention_accuracy": round(mean(abstain_ok), 4) if abstain_ok else None},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
