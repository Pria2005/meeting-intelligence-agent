from app.evaluation import metrics
from app.schemas.extraction import ActionItemOut, WindowExtraction, normalize_ref
from app.services.chunking import chunk_transcript
from app.services.diarization import Turn, assign_speakers
from app.services.dtypes import SegmentView
from app.services.extraction import make_windows, validate_extraction
from app.services.embeddings import HashingEmbedder
from app.services.transcription import RawSegment
from app.services.llm import extract_json


def seg(i, text, spk="A", dur=5.0):
    return SegmentView(ref=f"seg-{i:04d}", start=i * dur, end=(i + 1) * dur, speaker=spk, text=text)


def test_chunking_keeps_timestamps_overlap_and_limits():
    segs = [seg(i, "word " * 30) for i in range(1, 21)]
    chunks = chunk_transcript(segs, max_chars=500, overlap_chars=160)
    assert len(chunks) > 3
    assert all(len(c.text) <= 500 + 200 for c in chunks)
    assert chunks[0].start == segs[0].start and chunks[-1].end == segs[-1].end
    assert set(chunks[0].segment_refs) & set(chunks[1].segment_refs)  # overlap
    covered = {r for c in chunks for r in c.segment_refs}
    assert covered == {s.ref for s in segs}
    assert chunks[0].chunk_id("m1") == "m1:c0000"


def test_chunking_splits_giant_segment():
    chunks = chunk_transcript([seg(1, "This is a sentence. " * 200)], max_chars=400, overlap_chars=0)
    assert len(chunks) > 5 and all(len(c.text) < 600 for c in chunks)


def test_ref_normalisation_and_schema_cleaning():
    assert normalize_ref("seg-4") == "seg-0004" and normalize_ref("S12") == "seg-0012" and normalize_ref("x") is None
    a = ActionItemOut(task="Send report", owner="unknown", due_date="N/A", evidence_segment_ids=["seg-1", "1", 2])
    assert a.owner is None and a.due_date is None and a.evidence_segment_ids == ["seg-0001", "seg-0002"]


def test_validate_drops_bad_refs():
    ex = WindowExtraction.model_validate({"summary": {"overview": "ok ok"}, "decisions": [
        {"text": "Do the thing", "evidence_segment_ids": ["seg-1", "seg-99"]}]})
    from app.schemas.extraction import MeetingExtraction
    me = MeetingExtraction(summary=ex.summary, decisions=ex.decisions)
    me, issues, ratio = validate_extraction(me, {"seg-0001"})
    assert me.decisions[0].evidence_segment_ids == ["seg-0001"] and issues and ratio == 1.0


def test_windows_overlap():
    segs = [seg(i, "x" * 200) for i in range(1, 30)]
    w = make_windows(segs, 1500, overlap_segments=2)
    assert len(w) > 2 and w[1][0].ref == w[0][-2].ref


def test_diarization_assignment_and_unknown():
    raws = [RawSegment(0, 4, "a"), RawSegment(4, 8, "b"), RawSegment(8, 12, "c"), RawSegment(20, 24, "d")]
    turns = [Turn(0, 4, "S_X"), Turn(4, 8, "S_Y"), Turn(8, 12, "S_X")]
    out = assign_speakers(raws, turns)
    assert [r.speaker for r in out] == ["Speaker 1", "Speaker 2", "Speaker 1", "Unknown"]


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} Hope that helps') == {"a": {"b": 2}}


def test_hash_embedder_similarity():
    e = HashingEmbedder()
    v = e.embed(["deploy the payment service", "payment service deployment", "banana smoothie recipe"])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(v[0], v[1]) > dot(v[0], v[2])


def test_metrics():
    assert metrics.word_error_rate("the cat sat", "the cat sat") == 0
    assert metrics.word_error_rate("the cat sat", "the dog sat down") == 2 / 3
    assert metrics.recall_at_k(["a", "b", "c"], {"c", "z"}, 3) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    prf = metrics.precision_recall_f1(["send the report friday"], ["send report by friday", "other item"])
    assert prf["precision"] == 1.0 and prf["recall"] == 0.5
    ref = [(0, 5, "A"), (5, 10, "B")]
    assert metrics.diarization_error_rate(ref, [(0, 5, "x"), (5, 10, "y")]) < 0.05
    assert metrics.diarization_error_rate(ref, []) == 1.0


def test_heuristic_does_not_invent_owner_or_date():
    from app.services import heuristics

    out = heuristics.extract([seg(1, "We need to update the budget spreadsheet.", spk="Speaker 1")])
    assert out.action_items and out.action_items[0].owner is None and out.action_items[0].due_date is None
    named = heuristics.extract([seg(1, "I will update the budget spreadsheet by Friday.", spk="Dana")])
    assert named.action_items[0].owner == "Dana" and named.action_items[0].due_date.lower() == "friday"
