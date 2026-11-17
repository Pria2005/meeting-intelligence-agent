import pytest

from tests.conftest import upload_and_wait


@pytest.fixture()
def mid(client, wav_bytes):
    return upload_and_wait(client, wav_bytes)


def test_chat_grounded_and_abstains(client, mid):
    r = client.post("/chat", json={"question": "What did we decide about the launch date?"}).json()
    assert r["found"] and r["citations"]
    c = r["citations"][0]
    assert c["meeting_id"] == mid and c["segment_refs"] and c["end_time"] > c["start_time"]
    assert "thirtieth" in r["answer"].lower()
    miss = client.post("/chat", json={"question": "What is the capital of Mongolia?"}).json()
    assert miss["found"] is False and miss["answer_type"] == "not_found" and miss["citations"] == []


def test_cross_meeting_memory(client):
    ids = [m["id"] for m in client.post("/meetings/demo").json()]
    from app.services import jobs

    for i in ids:
        assert jobs.wait_for(i, 60)
    r = client.post("/chat", json={"question": "Which launch date is the thirtieth?"}).json()
    assert len({c["meeting_id"] for c in r["citations"]}) == 2
    assert r["answer_type"] == "synthesized"
    only = client.post("/chat", json={"question": "launch date decision", "meeting_ids": [ids[0]]}).json()
    assert {c["meeting_id"] for c in only["citations"]} <= {ids[0]}
    # demo creation is idempotent
    assert [m["id"] for m in client.post("/meetings/demo").json()] == ids


def test_action_item_patch_and_review_flow(client, mid):
    item = client.get(f"/meetings/{mid}/summary").json()["action_items"][0]
    r = client.patch(f"/action-items/{item['id']}", json={"owner": "Dana", "status": "done", "due_date": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["owner"] == "Dana" and body["status"] == "done" and body["due_date"] is None and body["user_edited"]
    assert client.patch(f"/action-items/{item['id']}", json={"status": "bogus"}).status_code == 422
    assert client.patch("/action-items/99999", json={"status": "done"}).status_code == 404
    assert any(a["id"] == item["id"] for a in client.get("/action-items", params={"status": "done"}).json())


def test_speaker_rename_and_segment_edit_reindex(client, mid):
    from app.services import jobs
    tr = client.get(f"/meetings/{mid}/transcript").json()
    seg = tr["segments"][7]
    r = client.patch(f"/meetings/{mid}/segments/{seg['id']}", json={"text": "We decided to launch on Zanzibar day."})
    assert r.status_code == 200 and r.json()["edited"]
    ans = client.post("/chat", json={"question": "Zanzibar"}).json()
    assert ans["found"]  # edit is searchable immediately
    assert client.patch(f"/meetings/{mid}/segments/999999", json={"text": "x"}).status_code == 404


def test_summary_versioning(client, mid):
    r = client.patch(f"/meetings/{mid}/summary", json={"overview": "My corrected overview."})
    assert r.json()["version"] == 2 and r.json()["edited_by_user"]
    assert client.get(f"/meetings/{mid}/summary").json()["summary"]["overview"] == "My corrected overview."


def test_reprocess_is_idempotent(client, mid):
    from app.services import jobs
    from app.services.vectorstore import get_vector_store

    before = get_vector_store().count(mid)
    n_actions = len(client.get(f"/meetings/{mid}/summary").json()["action_items"])
    assert client.post(f"/meetings/{mid}/transcribe").status_code == 202
    jobs.wait_for(mid, 60)
    assert get_vector_store().count(mid) == before  # no duplicate vectors
    assert len(client.get(f"/meetings/{mid}/summary").json()["action_items"]) == n_actions
    assert client.get(f"/meetings/{mid}/summary").json()["summary"]["version"] == 2
    assert client.get(f"/meetings/{mid}").json()["counts"]["segments"] > 0


def test_export_markdown_and_pdf(client, mid):
    md = client.get(f"/meetings/{mid}/export", params={"format": "md"})
    assert md.status_code == 200 and "## Decisions" in md.text and "## Transcript" in md.text
    assert "seg-" in md.text
    pdf = client.get(f"/meetings/{mid}/export", params={"format": "pdf", "include_transcript": "false"})
    assert pdf.content[:4] == b"%PDF"
    assert client.get(f"/meetings/{mid}/export", params={"format": "docx"}).status_code == 422


def test_audio_streaming_and_list_filters(client, mid, wav_bytes):
    assert client.get(f"/meetings/{mid}/audio").content[:4] == b"RIFF"
    assert len(client.get("/meetings", params={"status": "completed"}).json()) == 1
    assert client.get("/meetings", params={"q": "weekly"}).json()[0]["id"] == mid
    assert client.get("/meetings", params={"status": "nope"}).status_code == 422


def test_feedback_and_stats(client, mid):
    assert client.post("/feedback", json={"meeting_id": mid, "item_type": "summary", "rating": 1}).status_code == 201
    assert client.post("/feedback", json={"meeting_id": "mtg_x", "item_type": "summary", "rating": 1}).status_code == 404
    s = client.get("/stats").json()
    assert s["total_meetings"] == 1 and s["total_decisions"] >= 3 and s["open_action_items"] >= 1


def test_decisions_always_cite_valid_evidence(client, mid):
    notes = client.get(f"/meetings/{mid}/summary").json()
    refs = {s["ref"] for s in client.get(f"/meetings/{mid}/transcript").json()["segments"]}
    for group in ("decisions", "action_items", "risks", "open_questions"):
        for it in notes[group]:
            assert set(it["evidence_segment_ids"]) <= refs and it["evidence_segment_ids"]
