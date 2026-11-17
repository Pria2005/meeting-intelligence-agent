from tests.conftest import upload_and_wait


def test_root_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["name"]
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["database"] == "ok" and h["vector_store"] == "ok"


def test_unknown_meeting_returns_consistent_error(client):
    r = client.get("/meetings/mtg_doesnotexist")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "not_found"


def test_upload_pipeline_end_to_end(client, wav_bytes):
    mid = upload_and_wait(client, wav_bytes)
    detail = client.get(f"/meetings/{mid}").json()
    assert detail["status"] == "completed", detail
    assert detail["counts"]["segments"] > 5 and detail["counts"]["chunks"] >= 1
    assert detail["has_audio"] is True
    assert "storage_path" not in detail  # filesystem paths are never exposed

    tr = client.get(f"/meetings/{mid}/transcript").json()
    assert tr["segments"][0]["start_time"] < tr["segments"][0]["end_time"]
    assert tr["segments"][0]["ref"] == "seg-0001"

    notes = client.get(f"/meetings/{mid}/summary").json()
    assert notes["summary"]["overview"]
    decisions = " ".join(d["description"].lower() for d in notes["decisions"])
    assert "thirtieth" in decisions and "stripe" in decisions
    assert all(d["evidence"] for d in notes["decisions"])

    tasks = {a["task"]: a for a in notes["action_items"]}
    marcus = next(a for t, a in tasks.items() if "Stripe integration" in t)
    # Owner cannot be inferred without speaker labels -> must stay blank rather than be guessed.
    assert marcus["owner"] is None
    assert marcus["due_date"].lower() == "next wednesday"
    elena = next(a for t, a in tasks.items() if "screenshots" in t)
    assert elena["owner"] == "Elena" and elena["due_date"] is None
    assert any(r["kind"] in ("risk", "blocker") for r in notes["risks"])
    assert notes["open_questions"]


def test_events_recorded(client, wav_bytes):
    mid = upload_and_wait(client, wav_bytes)
    stages = [e["stage"] for e in client.get(f"/meetings/{mid}/events").json()]
    for expected in ("ingest", "transcribe", "chunk", "extract", "validate", "verify", "persist", "index"):
        assert expected in stages


def test_delete_removes_everything(client, wav_bytes):
    mid = upload_and_wait(client, wav_bytes)
    assert client.delete(f"/meetings/{mid}").status_code == 200
    assert client.get(f"/meetings/{mid}").status_code == 404
    assert client.get("/meetings").json() == []
    r = client.post("/chat", json={"question": "What was decided about the launch date?"})
    assert r.json()["found"] is False
