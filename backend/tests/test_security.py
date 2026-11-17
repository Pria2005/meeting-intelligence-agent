from tests.conftest import upload_and_wait


def register(c, name):
    r = c.post("/auth/register", json={"username": name, "password": "supersecret1"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_auth_required(auth_client):
    assert auth_client.get("/meetings").status_code == 401
    assert auth_client.get("/meetings", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert auth_client.get("/health").status_code == 200


def test_login_flow(auth_client):
    register(auth_client, "alice")
    assert auth_client.post("/auth/login", json={"username": "alice", "password": "wrongwrong"}).status_code == 401
    ok = auth_client.post("/auth/login", json={"username": "alice", "password": "supersecret1"})
    assert ok.status_code == 200 and ok.json()["user"]["username"] == "alice"
    assert auth_client.post("/auth/register", json={"username": "alice", "password": "supersecret1"}).status_code == 409


def test_users_cannot_access_each_others_meetings(auth_client, wav_bytes):
    from app.services import jobs

    a, b = register(auth_client, "alice"), register(auth_client, "bob")
    r = auth_client.post("/meetings/upload", files={"file": ("a.wav", wav_bytes, "audio/wav")}, headers=a)
    mid = r.json()["meeting_id"]
    jobs.wait_for(mid, 60)
    assert auth_client.get(f"/meetings/{mid}", headers=a).status_code == 200
    for path in ("", "/transcript", "/summary", "/export", "/audio", "/events"):
        assert auth_client.get(f"/meetings/{mid}{path}", headers=b).status_code == 404, path
    assert auth_client.delete(f"/meetings/{mid}", headers=b).status_code == 404
    assert auth_client.post(f"/meetings/{mid}/transcribe", headers=b).status_code == 404
    assert auth_client.get("/meetings", headers=b).json() == []
    # cross-meeting search never leaks another user's content
    ans = auth_client.post("/chat", json={"question": "What did we decide about the launch date?"}, headers=b).json()
    assert ans["found"] is False and ans["citations"] == []
    assert auth_client.post("/chat", json={"question": "launch date?", "meeting_ids": [mid]}, headers=b).status_code == 404
    mine = auth_client.post("/chat", json={"question": "What did we decide about the launch date?"}, headers=a).json()
    assert mine["found"] is True
    # action items are scoped too
    aid = auth_client.get("/action-items", headers=a).json()[0]["id"]
    assert auth_client.patch(f"/action-items/{aid}", json={"status": "done"}, headers=b).status_code == 404
    assert auth_client.get("/action-items", headers=b).json() == []
