"""Web app: a server restart (e.g. Ctrl+C mid-video) must send the user back to the login page and
must never resume the old video; a page reload must not process the same upload twice."""
import io

import pytest

import app as webapp


class _Cursor:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a):
        pass

    def fetchone(self):
        return {"username": "admin", "password": "pw"}

    def fetchall(self):
        return []


class _Conn:
    def cursor(self):
        return _Cursor()

    def close(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    processed = []
    monkeypatch.setattr(webapp, "get_connection", lambda: _Conn())
    monkeypatch.setattr(webapp, "main_fun", lambda video, out: processed.append(video.split("\\")[-1].split("/")[-1]))
    monkeypatch.setitem(webapp.app.config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setitem(webapp.app.config, "DETECTIONS_FOLDER", str(tmp_path))
    monkeypatch.setattr(webapp, "BOOT_ID", "boot-1", raising=False)
    getattr(webapp, "PENDING_JOBS", {}).clear()
    c = webapp.app.test_client()
    c.processed = processed
    return c


def _login(c):
    return c.post("/validate_login", data={"username": "admin", "password": "pw"})


def _upload(c, name):
    return c.post("/live_monitoring", content_type="multipart/form-data",
                  data={"numCameras": "1", "processType": "helmet", "cameraFile1": (io.BytesIO(b"video"), name)})


def _restart(monkeypatch):
    """What a server restart does: a new boot id and no jobs in memory."""
    monkeypatch.setattr(webapp, "BOOT_ID", "boot-2", raising=False)
    getattr(webapp, "PENDING_JOBS", {}).clear()


def test_pages_need_login(client):
    for page in ("/home", "/live_monitoring", "/start_processing", "/results"):
        r = client.get(page)
        assert r.status_code == 302 and r.headers["Location"].endswith("/")


def test_upload_is_processed_once(client):
    _login(client)
    assert _upload(client, "a.mp4").headers["Location"].endswith("/start_processing")
    assert client.get("/start_processing").headers["Location"].endswith("/results")
    assert client.processed == ["a.mp4"]
    # reloading the processing page does not run the video again
    assert client.get("/start_processing").headers["Location"].endswith("/live_monitoring")
    assert client.processed == ["a.mp4"]


def test_restart_goes_to_login_and_never_resumes_old_video(client, monkeypatch):
    _login(client)
    _upload(client, "old.mp4")        # processing starts, then the server is stopped with Ctrl+C
    _restart(monkeypatch)
    # the browser re-sends the unfinished request with its old cookie
    r = client.get("/start_processing")
    assert r.status_code == 302 and r.headers["Location"].endswith("/")      # login page
    assert client.get("/home").headers["Location"].endswith("/")
    assert client.processed == []
    # after logging in again, a new upload processes the NEW video only
    _login(client)
    _upload(client, "new.mp4")
    client.get("/start_processing")
    assert client.processed == ["new.mp4"]
