"""Download handling in data/mp_fetch.py, with the network mocked out."""
import pytest
import requests

from conftest import load_script

mp_fetch = load_script("data/mp_fetch.py")


class FakeResponse:
    def __init__(self, content=b"", status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"status {self.status_code}")


@pytest.fixture
def outdir(tmp_path, monkeypatch):
    monkeypatch.setattr(mp_fetch, "OUT", str(tmp_path))
    monkeypatch.setattr(mp_fetch.time, "sleep", lambda s: None)
    return tmp_path


def item(name="stock_daily_2020-01-02.parquet"):
    return {"filename": name, "download_url": "https://example.invalid/file"}


def test_writes_complete_file_and_no_temp_file(outdir, monkeypatch):
    monkeypatch.setattr(mp_fetch.requests, "get", lambda *a, **k: FakeResponse(b"x" * 2000))
    assert mp_fetch.grab(item()) == 2000
    assert (outdir / item()["filename"]).read_bytes() == b"x" * 2000
    assert not list(outdir.glob("*.part"))


def test_existing_file_is_skipped(outdir, monkeypatch):
    (outdir / item()["filename"]).write_bytes(b"y" * 5000)
    monkeypatch.setattr(mp_fetch.requests, "get", lambda *a, **k: pytest.fail("should not download"))
    assert mp_fetch.grab(item()) == 0


def test_failures_leave_nothing_behind(outdir, monkeypatch, capsys):
    def boom(*a, **k):
        raise requests.ConnectionError("network down")
    monkeypatch.setattr(mp_fetch.requests, "get", boom)
    assert mp_fetch.grab(item()) == 0
    assert not list(outdir.iterdir())
    assert "ConnectionError" in capsys.readouterr().out


def test_http_error_page_is_not_saved(outdir, monkeypatch):
    monkeypatch.setattr(mp_fetch.requests, "get", lambda *a, **k: FakeResponse(b"<error>" * 500, status=403))
    assert mp_fetch.grab(item()) == 0
    assert not list(outdir.iterdir())


def test_interrupted_write_does_not_create_the_final_file(outdir, monkeypatch):
    dst = str(outdir / "f.parquet")

    def crash(src, target):
        raise OSError("disk full")
    monkeypatch.setattr(mp_fetch.os, "replace", crash)
    with pytest.raises(OSError):
        mp_fetch.write_atomic(dst, b"z" * 2000)
    assert not (outdir / "f.parquet").exists()
