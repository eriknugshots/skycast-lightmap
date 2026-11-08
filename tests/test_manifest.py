import json
import os

import pytest

from pipeline import manifest as M


def test_failed_write_keeps_old_manifest_and_leaves_no_tmp(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text('{"old":true}')

    def boom(obj, f, **kw):
        f.write('{"trunc')
        raise OSError("disk full")
    monkeypatch.setattr(M.json, "dump", boom)
    with pytest.raises(OSError):
        M.write_manifest(str(tmp_path), {"new": True})
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"old": True}
    assert sorted(os.listdir(tmp_path)) == ["manifest.json"]


def test_write_replaces_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text('{"old":true}')
    M.write_manifest(str(tmp_path), {"new": True})
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"new": True}
    assert sorted(os.listdir(tmp_path)) == ["manifest.json"]
