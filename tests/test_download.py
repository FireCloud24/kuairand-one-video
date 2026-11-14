"""The download script: verification, skipping and safe extraction. No network."""
import hashlib
import io
import tarfile

import pytest

from scripts import download


class FakeResponse(io.BytesIO):
    status = 200
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_for(payload: bytes, calls: list):
    def opener(req):
        calls.append(req.full_url)
        return FakeResponse(payload)
    return opener


def spec_for(payload: bytes, name="file.bin"):
    return {"name": name, "url": "https://example.org/" + name, "md5": hashlib.md5(payload).hexdigest()}


def test_download_verifies_and_records(tmp_path):
    payload, calls = b"hello kuairand", []
    spec = spec_for(payload)
    path = download.download_verified(tmp_path, spec, opener_for(payload, calls))
    assert path.read_bytes() == payload
    assert download.load_manifest(tmp_path)["files"]["file.bin"]["md5"] == spec["md5"]
    assert calls == [spec["url"]]


def test_second_run_skips_the_fetch(tmp_path):
    payload, calls = b"abc", []
    spec = spec_for(payload)
    download.download_verified(tmp_path, spec, opener_for(payload, calls))
    download.download_verified(tmp_path, spec, opener_for(payload, calls))
    assert len(calls) == 1


def test_wrong_md5_is_an_error(tmp_path):
    spec = spec_for(b"what the authors published")
    with pytest.raises(ValueError, match="MD5"):
        download.download_verified(tmp_path, spec, opener_for(b"something else", []))
    assert "file.bin" not in download.load_manifest(tmp_path)["files"]


def make_tar(path, members: dict):
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def test_extract_takes_only_wanted_files_and_flattens(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_tar(archive, {
        "KuaiRand-1K/data/keep.csv": b"a,b\n1,2\n",
        "KuaiRand-1K/data/big_unwanted.csv": b"x" * 100,
        "../escape/keep2.csv": b"c\n3\n",
    })
    out = tmp_path / "out"
    names = download.extract(archive, out, ("keep.csv", "keep2.csv"))
    assert sorted(names) == ["keep.csv", "keep2.csv"]
    assert sorted(p.name for p in out.iterdir()) == ["keep.csv", "keep2.csv"]
    assert not (tmp_path / "escape").exists()
    assert (out / "keep.csv").read_bytes() == b"a,b\n1,2\n"


def test_extract_reports_missing_files(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_tar(archive, {"data/keep.csv": b"a\n"})
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        download.extract(archive, tmp_path / "out", ("keep.csv", "absent.csv"))
