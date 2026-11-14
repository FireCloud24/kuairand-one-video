"""Download KuaiRand-1K into data/raw/ and check it against the published MD5.

KuaiRand (Gao et al., CIKM 2022, arXiv:2208.08696) is hosted on Zenodo, record
10439422. The data is never committed; this script fetches it. Re-running skips
anything already downloaded and verified.

    python scripts/download.py                  # archive + topic names, extract what the analysis needs
    python scripts/download.py --all-files      # also extract the 3.1 GB video statistics file
    python scripts/download.py --no-categories  # skip the supplementary topic-name file

Versions (from https://kuairand.com, read 2026-11-14):
  KuaiRand-27K   27,285 users, the complete dataset (9.9 GB compressed)
  KuaiRand-1K    1,000 users drawn uniformly from the 27K, with all their logs (1.1 GB)
  KuaiRand-Pure  27,285 users, but only logs on the 7,583 videos in the random pool
I use KuaiRand-1K. Pure cannot answer my question because it drops the ordinary
feed outside the random pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ARCHIVE = {
    "name": "KuaiRand-1K.tar.gz",
    "url": "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
    "md5": "6b0b9c8222d67fcd4c676218edca3f1f",
}
# Supplementary file from the same authors (Zenodo record 18159199): a category
# hierarchy with names for each video.
CATEGORIES = {
    "name": "kuairand_video_categories.csv",
    "url": "https://zenodo.org/records/18159199/files/kuairand_video_categories.csv",
    "md5": "8a4c772d3d8f8cf65fa1e1eb896f4177",
}
# Files the analysis reads. The statistics file is 3.1 GB and is not needed.
NEEDED = (
    "log_random_4_22_to_5_08_1k.csv",
    "log_standard_4_08_to_4_21_1k.csv",
    "log_standard_4_22_to_5_08_1k.csv",
    "user_features_1k.csv",
    "video_features_basic_1k.csv",
)
OPTIONAL = ("video_features_statistic_1k.csv",)
MANIFEST = "manifest.json"
USER_AGENT = "kuairand-one-video-downloader/0.1 (research use)"
CHUNK = 1 << 20


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(raw_dir: Path) -> dict:
    path = raw_dir / MANIFEST
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}}


def save_manifest(raw_dir: Path, manifest: dict) -> None:
    (raw_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def fetch(url: str, dest: Path, opener=urllib.request.urlopen) -> None:
    """Stream url to dest, resuming a partial .part file when the server allows it."""
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    with opener(req) as resp:
        resumed = start and getattr(resp, "status", 200) == 206
        done = start if resumed else 0
        total = resp.headers.get("Content-Length")
        total = int(total) + done if total else None
        last = time.time()
        with open(part, "ab" if resumed else "wb") as f:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 10:
                    last = time.time()
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    print(f"    {done / 1e6:.0f} MB{pct}", flush=True)
    part.replace(dest)


def download_verified(raw_dir: Path, spec: dict, opener=urllib.request.urlopen) -> Path:
    """Fetch spec['url'] unless a verified copy exists. Raise if the MD5 is wrong."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir)
    dest = raw_dir / spec["name"]
    entry = manifest["files"].get(spec["name"])
    if dest.exists() and entry and entry["md5"] == spec["md5"] and entry["bytes"] == dest.stat().st_size:
        print(f"skip  {spec['name']} (already downloaded and verified)")
        return dest
    if not dest.exists():
        print(f"fetch {spec['name']}")
        fetch(spec["url"], dest, opener)
    got = md5_of(dest)
    if got != spec["md5"]:
        raise ValueError(f"{spec['name']}: MD5 {got} does not match the published {spec['md5']}")
    manifest["files"][spec["name"]] = {"url": spec["url"], "md5": got, "bytes": dest.stat().st_size}
    save_manifest(raw_dir, manifest)
    return dest


def extract(archive: Path, out_dir: Path, wanted: tuple[str, ...]) -> list[str]:
    """Extract only the wanted csv files, flat into out_dir. Member paths are
    reduced to their base name, so nothing can be written outside out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in wanted:
                continue
            target = out_dir / name
            if not target.exists() or target.stat().st_size != member.size:
                print(f"extract {name} ({member.size / 1e6:.0f} MB)", flush=True)
                src = tar.extractfile(member)
                with open(target, "wb") as f:
                    while chunk := src.read(CHUNK):
                        f.write(chunk)
            done.append(name)
    missing = sorted(set(wanted) - set(done))
    if missing:
        raise FileNotFoundError(f"not found in {archive.name}: {missing}")
    return done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--all-files", action="store_true", help="also extract the video statistics file")
    p.add_argument("--no-categories", action="store_true", help="skip the supplementary topic-name file")
    args = p.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    archive = download_verified(raw_dir, ARCHIVE)
    wanted = NEEDED + (OPTIONAL if args.all_files else ())
    names = extract(archive, raw_dir, wanted)
    print(f"extracted {len(names)} files to {raw_dir}")
    if not args.no_categories:
        download_verified(raw_dir, CATEGORIES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
