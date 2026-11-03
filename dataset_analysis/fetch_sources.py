"""Retrieve the teacher-linked DRUM record and selected original ZIP members.

HTTP byte ranges avoid downloading model checkpoints or duplicate image copies.
Run with --list first, then --extract after reviewing the archive manifest.
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources"
API = "https://conservancy.umn.edu/server/api/core"
ITEM = "6dd6a960-c44a-4510-a679-efb8c82ebfb7"


class RemoteZip(io.RawIOBase):
    def __init__(self, session, url, size):
        self.session, self.url, self.size, self.pos = session, url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        if size <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + size) - 1
        with self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=90, stream=True) as response:
            if response.status_code != 206:
                raise RuntimeError(f"Expected HTTP 206 for range read, received {response.status_code}")
            data = response.content
        if len(data) != end - self.pos + 1:
            raise RuntimeError("Unexpected byte range length")
        self.pos += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--samples", action="store_true")
    args = parser.parse_args()
    SOURCE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "COMP9444-academic-dataset-inspection/1.0"
    item_response = session.get(f"{API}/items/{ITEM}", timeout=45)
    item_response.raise_for_status()
    item = item_response.json()
    (SOURCE / "drum_item.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    bundle_response = session.get(item["_links"]["bundles"]["href"], timeout=45)
    bundle_response.raise_for_status()
    bundles = bundle_response.json()["_embedded"]["bundles"]
    original = next(b for b in bundles if b["name"] == "ORIGINAL")
    response = session.get(original["_links"]["bitstreams"]["href"], params={"size": 100}, timeout=45)
    response.raise_for_status()
    streams = response.json()["_embedded"]["bitstreams"]
    manifest = [{"name": b["name"], "uuid": b["uuid"], "sizeBytes": b["sizeBytes"], "url": b["_links"]["content"]["href"]} for b in streams]
    (SOURCE / "drum_files.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"name": item["name"], "files": manifest}, indent=2), flush=True)
    archive = next(b for b in manifest if b["name"] == "dataset.zip")
    with zipfile.ZipFile(RemoteZip(session, archive["url"], archive["sizeBytes"])) as zf:
        members = [{"name": i.filename, "size": i.file_size} for i in zf.infolist()]
        (SOURCE / "archive_members.json").write_text(json.dumps(members, indent=2), encoding="utf-8")
        selected = [i for i in zf.infolist() if ("/annotations/" not in i.filename) and i.filename.lower().endswith((".json", ".txt", ".md")) and "__MACOSX" not in i.filename]
        print(json.dumps({"archive_members": len(members), "metadata_members": [{"name": i.filename, "size": i.file_size} for i in selected]}, indent=2), flush=True)
        if args.extract:
            for member in selected:
                target = SOURCE / "original" / member.filename
                if not target.resolve().is_relative_to((SOURCE / "original").resolve()):
                    raise RuntimeError("Archive path escapes destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(zf.read(member))
                print(f"Saved {member.filename} ({member.file_size:,} bytes)", flush=True)
        if args.samples:
            images = [i for i in zf.infolist() if i.filename.lower().endswith((".jpg", ".jpeg", ".png")) and "/train/" in i.filename and "instance" in i.filename.lower()]
            for member in images[:6]:
                target = SOURCE / "samples" / Path(member.filename).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(zf.read(member))
                print(f"Saved sample {member.filename}", flush=True)


if __name__ == "__main__":
    main()
