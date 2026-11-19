"""Fetch a cross-dataset generalisation set (never used for training or tuning).

Sources (all from the Open3D sample-data releases on GitHub, because the original
dataset servers were unreachable from our build environment):

* Stanford Lounge RGB-D  (Zhou & Koltun)  - 25 frames, evenly spaced over 3,000
* Redwood Bedroom RGB-D  (bedroom01 part) - 25 frames, evenly spaced over 5,000
* Stanford Fountain RGB-D (Zhou & Koltun) - all 33 frames (close-range sculpture)
* Augmented ICL-NUIM (Choi et al., "Redwood indoor") living room 1 and office 1 -
  25 frames each, evenly spaced. These are *synthetic* renders, but they come with
  exact, hole-free ground-truth depth and a simulated depth-sensor stream (noise
  and holes from a real sensor model), so they allow an evaluation against true GT.
  (The 5-frame "SampleRedwoodRGBDImages" set is living room 1 again, so it is not
  added a second time.)

Only the selected frames are read out of the large archives, using HTTP range
requests (the Lounge archive is 1.8 GB; we download a few MB of it).

Every frame is re-encoded as 320x240 (RGB: area, depth: nearest, like the NYU
pipeline) PNG with depth in millimetres. For the real-sensor sources the reference
is the *raw sensor depth*, which has holes (excluded from metrics) and its own noise.
For ICL-NUIM, `_depth.png` is the clean ground truth and `_simdepth.png` the
simulated sensor depth.

Usage:
    python scripts/fetch_cross_dataset.py            # -> data/cross/
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import cv2
import numpy as np
import requests

BASE = "https://github.com/isl-org/open3d_downloads/releases/download/"
OUT = "data/cross"
SIZE = (320, 240)
BLK = 1 << 20


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using explicit byte ranges."""

    def __init__(self, url):
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.url, self.size, self.pos, self.cache = r.url, int(r.headers["Content-Range"].split("/")[1]), 0, {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            a = b * BLK
            e = min(a + BLK, self.size) - 1
            last = None
            for _ in range(5):
                try:
                    r = requests.get(self.url, headers={"Range": f"bytes={a}-{e}"}, timeout=120)
                    r.raise_for_status()
                    break
                except Exception as ex:  # transient network error: retry
                    last = ex
            else:
                raise last
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[b] = r.content
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            b, o = divmod(self.pos, BLK)
            d = self._block(b)[o:o + n]
            out += d
            self.pos += len(d)
            n -= len(d)
        return bytes(out)


def remote_zip(rel):
    return zipfile.ZipFile(RangeFile(BASE + rel))


def decode(buf, flag):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), flag)


def save(src, idx, rgb, depth_mm, meta, sim_mm=None):
    rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)
    d = cv2.resize(depth_mm, SIZE, interpolation=cv2.INTER_NEAREST).astype(np.uint16)
    os.makedirs(os.path.join(OUT, src), exist_ok=True)
    fid = f"{src}_{idx:03d}"
    cv2.imwrite(os.path.join(OUT, src, f"{fid}_rgb.png"), rgb)
    cv2.imwrite(os.path.join(OUT, src, f"{fid}_depth.png"), d)
    rec = {"id": fid, "source": src, "holes_pct": round(float((d == 0).mean() * 100), 2)}
    if sim_mm is not None:
        sd = cv2.resize(sim_mm, SIZE, interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        cv2.imwrite(os.path.join(OUT, src, f"{fid}_simdepth.png"), sd)
        rec["sim_holes_pct"] = round(float((sd == 0).mean() * 100), 2)
    meta.append(rec)


def evenly(items, k):
    return [items[int(round(i))] for i in np.linspace(0, len(items) - 1, k)] if len(items) > k else items


def fetch_pairs(rel, src, colour_dir, depth_dir, k, meta):
    z = remote_zip(rel)
    names = z.namelist()
    col = sorted(n for n in names if n.startswith(colour_dir) and not n.endswith("/") and n[-4:] in (".png", ".jpg"))
    dep = sorted(n for n in names if n.startswith(depth_dir) and n.endswith(".png"))
    if len(col) != len(dep):  # pair by frame stem when counts differ
        stems = {os.path.splitext(os.path.basename(n))[0]: n for n in dep}
        pairs = [(c, stems[os.path.splitext(os.path.basename(c))[0]]) for c in col
                 if os.path.splitext(os.path.basename(c))[0] in stems]
    else:
        pairs = list(zip(col, dep))
    for i, (c, d) in enumerate(evenly(pairs, k)):
        save(src, i, decode(z.read(c), cv2.IMREAD_COLOR), decode(z.read(d), cv2.IMREAD_UNCHANGED), meta)
        print(src, i, os.path.basename(c), os.path.basename(d), flush=True)


def fetch_icl(scene, src, k, meta):
    z = {t: remote_zip(f"augmented-icl-nuim/{scene}-{t}.zip") for t in ("color", "depth-clean", "depth-simulated")}
    names = {t: sorted(n for n in z[t].namelist() if n[-4:] in (".jpg", ".png")) for t in z}
    assert len(names["color"]) == len(names["depth-clean"]) == len(names["depth-simulated"])
    idx = evenly(list(range(len(names["color"]))), k)
    for i, j in enumerate(idx):
        save(src, i, decode(z["color"].read(names["color"][j]), cv2.IMREAD_COLOR),
             decode(z["depth-clean"].read(names["depth-clean"][j]), cv2.IMREAD_UNCHANGED), meta,
             sim_mm=decode(z["depth-simulated"].read(names["depth-simulated"][j]), cv2.IMREAD_UNCHANGED))
        print(src, i, names["color"][j], flush=True)


def main():
    meta = []
    fetch_pairs("20220301-data/LoungeRGBDImages.zip", "lounge", "color/", "depth/", 25, meta)
    fetch_pairs("20220301-data/bedroom01.zip", "bedroom", "image/", "depth/", 25, meta)
    # Fountain: colour and depth names carry separate timestamps; sorted order pairs them (~6 ms apart)
    fetch_pairs("20220201-data/SampleFountainRGBDImages.zip", "fountain", "image/", "depth/", 33, meta)
    fetch_icl("livingroom1", "icl_living", 25, meta)
    fetch_icl("office1", "icl_office", 25, meta)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump({"note": "depth in mm, 320x240; real sources: raw sensor reference; icl_*: clean GT + _simdepth; never used for training or tuning",
                   "sources": {"lounge": "Stanford Lounge RGB-D (Open3D LoungeRGBDImages)",
                               "bedroom": "Redwood Bedroom RGB-D (Open3D BedroomRGBDImages, bedroom01)",
                               "fountain": "Stanford Fountain RGB-D (Open3D SampleFountainRGBDImages)",
                               "icl_living": "Augmented ICL-NUIM living room 1 (synthetic; clean GT + simulated sensor depth)",
                               "icl_office": "Augmented ICL-NUIM office 1 (synthetic; clean GT + simulated sensor depth)"},
                   "frames": meta}, f, indent=1)
    print(len(meta), "frames")


if __name__ == "__main__":
    main()
