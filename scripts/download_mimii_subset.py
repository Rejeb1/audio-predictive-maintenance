"""Fetch a fixed MIMII subset out of the Zenodo zips (one microphone kept). Resumable.

Per machine ID: 200 normal + 50 abnormal clips, the same file numbers at every SNR.
"""
import io
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from asd.config import DATA, SEED
from list_mimii import URL, HttpFile

COUNTS = {"normal": 200, "abnormal": 50}
local = threading.local()


def fetch(url, info, dest):
    if dest.exists():
        return
    for attempt in range(8):
        try:
            if getattr(local, "url", None) != url:
                local.f, local.url = HttpFile(url), url
            # One request: local header (30 bytes + name + extra) then the data.
            local.f.seek(info.header_offset)
            raw = local.f.read(30 + len(info.filename.encode()) + 512 + info.compress_size)
            assert raw[:4] == b"PK\x03\x04"
            n, e = struct.unpack("<HH", raw[26:30])
            body = raw[30 + n + e:30 + n + e + info.compress_size]
            assert len(body) == info.compress_size
            wav = body if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(body, -15)
            assert zlib.crc32(wav) == info.CRC
            y, sr = sf.read(io.BytesIO(wav), dtype="int16", always_2d=True)
            tmp = dest.with_suffix(".tmp")
            sf.write(tmp, y[:, 0], sr, format="WAV", subtype="PCM_16")
            tmp.replace(dest)
            return
        except Exception as ex:
            if attempt == 7:
                raise
            print(f"retry {dest.name}: {ex!r}", flush=True)
            time.sleep(min(120, 5 * 2 ** attempt))


def index(url):
    for attempt in range(8):
        try:
            return {i.filename: i for i in zipfile.ZipFile(HttpFile(url)).infolist()
                    if i.filename.endswith(".wav")}
        except Exception as ex:
            if attempt == 7:
                raise
            print(f"retry index: {ex!r}", flush=True)
            time.sleep(min(120, 5 * 2 ** attempt))


if __name__ == "__main__":
    for machine in ("pump", "fan"):
        zips = {s: index(URL.format(f"{s}_dB_{machine}.zip")) for s in ("-6", "0", "6")}
        common = sorted(set.intersection(*map(set, zips.values())))
        rng, chosen = np.random.default_rng(SEED), []
        for mid in ("id_00", "id_02", "id_04", "id_06"):
            for lab, n in COUNTS.items():
                names = [x for x in common if x.startswith(f"{machine}/{mid}/{lab}/")]
                chosen += sorted(rng.choice(names, n, replace=False).tolist())
        for snr, infos in zips.items():
            t0, jobs = time.time(), []
            for name in chosen:
                dest = DATA / f"mimii_{snr}dB" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((URL.format(f"{snr}_dB_{machine}.zip"), infos[name], dest))
            with ThreadPoolExecutor(4) as ex:
                list(ex.map(lambda j: fetch(*j), jobs))
            print(f"{machine} {snr} dB: {len(jobs)} clips in {time.time() - t0:.0f}s", flush=True)
