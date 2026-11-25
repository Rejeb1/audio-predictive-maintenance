"""List the contents of remote MIMII zips (reads only their index, not the audio)."""
import collections
import io
import urllib.request
import zipfile

URL = "https://zenodo.org/records/3384388/files/{}?download=1"


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a URL using HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: 0, 1: self.pos, 2: self.size}[whence] + off
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            if r.status != 206:
                raise OSError("server ignored the range request")
            data = r.read()
        self.pos += len(data)
        return data


if __name__ == "__main__":
    for machine in ("pump", "fan"):
        for snr in ("-6", "0", "6"):
            z = zipfile.ZipFile(HttpFile(URL.format(f"{snr}_dB_{machine}.zip")))
            g = collections.defaultdict(lambda: [0, 0])
            for i in z.infolist():
                if i.filename.endswith(".wav"):
                    k = "/".join(i.filename.split("/")[:-1])
                    g[k][0] += 1
                    g[k][1] += i.compress_size
            print(f"{snr} dB {machine}")
            for k, (n, b) in sorted(g.items()):
                print(f"  {k:26s} {n:5d} files {b / 1e9:6.2f} GB", flush=True)
