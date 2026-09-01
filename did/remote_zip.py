"""Random-access reader for a remote ZIP over HTTP range requests (ModelScope/OSS)."""
import io, urllib.request, zipfile

class HttpRangeFile(io.RawIOBase):
    """Seekable file-like over HTTP range requests, with a read-ahead buffer so
    zipfile's small 4 KB reads don't each become a network round-trip."""

    def __init__(self, url, timeout=60, chunk=1 << 22):
        self.url = url; self.timeout = timeout; self._pos = 0; self._chunk = chunk
        self._buf = b""; self._buf_start = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self._size = int(r.headers["Content-Length"])

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        last = RuntimeError("unreachable")
        for _ in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except Exception as e:  # noqa: BLE001
                last = e
        raise last

    def seek(self, off, whence=0):
        if whence == 0: self._pos = off
        elif whence == 1: self._pos += off
        elif whence == 2: self._pos = self._size + off
        return self._pos

    def tell(self): return self._pos
    def seekable(self): return True
    def readable(self): return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + n, self._size)
        # ensure buffer covers [self._pos, end)
        if not (self._buf_start <= self._pos and end <= self._buf_start + len(self._buf)):
            fetch_start = self._pos
            fetch_end = min(max(self._pos + self._chunk, end), self._size) - 1
            self._buf = self._fetch(fetch_start, fetch_end)
            self._buf_start = fetch_start
        lo = self._pos - self._buf_start
        data = self._buf[lo: lo + (end - self._pos)]
        self._pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def open_remote_zip(url):
    return zipfile.ZipFile(HttpRangeFile(url), "r")

DATASET_URL = "https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath="
def wildfake_url(path): return DATASET_URL + path

if __name__ == "__main__":
    import sys, collections
    zf = open_remote_zip(wildfake_url(sys.argv[1]))
    names = [n for n in zf.namelist() if not n.endswith("/")]
    print("total entries:", len(names))
    pref = collections.Counter("/".join(n.split("/")[:4]) for n in names)
    for k, v in pref.most_common(30): print(v, k)
    print("sample:", names[:5])
