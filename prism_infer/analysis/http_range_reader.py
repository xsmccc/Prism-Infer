"""严格校验 HTTP Range 的只读、可 seek 文件适配器。

用途是让标准库 :mod:`zipfile` 只读取远端大型 ZIP 的 central directory 和被选中的
member。服务端若忽略 Range、返回错误区间或内容长度变化，本模块立即失败，避免把
11GB archive 意外当成一个“小范围请求”下载。
"""

from __future__ import annotations

import io
import re
from collections import OrderedDict
from typing import Any


DEFAULT_RANGE_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_CACHE_CHUNKS = 4
CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    """解析严格的 ``bytes start-end/total`` 响应头。"""

    if value is None:
        raise ValueError("HTTP 206 response has no Content-Range")
    match = CONTENT_RANGE_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid Content-Range header: {value!r}")
    start, end, total = (int(group) for group in match.groups())
    if start < 0 or end < start or total <= end:
        raise ValueError(f"inconsistent Content-Range header: {value!r}")
    return start, end, total


class HTTPRangeReader(io.RawIOBase):
    """将支持 HTTP 206 的 URL 暴露为带小型 LRU cache 的 seekable reader。"""

    def __init__(
        self,
        url: str,
        *,
        expected_size: int,
        chunk_bytes: int = DEFAULT_RANGE_CHUNK_BYTES,
        cache_chunks: int = DEFAULT_CACHE_CHUNKS,
        connect_timeout_seconds: float = 20.0,
        read_timeout_seconds: float = 60.0,
        retries: int = 5,
    ) -> None:
        super().__init__()
        if not url:
            raise ValueError("HTTP range URL must be non-empty")
        for name, value in (
            ("expected_size", expected_size),
            ("chunk_bytes", chunk_bytes),
            ("cache_chunks", cache_chunks),
            ("retries", retries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
        except ImportError as exc:
            raise RuntimeError(
                "HTTP range materialization requires optional dependency requests"
            ) from exc

        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(("GET",)),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session: Any = requests.Session()
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._source_url = url
        self._resolved_url = url
        self._expected_size = expected_size
        self._chunk_bytes = chunk_bytes
        self._cache_chunks = cache_chunks
        self._timeout = (connect_timeout_seconds, read_timeout_seconds)
        self._position = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self.range_requests = 0
        self.range_response_bytes = 0
        self.cache_hits = 0
        self._resolve()

    @property
    def size(self) -> int:
        return self._expected_size

    @property
    def resolved_url(self) -> str:
        return self._resolved_url

    def _range_get(self, url: str, start: int, end: int) -> Any:
        response = self._session.get(
            url,
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
            },
            timeout=self._timeout,
            allow_redirects=True,
        )
        if response.status_code != 206:
            response.close()
            raise OSError(
                "remote server did not honor Range: "
                f"status={response.status_code}, requested={start}-{end}"
            )
        actual_start, actual_end, total = parse_content_range(
            response.headers.get("Content-Range")
        )
        if (actual_start, actual_end) != (start, end):
            response.close()
            raise OSError(
                "remote server returned the wrong byte interval: "
                f"requested={start}-{end}, actual={actual_start}-{actual_end}"
            )
        if total != self._expected_size:
            response.close()
            raise OSError(
                "remote object size differs from frozen archive metadata: "
                f"expected={self._expected_size}, actual={total}"
            )
        return response

    def _resolve(self) -> None:
        response = self._range_get(self._source_url, 0, 0)
        try:
            payload = response.content
            if len(payload) != 1:
                raise OSError("one-byte range probe returned the wrong payload length")
            self._resolved_url = response.url
        finally:
            response.close()

    def _fetch_chunk(self, chunk_index: int) -> bytes:
        cached = self._cache.get(chunk_index)
        if cached is not None:
            self._cache.move_to_end(chunk_index)
            self.cache_hits += 1
            return cached
        start = chunk_index * self._chunk_bytes
        if start >= self._expected_size:
            return b""
        end = min(start + self._chunk_bytes, self._expected_size) - 1
        # Hugging Face Xet signs the requested byte interval into its redirect.
        # A URL obtained by the 0-0 probe therefore cannot be reused for another
        # range. Always enter through the immutable revision URL and retain the
        # latest resolved URL only as diagnostic metadata.
        response = self._range_get(self._source_url, start, end)
        self._resolved_url = response.url
        try:
            payload = response.content
        finally:
            response.close()
        expected_length = end - start + 1
        if len(payload) != expected_length:
            raise OSError(
                "range payload length mismatch: "
                f"expected={expected_length}, actual={len(payload)}"
            )
        self.range_requests += 1
        self.range_response_bytes += len(payload)
        self._cache[chunk_index] = payload
        self._cache.move_to_end(chunk_index)
        while len(self._cache) > self._cache_chunks:
            self._cache.popitem(last=False)
        return payload

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._expected_size + offset
        else:
            raise ValueError(f"unsupported seek whence: {whence}")
        if position < 0:
            raise ValueError("cannot seek before the start of the remote object")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed HTTPRangeReader")
        if self._position >= self._expected_size:
            return b""
        if size is None or size < 0:
            size = self._expected_size - self._position
        if size == 0:
            return b""
        end_position = min(self._position + size, self._expected_size)
        chunks = []
        while self._position < end_position:
            chunk_index = self._position // self._chunk_bytes
            chunk = self._fetch_chunk(chunk_index)
            chunk_offset = self._position % self._chunk_bytes
            take = min(len(chunk) - chunk_offset, end_position - self._position)
            if take <= 0:
                raise OSError("range reader made no forward progress")
            chunks.append(chunk[chunk_offset : chunk_offset + take])
            self._position += take
        return b"".join(chunks)

    def close(self) -> None:
        if not self.closed:
            self._cache.clear()
            self._session.close()
        super().close()
