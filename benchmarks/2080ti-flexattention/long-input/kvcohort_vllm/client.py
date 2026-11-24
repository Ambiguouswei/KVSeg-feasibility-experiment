"""Small, explicit client for version 1 of the kvcohort daemon protocol."""

import json
import socket
import struct
import threading
from collections import OrderedDict


MAX_FRAME = 64 * 1024 * 1024
MAX_OBJECT = 1024 * 1024
VERSION = 1
MISSING = 2


class CPUCache:
    """Process-local, byte-bounded LRU for persisted KV chunks."""
    def __init__(self, capacity):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
            raise ValueError("cpu_cache_bytes must be a nonnegative integer")
        self.capacity = capacity
        self.bytes = 0
        self._items = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key, value):
        value = bytes(value)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.bytes -= len(old)
            if len(value) > self.capacity:
                return
            while self.bytes + len(value) > self.capacity:
                _, evicted = self._items.popitem(last=False)
                self.bytes -= len(evicted)
                self.evictions += 1
            self._items[key] = value
            self.bytes += len(value)


class CachedClient:
    """Write through to disk; read chunks from CPU before the daemon."""
    def __init__(self, client, cache):
        self.client, self.cache = client, cache

    def get(self, space, digest, cacheable=True):
        key = (space, digest)
        value = self.cache.get(key) if cacheable else None
        if value is None:
            value = self.client.get(space, digest)
            if value is not None and cacheable:
                self.cache.put(key, value)
        return value

    def put(self, space, digest, payload, prefix_group=""):
        result = self.client.put(space, digest, payload, prefix_group)
        self.cache.put((space, digest), payload)
        return result

    def close(self):
        self.client.close()


class DaemonError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"kvcohort daemon status {status}: {message}")
        self.status = status


def _str(value):
    value = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<I", len(value)) + value


class Client:
    def __init__(self, path, timeout=120):
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        self._socket.connect(path)

    def close(self):
        self._socket.close()

    def _read_exact(self, size):
        out = bytearray(size)
        view = memoryview(out)
        while view:
            count = self._socket.recv_into(view)
            if count == 0:
                raise ConnectionError("kvcohort daemon closed its connection")
            view = view[count:]
        return bytes(out)

    def _call(self, op, body=b""):
        if len(body) + 2 > MAX_FRAME:
            raise ValueError("request exceeds protocol frame limit")
        self._socket.sendall(struct.pack("<I", len(body) + 2)
                             + bytes((VERSION, op)) + body)
        length = struct.unpack("<I", self._read_exact(4))[0]
        if not 1 <= length <= MAX_FRAME:
            raise ValueError("invalid response frame length")
        response = self._read_exact(length)
        status, payload = response[0], response[1:]
        if status:
            raise DaemonError(status, payload.decode("utf-8", "replace"))
        return payload

    @staticmethod
    def _key(space, digest, parent=""):
        return _str(space) + _str(digest) + _str(parent)

    def put(self, space, digest, payload, prefix_group=""):
        if not 0 < len(payload) <= MAX_OBJECT:
            raise ValueError("object must be between 1 byte and 1 MiB")
        body = (self._key(space, digest) + b"\x00" + _str(prefix_group)
                + _str(payload))
        response = self._call(2, body)
        if len(response) != 8:
            raise ValueError("invalid PUT response")
        return struct.unpack("<Q", response)[0]

    def get(self, space, digest):
        try:
            return self._call(3, self._key(space, digest))
        except DaemonError as exc:
            if exc.status == MISSING:
                return None
            raise

    def stats(self):
        payload = self._call(8)
        if len(payload) < 4:
            raise ValueError("invalid stats response")
        size = struct.unpack_from("<I", payload)[0]
        if len(payload) != size + 4:
            raise ValueError("invalid stats response length")
        return json.loads(payload[4:])

    def shutdown(self):
        self._call(9)
