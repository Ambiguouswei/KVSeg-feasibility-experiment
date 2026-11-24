import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

KVD = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
sys.argv = sys.argv[:1]

VERSION = 1
OP = {"ping": 1, "put": 2, "get": 3, "lookup": 4, "retire": 5,
      "seal": 6, "gc": 7, "stats": 8, "shutdown": 9}
ST_OK, ST_INVALID, ST_MISSING, ST_CONFLICT = 0, 1, 2, 3
ST_CAPACITY, ST_CORRUPTION, ST_BUSY, ST_IO, ST_DECLINED = 4, 5, 6, 7, 8


def digest(*parts):
    """构造一个合法的 64 位十六进制身份，与服务端的 IsDigest 校验对齐。"""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode() if isinstance(part, str) else part)
    return h.hexdigest()


class Wire:
    """独立于 C++ 实现的协议编解码，用于交叉验证线上格式。"""

    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(20)
        self.sock.connect(path)
        self.buf = b""

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def _recv_exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self.buf)))
            if not chunk:
                raise ConnectionError("daemon closed the connection")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def call(self, op, body=b""):
        frame = struct.pack("<I", len(body) + 2) + bytes([VERSION, OP[op]]) + body
        self.sock.sendall(frame)
        (length,) = struct.unpack("<I", self._recv_exact(4))
        if length == 0 or length > 64 * 1024 * 1024:
            raise ValueError(f"invalid response length {length}")
        payload = self._recv_exact(length)
        return payload[0], payload[1:]

    # ---- 编码 ----
    @staticmethod
    def s(text):
        raw = text.encode() if isinstance(text, str) else text
        return struct.pack("<I", len(raw)) + raw

    @staticmethod
    def key(space, dg, parent=""):
        return Wire.s(space) + Wire.s(dg) + Wire.s(parent)

    # ---- 解码 ----
    @staticmethod
    def rd_str(data, pos):
        (n,) = struct.unpack_from("<I", data, pos)
        pos += 4
        return data[pos:pos + n], pos + n

    @staticmethod
    def rd_u64(data, pos=0):
        return struct.unpack_from("<Q", data, pos)[0]


class DaemonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="kvd-")
        self.root = Path(self.tmp.name) / "root"
        self.root.mkdir()
        self.sock_path = str(Path(self.tmp.name) / "kvd.sock")
        self.proc = subprocess.Popen(
            [str(KVD), str(self.root), "--socket", self.sock_path,
             "--segment-kib", "64", "--payload-budget-kib", "256"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._wait_ready()
        self.space = digest("space", "tenant-a")
        self.clients = []

    def tearDown(self):
        for wire in self.clients:
            wire.close()
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        out = self.proc.stdout.read() if self.proc.stdout else ""
        err = self.proc.stderr.read() if self.proc.stderr else ""
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream is not None:
                stream.close()
        self.tmp.cleanup()
        if self.proc.returncode not in (0, -9, -15, 137):
            self.fail(f"kvd exited with {self.proc.returncode}\nstdout={out}\nstderr={err}")

    def _wait_ready(self):
        for _ in range(200):
            if os.path.exists(self.sock_path):
                return
            if self.proc.poll() is not None:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                self.fail(f"kvd exited during startup: {err}")
            time.sleep(0.05)
        self.fail("socket did not appear within 10s")

    def connect(self):
        wire = Wire(self.sock_path)
        self.clients.append(wire)
        return wire

    # ---- 用例 ----

    def test_ping_and_version(self):
        status, body = self.connect().call("ping")
        self.assertEqual(status, ST_OK)
        self.assertIn(b"kvcohort/daemon", body)

    def test_put_get_roundtrip(self):
        wire = self.connect()
        root = digest("b1")
        payload = bytes(range(256)) * 8
        status, body = wire.call("put", Wire.key("", "", "")[:0] + Wire.key(self.space, root)
                                 + bytes([0]) + Wire.s("") + Wire.s(payload))
        self.assertEqual(status, ST_OK, body)
        generation = Wire.rd_u64(body)
        self.assertGreater(generation, 0)

        status, body = wire.call("get", Wire.key(self.space, root))
        self.assertEqual(status, ST_OK, body)
        self.assertEqual(body, payload)

    def test_get_missing_returns_missing_status(self):
        wire = self.connect()
        status, _ = wire.call("get", Wire.key(self.space, digest("nope")))
        self.assertEqual(status, ST_MISSING)

    def test_conflicting_content_is_rejected(self):
        wire = self.connect()
        key = Wire.key(self.space, digest("b2"))
        self.assertEqual(wire.call("put", key + bytes([0]) + Wire.s("") + Wire.s(b"A" * 4096))[0], ST_OK)
        status, _ = wire.call("put", key + bytes([0]) + Wire.s("") + Wire.s(b"B" * 4096))
        self.assertEqual(status, ST_CONFLICT)

    def test_prefix_stops_at_gap(self):
        wire = self.connect()
        b0, b1, b2 = digest("p0"), digest("p1"), digest("p2")
        for dg, parent, tag in ((b0, "", "0"), (b1, b0, "1"), (b2, b1, "2")):
            status, _ = wire.call("put", Wire.key(self.space, dg, parent)
                                  + bytes([2]) + Wire.s("") + Wire.s(tag * 4096))
            self.assertEqual(status, ST_OK)
        # 完整链应当返回 3 段
        chain = struct.pack("<I", 3) + Wire.key(self.space, b0) \
            + Wire.key(self.space, b1, b0) + Wire.key(self.space, b2, b1)
        status, body = wire.call("lookup", chain)
        self.assertEqual(status, ST_OK, body)
        (count,) = struct.unpack_from("<I", body, 0)
        self.assertEqual(count, 3)
        # 删掉中间块后，只能返回 1 段
        self.assertEqual(wire.call("retire", Wire.key(self.space, b1, b0))[0], ST_OK)
        status, body = wire.call("lookup", chain)
        self.assertEqual(status, ST_OK)
        (count,) = struct.unpack_from("<I", body, 0)
        self.assertEqual(count, 1)

    def test_retire_then_get_missing(self):
        wire = self.connect()
        dg = digest("b3")
        self.assertEqual(wire.call("put", Wire.key(self.space, dg) + bytes([0])
                                   + Wire.s("") + Wire.s(b"C" * 2048))[0], ST_OK)
        self.assertEqual(wire.call("retire", Wire.key(self.space, dg))[0], ST_OK)
        _status, body = wire.call("retire", Wire.key(self.space, dg))
        self.assertEqual(body, b"\x00")
        self.assertEqual(wire.call("get", Wire.key(self.space, dg))[0], ST_MISSING)

    def test_stats_is_json(self):
        wire = self.connect()
        self.assertEqual(wire.call("put", Wire.key(self.space, digest("b4"))
                                   + bytes([0]) + Wire.s("") + Wire.s(b"D" * 1024))[0], ST_OK)
        status, body = wire.call("stats")
        self.assertEqual(status, ST_OK)
        payload, _ = Wire.rd_str(body, 0)
        stats = json.loads(payload)
        self.assertEqual(stats["objects"], 1)
        self.assertGreater(stats["allocated_bytes"], 0)

    def test_two_clients_share_state(self):
        first, second = self.connect(), self.connect()
        dg = digest("shared")
        self.assertEqual(first.call("put", Wire.key(self.space, dg) + bytes([0])
                                    + Wire.s("") + Wire.s(b"E" * 512))[0], ST_OK)
        status, body = second.call("get", Wire.key(self.space, dg))
        self.assertEqual(status, ST_OK)
        self.assertEqual(body, b"E" * 512)

    def test_gc_and_seal_are_accepted(self):
        wire = self.connect()
        self.assertEqual(wire.call("seal")[0], ST_OK)
        status, _ = wire.call("gc")
        self.assertEqual(status, ST_OK)

    def test_shutdown_stops_daemon(self):
        wire = self.connect()
        status, _ = wire.call("shutdown")
        self.assertEqual(status, ST_OK)
        self.proc.wait(timeout=10)
        self.assertEqual(self.proc.returncode, 0)


if __name__ == "__main__":
    if KVD is None:
        raise SystemExit("usage: daemon_test.py PATH_TO_KVD")
    unittest.main(verbosity=2)
