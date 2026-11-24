"""Exercise the Python wire client against a real, private kvd process."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from kvcohort_vllm.client import Client, DaemonError


def start_daemon(binary, data_root, socket_path, log_path):
    log = log_path.open("x", encoding="utf-8")
    proc = subprocess.Popen(
        [str(binary), str(data_root), "--socket", str(socket_path),
         "--layout", "joint", "--segment-kib", "4096"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(200):
        if socket_path.exists():
            return proc, log
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(f"private daemon exited: {proc.returncode}")
        time.sleep(0.05)
    proc.terminate()
    proc.wait(timeout=10)
    log.close()
    raise TimeoutError("private daemon did not become ready")


def stop_daemon(proc, log, socket_path):
    try:
        if proc.poll() is None:
            client = Client(str(socket_path))
            try:
                client.shutdown()
            finally:
                client.close()
            proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log.close()


def main():
    experiment = Path(sys.argv[1]).resolve()
    scratch = Path(sys.argv[2]).resolve()
    result_dir = experiment / "connector-client-test"
    runtime_dir = scratch / "connector-client-test"
    result_dir.mkdir()
    runtime_dir.mkdir()
    data_root = result_dir / "data"
    binary = experiment / "bin" / "kvd"
    space = hashlib.sha256(b"client test namespace").hexdigest()
    other_space = hashlib.sha256(b"another namespace").hexdigest()
    key = hashlib.sha256(b"client test key").hexdigest()
    group = hashlib.sha256(b"prefix group").hexdigest()
    payload = bytes(range(256)) * 3072

    first_socket = runtime_dir / "first.sock"
    proc, log = start_daemon(binary, data_root, first_socket, result_dir / "daemon-first.log")
    try:
        client = Client(str(first_socket))
        try:
            assert client.get(space, key) is None
            assert client.put(space, key, payload, group) > 0
            assert client.get(space, key) == payload
            assert client.get(other_space, key) is None
            try:
                client.put(space, key, b"different bytes", group)
            except DaemonError as error:
                assert error.status == 3, error
            else:
                raise AssertionError("conflicting immutable object was accepted")
            first_stats = client.stats()
        finally:
            client.close()
    finally:
        stop_daemon(proc, log, first_socket)

    second_socket = runtime_dir / "second.sock"
    proc, log = start_daemon(binary, data_root, second_socket, result_dir / "daemon-second.log")
    try:
        client = Client(str(second_socket))
        try:
            restored = client.get(space, key)
            assert restored == payload
            second_stats = client.stats()
        finally:
            client.close()
    finally:
        stop_daemon(proc, log, second_socket)

    result = {"passed": True, "payload_bytes": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest(),
              "verified": ["missing", "put/get", "namespace isolation",
                           "conflicting write rejected", "daemon restart and restore"],
              "before_restart": first_stats, "after_restart": second_stats}
    (result_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
