"""Run a private daemon and two separate single-GPU vLLM processes."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from kvcohort_vllm.client import Client
from test_client import start_daemon, stop_daemon


GPU_UUID = "GPU-9fa2c898-5a2b-b4c3-66d4-2afcd0aa1adb"


def assert_gpu_free():
    gpu = subprocess.check_output(
        ["nvidia-smi", "-i", "2", "--query-gpu=uuid,memory.used",
         "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"], text=True)
    if gpu[0] != GPU_UUID or int(gpu[1]) > 512 or GPU_UUID in processes:
        raise RuntimeError("third GPU is not free; did not start inference")


def daemon_stats(socket_path):
    client = Client(str(socket_path))
    try:
        return client.stats()
    finally:
        client.close()


def run_engine(experiment, socket_path, result_dir, mode):
    assert_gpu_free()
    log_path = result_dir / f"{mode}.log"
    command = [sys.executable, "-B", str(experiment / "integration" / "run_inference.py"),
               str(experiment), str(socket_path), str(result_dir), mode]
    with log_path.open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            status = proc.wait(timeout=600)
            if status:
                raise RuntimeError(f"{mode} inference failed with status {status}: {log_path}")
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            raise
    return json.loads((result_dir / f"{mode}-result.json").read_text())


def main():
    experiment = Path(sys.argv[1]).resolve()
    scratch = Path(sys.argv[2]).resolve()
    name = sys.argv[3]
    if not name.startswith("e2e-") or "/" in name or "\\" in name:
        raise ValueError("invalid experiment name")
    result_dir, runtime_dir = experiment / name, scratch / name
    result_dir.mkdir()
    runtime_dir.mkdir()
    data_root = result_dir / "data"
    binary = experiment / "bin" / "kvd"

    writer_socket = runtime_dir / "writer.sock"
    proc, log = start_daemon(binary, data_root, writer_socket, result_dir / "daemon-writer.log")
    try:
        writer = run_engine(experiment, writer_socket, result_dir, "writer")
        before_restart = daemon_stats(writer_socket)
    finally:
        stop_daemon(proc, log, writer_socket)

    reader_socket = runtime_dir / "reader.sock"
    proc, log = start_daemon(binary, data_root, reader_socket, result_dir / "daemon-reader.log")
    try:
        reader = run_engine(experiment, reader_socket, result_dir, "reader")
        after_restart = daemon_stats(reader_socket)
    finally:
        stop_daemon(proc, log, reader_socket)

    first, restored = writer["same_prompt"], reader["same_prompt"]
    result = {"model": "Qwen/Qwen2.5-3B-Instruct", "precision": "FP16",
              "storage": "kvcohort on the experiment's network filesystem",
              "built_in_vllm_prefix_cache": False,
              "daemon_restarted_between_runs": True,
              "writer": writer, "reader": reader,
              "before_daemon_restart": before_restart,
              "after_daemon_restart": after_restart,
              "same_output_tokens": first["output_token_ids"] == restored["output_token_ids"],
              "backend_retrieved_bytes": after_restart["demand_read_bytes"]}
    (result_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if first["cached_tokens"] != 0 or not restored["cached_tokens"]:
        raise AssertionError("expected a miss before writing and an external hit after restart")
    if reader["different_prompt"]["cached_tokens"] != 0:
        raise AssertionError("a different prefix was incorrectly reported as cached")
    if before_restart["payload_write_bytes"] < 1024 * 1024:
        raise AssertionError("no substantial KV payload was written")
    if after_restart["demand_read_bytes"] < 1024 * 1024:
        raise AssertionError("no substantial KV payload was read after restart")
    if not result["same_output_tokens"]:
        raise AssertionError("restored output token IDs differ from writer output")


if __name__ == "__main__":
    main()
