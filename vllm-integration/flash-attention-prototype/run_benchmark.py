"""Compare no external cache with the persisted kvcohort connector."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from run_end_to_end import assert_gpu_free, daemon_stats
from test_client import start_daemon, stop_daemon


def run_mode(experiment, sock, result_dir, mode):
    assert_gpu_free()
    command = [sys.executable, "-B", str(experiment / "integration" / "benchmark_inference.py"),
               str(experiment), str(sock), str(result_dir), mode]
    with (result_dir / f"{mode}.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            status = proc.wait(timeout=600)
            if status:
                raise RuntimeError(f"{mode} exited with status {status}; inspect {log.name}")
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
    return json.loads((result_dir / f"{mode}.json").read_text())


def main():
    experiment = Path(sys.argv[1]).resolve()
    scratch = Path(sys.argv[2]).resolve()
    name = sys.argv[3]
    if not name.startswith("bench-") or "/" in name or "\\" in name:
        raise ValueError("invalid experiment name")
    result_dir, runtime_dir = experiment / name, scratch / name
    result_dir.mkdir()
    runtime_dir.mkdir()
    sock = runtime_dir / "kvd.sock"
    root = experiment / "e2e-20261109-01" / "data"
    binary = experiment / "bin" / "kvd"
    proc, log = start_daemon(binary, root, sock, result_dir / "daemon.log")
    try:
        before = daemon_stats(sock)
        baseline = run_mode(experiment, sock, result_dir, "baseline")
        restore = run_mode(experiment, sock, result_dir, "restore")
        prefetch = run_mode(experiment, sock, result_dir, "prefetch")
        bypass = run_mode(experiment, sock, result_dir, "bypass")
        after = daemon_stats(sock)
    finally:
        stop_daemon(proc, log, sock)

    no_cache = baseline["median_one_token_response_seconds"]
    external = restore["median_one_token_response_seconds"]
    prefetch_time = prefetch["median_one_token_response_seconds"]
    bypass_time = bypass["median_one_token_response_seconds"]
    equal_output = all(a["output_token_ids"] == b["output_token_ids"]
                       for other in (restore, prefetch, bypass)
                       for a, b in zip(baseline["samples"], other["samples"]))
    read_bytes = after["demand_read_bytes"] - before["demand_read_bytes"]
    summary = {"model": "Qwen/Qwen2.5-3B-Instruct FP16", "prompt_tokens": 620,
               "output_tokens": 1, "samples_per_mode": 7,
               "metric": "warmed LLM.generate wall time for one token; not streaming TTFT",
               "baseline_median_seconds": no_cache,
               "kvcohort_median_seconds": external,
               "two_reader_prefetch_median_seconds": prefetch_time,
               "short_prefix_bypass_median_seconds": bypass_time,
               "restore_to_baseline_ratio": external / no_cache,
               "prefetch_to_sequential_restore_ratio": prefetch_time / external,
               "bypass_threshold_tokens": 1024,
               "matched_tokens_per_restore": 608,
               "backend_read_bytes": read_bytes,
               "output_tokens_match": equal_output,
               "storage": "network filesystem with normal OS caching; not an SSD benchmark"}
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not equal_output or read_bytes < 14 * 1024 * 1024:
        raise AssertionError("comparison did not verify matching output and external reads")


if __name__ == "__main__":
    main()
