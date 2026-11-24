"""Compare identical private KV stores on /root overlay and the NFS workdir.

Only the experiment's own files are read or created. No system cache is
cleared. Results and logs are stored in the NFS experiment directory.
"""

import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

from run_benchmark import run_mode
from run_end_to_end import daemon_stats
from test_client import start_daemon, stop_daemon


def tree_hashes(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in experiment store: {path}")
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    h.update(block)
            files[str(path.relative_to(root))] = h.hexdigest()
    return files


def direct_read_once(root):
    """Read aligned portions of segment files with Linux O_DIRECT.

    NFS may still be served from the remote server's cache. This is a
    filesystem-level read probe, not a vLLM latency or SSD hardware test.
    """
    if not hasattr(os, "O_DIRECT"):
        raise OSError("O_DIRECT is not available on this platform")
    alignment, block_size = 4096, 1024 * 1024
    buffer = mmap.mmap(-1, block_size)
    total = 0
    started = time.perf_counter()
    try:
        paths = sorted((root / "segments").glob("*.seg"))
        if not paths:
            raise ValueError("no segment files found")
        for path in paths:
            remaining = path.stat().st_size // alignment * alignment
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                while remaining:
                    size = min(block_size, remaining)
                    view = memoryview(buffer)[:size]
                    try:
                        count = os.readv(fd, [view])
                    finally:
                        view.release()
                    if count <= 0 or count % alignment:
                        raise OSError("short or unaligned direct read")
                    remaining -= count
                    total += count
            finally:
                os.close(fd)
    finally:
        buffer.close()
    elapsed = time.perf_counter() - started
    return {"bytes": total, "seconds": elapsed,
            "mib_per_second": total / (1024 * 1024) / elapsed}


def direct_probe(stores):
    results = {"nfs": [], "root_overlay": []}
    errors = {}
    for name in ("nfs", "root_overlay", "root_overlay", "nfs",
                 "nfs", "root_overlay", "root_overlay", "nfs"):
        if name in errors:
            continue
        try:
            results[name].append(direct_read_once(stores[name]))
        except (OSError, ValueError) as error:
            errors[name] = str(error)
    return {"samples": results, "errors": errors,
            "description": "O_DIRECT aligned segment reads; client page cache bypass requested, remote caches may remain"}


def main():
    experiment, scratch = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    name = sys.argv[3]
    if not name.startswith("root-vs-nfs-") or "/" in name or "\\" in name:
        raise ValueError("invalid experiment name")
    source = experiment / "e2e-20261109-02" / "data"
    result_dir, runtime_dir = experiment / name, scratch / name
    result_dir.mkdir()
    runtime_dir.mkdir()
    stores = {"nfs": result_dir / "nfs-store", "root_overlay": runtime_dir / "root-store"}

    with (source / "OWNER.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_hashes = tree_hashes(source)
        for target in stores.values():
            shutil.copytree(source, target)
            if tree_hashes(target) != source_hashes:
                raise ValueError(f"copied store does not match source: {target}")
        fcntl.flock(lock, fcntl.LOCK_UN)
    print("Copied and verified identical stores", flush=True)

    direct = direct_probe(stores)
    (result_dir / "direct-read.json").write_text(json.dumps(direct, indent=2) + "\n")
    print("Completed direct-read probe", flush=True)

    binary = experiment / "bin" / "kvd"
    nfs_sock, root_sock = runtime_dir / "nfs.sock", runtime_dir / "root.sock"
    nfs_proc, nfs_log = start_daemon(binary, stores["nfs"], nfs_sock, result_dir / "nfs-daemon.log")
    try:
        root_proc, root_log = start_daemon(binary, stores["root_overlay"], root_sock,
                                          result_dir / "root-daemon.log")
        try:
            reference_dir = result_dir / "baseline"
            reference_dir.mkdir()
            baseline = run_mode(experiment, nfs_sock, reference_dir, "baseline")
            print("Completed baseline", flush=True)

            results = {"nfs": [], "root_overlay": []}
            for index, storage in enumerate(("nfs", "root_overlay", "root_overlay", "nfs"), 1):
                sock = nfs_sock if storage == "nfs" else root_sock
                folder = result_dir / f"{index:02d}-{storage}"
                folder.mkdir()
                before = daemon_stats(sock)
                measured = run_mode(experiment, sock, folder, "prefetch")
                after = daemon_stats(sock)
                bytes_read = after["demand_read_bytes"] - before["demand_read_bytes"]
                if bytes_read < 7 * 1024 * 1024:
                    raise AssertionError(f"no substantial backend read for {storage}")
                measured["backend_read_bytes"] = bytes_read
                results[storage].append(measured)
                print(f"Completed {index}/4: {storage}", flush=True)
        finally:
            stop_daemon(root_proc, root_log, root_sock)
    finally:
        stop_daemon(nfs_proc, nfs_log, nfs_sock)

    reference_ids = baseline["samples"][0]["output_token_ids"]
    for storage, phases in results.items():
        for phase in phases:
            for sample in phase["samples"]:
                if sample["cached_tokens"] != 608 or sample["output_token_ids"] != reference_ids:
                    raise AssertionError(f"unexpected cache hit or output for {storage}")
    summary = {"model": "Qwen/Qwen2.5-3B-Instruct FP16", "prompt_tokens": 620,
               "cached_tokens": 608, "output_tokens": 1,
               "metric": "warmed LLM.generate wall time for one token, not streaming TTFT",
               "prefetch_workers": 2, "samples_per_storage": 14,
               "phase_order": ["nfs", "root_overlay", "root_overlay", "nfs"],
               "baseline_median_seconds": baseline["median_one_token_response_seconds"],
               "stores_initially_byte_identical": True,
               "data_roots": {key: str(value) for key, value in stores.items()},
               "direct_read": direct,
               "note": "shared server; normal OS cache retained; root overlay physical backing not established"}
    for storage, phases in results.items():
        times = [sample["generate_seconds"] for phase in phases for sample in phase["samples"]]
        summary[storage] = {"median_seconds": statistics.median(times),
                            "min_seconds": min(times), "max_seconds": max(times),
                            "phase_medians_seconds": [phase["median_one_token_response_seconds"] for phase in phases],
                            "backend_read_bytes": sum(phase["backend_read_bytes"] for phase in phases),
                            "samples_seconds": times}
    summary["root_to_nfs_latency_ratio"] = (summary["root_overlay"]["median_seconds"] /
                                             summary["nfs"]["median_seconds"])
    path = result_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
