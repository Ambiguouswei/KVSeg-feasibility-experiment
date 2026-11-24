"""Check connector publication and restore with CPU tensors and real kvd."""

from pathlib import Path
from types import SimpleNamespace as NS
import sys

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    SharedStorageConnectorMetadata,
)

from kvcohort_vllm.connector import KVCohortConnector
from test_client import start_daemon, stop_daemon


class HFConfig:
    num_hidden_layers = 2

    def to_dict(self):
        return {"num_hidden_layers": 2, "model_type": "test"}


class TransferConfig:
    def __init__(self, socket_path, revision):
        self.kv_connector_extra_config = {"socket_path": str(socket_path),
                                          "model_revision": revision}

    def get_from_extra_config(self, key, default):
        return self.kv_connector_extra_config.get(key, default)


def config(socket_path, revision="test-revision", **extra):
    transfer = TransferConfig(socket_path, revision)
    transfer.kv_connector_extra_config.update(extra)
    return NS(cache_config=NS(block_size=16, cache_dtype="auto", enable_prefix_caching=False),
              model_config=NS(dtype=torch.float16, hf_config=HFConfig()),
              parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
              scheduler_config=NS(max_num_seqs=1), lora_config=None,
              kv_transfer_config=transfer)


def metadata(tokens, store):
    meta = SharedStorageConnectorMetadata()
    meta.add_request(token_ids=tokens, block_ids=[0, 1, 2], block_size=16,
                     is_store=store, mm_hashes=[])
    return meta


def main():
    experiment = Path(sys.argv[1]).resolve()
    scratch = Path(sys.argv[2]).resolve()
    name = sys.argv[3] if len(sys.argv) > 3 else "connector-cpu-test"
    if not name.startswith("connector-cpu-test") or "/" in name or "\\" in name:
        raise ValueError("invalid test name")
    result_dir = experiment / name
    runtime_dir = scratch / name
    result_dir.mkdir()
    runtime_dir.mkdir()
    root = result_dir / "data"
    binary = experiment / "bin" / "kvd"
    tokens = list(range(33))
    original = [torch.arange(2 * 3 * 16 * 2 * 4, dtype=torch.float16).reshape(2, 3, 16, 2, 4),
                torch.arange(2 * 3 * 16 * 2 * 4, dtype=torch.float16).reshape(2, 3, 16, 2, 4) + 3]

    sock = runtime_dir / "writer.sock"
    proc, log = start_daemon(binary, root, sock, result_dir / "daemon-writer.log")
    worker = KVCohortConnector(config(sock), KVConnectorRole.WORKER)
    scheduler = KVCohortConnector(config(sock), KVConnectorRole.SCHEDULER)
    try:
        worker.bind_connector_metadata(metadata(tokens, True))
        worker.save_kv_layer("layer.0", original[0], None)
        try:
            worker.wait_for_save()
        except ValueError as error:
            assert "incomplete" in str(error)
        else:
            raise AssertionError("incomplete manifest was published")
        assert scheduler.get_num_new_matched_tokens(NS(prompt_token_ids=tokens, mm_hashes=[]), 0) == (0, False)
        worker.save_kv_layer("layer.0", original[0], None)
        worker.save_kv_layer("layer.1", original[1], None)
        worker.wait_for_save()
        assert scheduler.get_num_new_matched_tokens(NS(prompt_token_ids=tokens, mm_hashes=[]), 0) == (32, False)
        different = list(tokens)
        different[4] += 1
        assert scheduler.get_num_new_matched_tokens(NS(prompt_token_ids=different, mm_hashes=[]), 0) == (0, False)
    finally:
        worker.shutdown()
        scheduler.shutdown()
        stop_daemon(proc, log, sock)

    sock = runtime_dir / "reader.sock"
    proc, log = start_daemon(binary, root, sock, result_dir / "daemon-reader.log")
    worker = KVCohortConnector(config(sock, prefetch_workers=2), KVConnectorRole.WORKER)
    scheduler = KVCohortConnector(config(sock), KVConnectorRole.SCHEDULER)
    other_model = KVCohortConnector(config(sock, "different-revision"), KVConnectorRole.SCHEDULER)
    bypass = KVCohortConnector(config(sock, min_restore_tokens=64), KVConnectorRole.SCHEDULER)
    try:
        assert scheduler.get_num_new_matched_tokens(NS(prompt_token_ids=tokens, mm_hashes=[]), 0) == (32, False)
        assert other_model.get_num_new_matched_tokens(NS(prompt_token_ids=tokens, mm_hashes=[]), 0) == (0, False)
        assert bypass.get_num_new_matched_tokens(NS(prompt_token_ids=tokens, mm_hashes=[]), 0) == (0, False)
        assert bypass._found_match_for_request(NS(prompt_token_ids=tokens, mm_hashes=[]))
        destination = [torch.zeros_like(original[0]), torch.zeros_like(original[1])]
        context = NS(attn_metadata={}, virtual_engine=0,
                     no_compile_layers={"layer.0": NS(kv_cache=[destination[0]]),
                                        "layer.1": NS(kv_cache=[destination[1]])})
        worker.bind_connector_metadata(metadata(tokens, False))
        worker.start_load_kv(context)
        for before, after in zip(original, destination):
            assert torch.equal(before[:, :2], after[:, :2])
            assert torch.count_nonzero(after[:, 2]) == 0
        prefix = worker._prefix_hash(tokens[:32], [])
        manifest = worker._manifest(prefix, 32)
        corrupted_info = dict(manifest["layers"]["layer.0"])
        corrupted_info["sha256"] = "0" * 64
        try:
            worker._read_layer(worker._client(), prefix, "layer.0", corrupted_info)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("a corrupt tensor checksum was accepted")
    finally:
        worker.shutdown()
        scheduler.shutdown()
        other_model.shutdown()
        bypass.shutdown()
        stop_daemon(proc, log, sock)

    (result_dir / "passed.txt").write_text("CPU tensor round-trip and checks passed\n")
    print("CPU connector test passed: complete publication, identity, restart, tensor restore and checksum")


if __name__ == "__main__":
    main()
