"""Measure warmed one-token completions with and without external KV."""

import json
from pathlib import Path
import statistics
import sys

from vllm import LLM
from vllm.config.kv_transfer import KVTransferConfig

from run_inference import MODEL_REVISION, request_result


def main():
    experiment = Path(sys.argv[1]).resolve()
    socket_path = sys.argv[2]
    result_dir = Path(sys.argv[3]).resolve()
    mode = sys.argv[4]
    if mode not in ("baseline", "restore", "prefetch", "bypass"):
        raise ValueError("invalid benchmark mode")
    prefix = ("KV cache stores attention keys and values for already processed tokens. "
              "Matching prefixes may reuse existing values. ") * 32
    prompt = prefix + "\nQuestion: What does a KV cache store?\nAnswer:"
    warmup = prompt.replace("attention keys and values", "attention keys and vectors", 1)
    config = None
    if mode != "baseline":
        config = KVTransferConfig(
            kv_connector="KVCohortConnector", kv_role="kv_both",
            kv_connector_module_path="kvcohort_vllm.connector",
            kv_connector_extra_config={"socket_path": socket_path,
                                       "model_revision": MODEL_REVISION,
                                       "read_only": True,
                                       "prefetch_workers": 2 if mode == "prefetch" else 1,
                                       "min_restore_tokens": 1024 if mode == "bypass" else 0})
    llm = LLM(model=str(experiment / "model"), dtype="float16", tensor_parallel_size=1,
              gpu_memory_utilization=0.5, max_model_len=2048, max_num_seqs=1,
              max_num_batched_tokens=2048, enable_prefix_caching=False,
              enable_chunked_prefill=False, enforce_eager=True, disable_log_stats=True,
              kv_transfer_config=config)
    warm = request_result(llm, warmup, 1)
    samples = [request_result(llm, prompt, 1) for _ in range(7)]
    expected_hit = 608 if mode in ("restore", "prefetch") else 0
    if any(sample["cached_tokens"] != expected_hit for sample in samples):
        raise AssertionError(f"unexpected cache hits in {mode}: {samples}")
    times = [sample["generate_seconds"] for sample in samples]
    result = {"mode": mode, "warmup": warm, "samples": samples,
              "median_one_token_response_seconds": statistics.median(times),
              "min_one_token_response_seconds": min(times),
              "max_one_token_response_seconds": max(times),
              "timing_scope": "LLM.generate wall time, one output token; not streaming TTFT"}
    (result_dir / f"{mode}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
