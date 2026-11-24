"""Run one isolated vLLM writer or reader for the end-to-end experiment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig


MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def request_result(llm, prompt, max_tokens):
    start = time.monotonic()
    result = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=max_tokens,
                                                   logprobs=1), use_tqdm=False)[0]
    elapsed = time.monotonic() - start
    completion = result.outputs[0]
    chosen_logprobs = []
    for token, candidates in zip(completion.token_ids, completion.logprobs or []):
        entry = candidates.get(token)
        chosen_logprobs.append(None if entry is None else entry.logprob)
    return {"prompt_tokens": len(result.prompt_token_ids),
            "cached_tokens": result.num_cached_tokens,
            "output_token_ids": list(completion.token_ids),
            "chosen_logprobs": chosen_logprobs,
            "text": completion.text,
            "generate_seconds": round(elapsed, 6)}


def main():
    experiment = Path(sys.argv[1]).resolve()
    socket_path = sys.argv[2]
    run_dir = Path(sys.argv[3]).resolve()
    mode = sys.argv[4]
    if mode not in ("writer", "reader"):
        raise ValueError("mode must be writer or reader")
    prefix = ("KV cache stores attention keys and values for already processed tokens. "
              "Matching prefixes may reuse existing values. ") * 32
    prompt = prefix + "\nQuestion: What does a KV cache store?\nAnswer:"
    different_prompt = prompt.replace("attention keys and values",
                                      "attention keys and vectors", 1)
    config = KVTransferConfig(
        kv_connector="KVCohortConnector",
        kv_connector_module_path="kvcohort_vllm.connector",
        kv_role="kv_both",
        kv_connector_extra_config={"socket_path": socket_path,
                                   "model_revision": MODEL_REVISION,
                                   "read_only": mode == "reader"},
    )
    llm = LLM(model=str(experiment / "model"), dtype="float16",
              tensor_parallel_size=1, gpu_memory_utilization=0.5,
              max_model_len=2048, max_num_seqs=1, max_num_batched_tokens=2048,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              enforce_eager=True, disable_log_stats=True,
              kv_transfer_config=config)
    used_mib = int(subprocess.check_output(
        ["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip())
    result = {"mode": mode, "gpu_memory_mib": used_mib,
              "same_prompt": request_result(llm, prompt, 16)}
    if mode == "reader":
        result["different_prompt"] = request_result(llm, different_prompt, 1)
    result_path = run_dir / f"{mode}-result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
