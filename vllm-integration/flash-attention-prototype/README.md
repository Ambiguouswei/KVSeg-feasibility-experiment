# Experimental kvcohort / vLLM integration

This directory contains a proof of concept for vLLM 0.10.2 V1 and the existing
kvcohort daemon. It is not a production cache connector or a benchmark of a
local SSD.

## What it does

- Runs a single Qwen2.5-3B-Instruct FP16 text request on one GPU, with vLLM's
  built-in prefix cache disabled.
- Saves attention KV tensors through the daemon protocol in objects no larger
  than 768 KiB. A manifest is published only after all model layers are saved.
- Includes the model revision, model configuration, representation and engine
  versions in the namespace. An exact, block-aligned prompt prefix determines
  the cache key. A different prefix or model revision is a miss.
- Checks each tensor's byte length and SHA-256 before restoring it into vLLM's
  GPU KV cache. A missing or corrupt published object fails closed; automatic
  recomputation after corruption is not implemented.
- Optionally uses two bounded reader threads (`prefetch_workers=2`). A separate,
  opt-in `min_restore_tokens` threshold skips restoring short prefixes.
  The threshold is a fixed experiment setting, not a learned cost model.

The current connector matches a whole aligned prompt prefix. It does not yet
find the longest reusable prefix among multiple related prompts. Storage and
GPU transfers are synchronous. It requires `VLLM_ATTENTION_BACKEND=FLASH_ATTN`,
one GPU, `max_num_seqs=1`, an FP16 model and the default FP16 KV format; LoRA,
multimodal requests, multiple GPUs, partial prefill and vLLM's own prefix
cache are outside this prototype's supported scope. Keep the prompt within a
single scheduled prefill and the daemon's configured capacity. The current
daemon defaults to a 64 MiB conservative capacity with reserved headroom.

## Files and verification

`kvcohort_vllm/client.py` implements the daemon wire protocol;
`kvcohort_vllm/connector.py` implements the experimental V1 connector. The
connector uses upstream vLLM's `SharedStorageConnector` scheduler metadata
implementation, available under vLLM's Apache-2.0 license.

`test_client.py` checks daemon round trips, namespace isolation, conflicting
writes and recovery after a daemon restart. `test_connector.py` uses CPU
tensors and a real private daemon to verify complete publication, exact match,
model isolation, two-reader restore and the short-prefix policy. Each test
creates a new directory and expects a compiled `kvd` in `<experiment>/bin`.

`run_end_to_end.py` starts its own daemon, performs a write with one vLLM
process, restarts the daemon and restores the same prompt in a second process.
It checks storage counters, output token equality and a different-prefix miss.
`run_benchmark.py` compares warmed one-token responses with no external cache,
sequential external restore, two-reader restore, and a short-prefix bypass.
These scripts are specific to the experiment's physical GPU 2 and model path;
they stop only the daemon and processes they start.

Use a new private socket path: the current `kvd` unlinks its specified socket
path at startup. The caller must provide the exact verified model revision;
the connector cannot independently prove that a supplied revision matches
the model files. Failed/incomplete saves may leave orphan objects. Neither
background cleanup nor multi-tenant authentication is implemented here.
