"""Bounded multi-request connector for controlled shared-prefix experiments.

The candidate prefix lengths are explicitly configured. This is an
experimental checkpoint-based connector, not a general longest-prefix index.
All transfers are synchronous and a prefill that is split before the chosen
checkpoint is not saved.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import safetensors.torch
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    ReqMeta,
    SharedStorageConnectorMetadata,
)

from .client import Client
from .connector import KVCohortConnector


class KVCohortBenchmarkConnector(KVCohortConnector):
    _allow_multiple_requests = True
    _allow_local_prefix_cache = True

    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        lengths = extra.get("prefix_checkpoints")
        if not isinstance(lengths, list) or not lengths:
            raise ValueError("non-empty prefix_checkpoints list is required")
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0
               or n % self._block_size for n in lengths):
            raise ValueError("prefix checkpoints must be positive multiples of the KV block size")
        self._checkpoints = sorted(set(lengths), reverse=True)
        if vllm_config.scheduler_config.max_num_seqs > 16:
            raise ValueError("the experiment connector supports at most sixteen requests")
        self._load_counts = {}

    def _matching_checkpoint(self, prompt_token_ids, mm_hashes):
        for count in self._checkpoints:
            if count >= len(prompt_token_ids) or count < self._min_restore_tokens:
                continue
            prefix = self._prefix_hash(prompt_token_ids[:count], mm_hashes)
            if self._manifest(prefix, count) is not None:
                return count
        return 0

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # A partially resident GPU prefix is used as-is and the remainder is
        # recomputed. We never overwrite a resident shared GPU block from disk.
        if num_computed_tokens:
            return 0, False
        return self._matching_checkpoint(request.prompt_token_ids, request.mm_hashes), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            self._requests_need_load[request.request_id] = request
            self._load_counts[request.request_id] = num_external_tokens

    def _make_request(self, token_ids, block_ids, count, is_store, mm_hashes):
        if count % self._block_size or len(block_ids) * self._block_size < count:
            raise ValueError("insufficient KV blocks for requested checkpoint")
        blocks = torch.tensor(block_ids[:count // self._block_size], dtype=torch.long)
        offsets = torch.arange(self._block_size, dtype=torch.long)
        slots = (blocks[:, None] * self._block_size + offsets[None, :]).flatten()
        return ReqMeta(token_ids=torch.tensor(token_ids[:count], dtype=torch.long),
                       slot_mapping=slots, is_store=is_store, mm_hashes=mm_hashes)

    def build_connector_meta(self, scheduler_output):
        meta = SharedStorageConnectorMetadata()
        planned_saves = set()
        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id in self._requests_need_load:
                count = self._load_counts.pop(request.req_id)
                meta.requests.append(self._make_request(
                    request.prompt_token_ids, request.block_ids[0], count, False,
                    request.mm_hashes))
                del self._requests_need_load[request.req_id]
                continue
            if self._read_only or request.num_computed_tokens:
                continue
            candidates = [n for n in self._checkpoints if n < len(request.prompt_token_ids)]
            if not candidates:
                continue
            count = candidates[0]
            if scheduler_output.num_scheduled_tokens[request.req_id] < count:
                # A later prefill chunk may be correct, but we do not capture it.
                continue
            prefix = self._prefix_hash(request.prompt_token_ids[:count], request.mm_hashes)
            if prefix in planned_saves or self._manifest(prefix, count) is not None:
                continue
            planned_saves.add(prefix)
            meta.requests.append(self._make_request(
                request.prompt_token_ids, request.block_ids[0], count, True,
                request.mm_hashes))
        if self._requests_need_load:
            raise ValueError("resumed or preempted external loads are not supported")
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SharedStorageConnectorMetadata):
            raise TypeError("unexpected connector metadata")
        requests = [request for request in metadata.requests if not request.is_store]
        if not requests:
            return
        if forward_context.attn_metadata is None:
            raise ValueError("attention metadata is required")

        layers = {}
        for name, layer in forward_context.no_compile_layers.items():
            caches = getattr(layer, "kv_cache", None)
            if caches is not None:
                layers[name] = caches[forward_context.virtual_engine]

        groups = defaultdict(list)
        for request in requests:
            if len(request.token_ids) != len(request.slot_mapping):
                raise ValueError("invalid checkpoint slot mapping")
            prefix = self._prefix_hash(request.token_ids.tolist(), request.mm_hashes)
            groups[prefix].append(request)

        for prefix, group in groups.items():
            manifest = self._manifest(prefix, len(group[0].token_ids))
            if manifest is None or set(layers) != set(manifest["layers"]):
                raise ValueError("missing KV manifest or attention layer mismatch")

            prefetched = None
            if self._prefetch_workers == 2:
                names = list(layers)

                def read_group(names_to_read):
                    client = self._new_client()
                    try:
                        return {name: self._read_layer(client, prefix, name,
                                                       manifest["layers"][name])
                                for name in names_to_read}
                    finally:
                        client.close()

                with ThreadPoolExecutor(max_workers=2) as pool:
                    fetched = list(pool.map(read_group, (names[::2], names[1::2])))
                prefetched = {name: data for part in fetched for name, data in part.items()}

            for layer_name, destination in layers.items():
                info = manifest["layers"][layer_name]
                data = (prefetched[layer_name] if prefetched is not None else
                        self._read_layer(self._client(), prefix, layer_name, info))
                source = safetensors.torch.load(data)["kv_cache"]
                if destination.ndim != 5 or destination.shape[0] != 2 or not destination.is_contiguous():
                    raise ValueError("unsupported attention KV layout")
                flat = destination.reshape(2, destination.shape[1] * destination.shape[2], -1)
                if source.dtype != destination.dtype or tuple(source.shape) != (
                        2, len(group[0].slot_mapping), flat.shape[-1]):
                    raise ValueError("KV tensor shape or dtype mismatch")
                device_source = source.to(device=destination.device)
                for request in group:
                    if len(request.slot_mapping) != source.shape[1]:
                        raise ValueError("inconsistent shared-prefix length")
                    slots = request.slot_mapping.to(device=destination.device)
                    flat[:, slots, :] = device_source
