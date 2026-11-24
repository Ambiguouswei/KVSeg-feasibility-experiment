"""A deliberately narrow vLLM 0.10.2 V1 connector for kvcohort.

This proof of concept stores one exact prompt prefix, not arbitrary longest
prefixes. It is synchronous and restricted to one text-only request on one
GPU. The scheduler/worker metadata handling inherits the upstream vLLM
SharedStorageConnector (Apache-2.0), while persistence below uses kvcohort.
"""

import hashlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import safetensors.torch
import torch
import vllm
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    SharedStorageConnector,
    SharedStorageConnectorMetadata,
)

from .client import Client


FORMAT = "kvcohort-vllm-v1-exact-prompt-1"
CHUNK_BYTES = 768 * 1024


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _hash(value):
    return hashlib.sha256(value).hexdigest()


class KVCohortConnector(SharedStorageConnector):
    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._socket_path = extra.get("socket_path")
        self._read_only = bool(extra.get("read_only", False))
        self._prefetch_workers = int(extra.get("prefetch_workers", 1))
        self._min_restore_tokens = int(extra.get("min_restore_tokens", 0))
        revision = extra.get("model_revision")
        if not self._socket_path or not revision:
            raise ValueError("socket_path and exact model_revision are required")
        if self._prefetch_workers not in (1, 2) or self._min_restore_tokens < 0:
            raise ValueError("prefetch_workers must be 1 or 2, and min_restore_tokens nonnegative")
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("this prototype requires a single GPU")
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("this prototype requires max_num_seqs=1")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("disable vLLM's local prefix cache for this prototype")
        if vllm_config.lora_config is not None:
            raise ValueError("LoRA is not supported")
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError("this prototype requires FP16 model weights and KV")
        if vllm_config.cache_config.cache_dtype != "auto":
            raise ValueError("this prototype requires the default FP16 KV cache")
        if os.environ.get("VLLM_ATTENTION_BACKEND") != "FLASH_ATTN":
            raise ValueError("this prototype requires VLLM_ATTENTION_BACKEND=FLASH_ATTN")

        hf_config = vllm_config.model_config.hf_config
        self._layer_count = int(hf_config.num_hidden_layers)
        identity = {
            "format": FORMAT,
            "model_revision": revision,
            "model_config": hf_config.to_dict(),
            "model_dtype": str(vllm_config.model_config.dtype),
            "cache_dtype": vllm_config.cache_config.cache_dtype,
            "block_size": self._block_size,
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "attention_layout": "v1-flash-attention-paged-kv",
        }
        self._space = _hash(_canonical(identity))
        self._client_instance = None
        self._pending = {}

    def _client(self):
        if self._client_instance is None:
            self._client_instance = Client(self._socket_path)
        return self._client_instance

    def shutdown(self):
        if self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None

    def _aligned(self, token_count):
        return max(0, ((token_count - 1) // self._block_size) * self._block_size)

    @staticmethod
    def _prefix_hash(token_ids, mm_hashes):
        if mm_hashes:
            raise ValueError("multimodal requests are not supported")
        tokens = [int(token) for token in token_ids]
        encoded = struct.pack("<I", len(tokens))
        if tokens:
            encoded += struct.pack(f"<{len(tokens)}q", *tokens)
        return _hash(b"kvcohort-vllm-prefix-v1\0" + encoded)

    @staticmethod
    def _manifest_key(prefix_hash):
        return _hash(f"manifest:{prefix_hash}".encode())

    @staticmethod
    def _chunk_key(prefix_hash, layer_name, index):
        return _hash(_canonical(["layer", prefix_hash, layer_name, index]))

    def _manifest(self, prefix_hash, expected_tokens):
        raw = self._client().get(self._space, self._manifest_key(prefix_hash))
        if raw is None:
            return None
        manifest = json.loads(raw)
        if (manifest.get("format") != FORMAT
                or manifest.get("space") != self._space
                or manifest.get("prefix_hash") != prefix_hash
                or manifest.get("num_tokens") != expected_tokens
                or not isinstance(manifest.get("layers"), dict)
                or len(manifest["layers"]) != self._layer_count):
            raise ValueError("invalid or incomplete KV manifest")
        return manifest

    def _found_match_for_request(self, request):
        token_ids = request.prompt_token_ids
        count = self._aligned(len(token_ids))
        if count == 0:
            return False
        prefix = self._prefix_hash(token_ids[:count], request.mm_hashes)
        return self._manifest(prefix, count) is not None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if num_computed_tokens:
            raise ValueError("local prefix hits are outside the prototype scope")
        if self._aligned(len(request.prompt_token_ids)) < self._min_restore_tokens:
            return 0, False
        if not self._found_match_for_request(request):
            return 0, False
        return self._aligned(len(request.prompt_token_ids)), False

    def _read_layer(self, client, prefix, layer_name, info):
        chunks = []
        for index, expected in enumerate(info["chunks"]):
            key = self._chunk_key(prefix, layer_name, index)
            chunk = client.get(self._space, key)
            if chunk is None or len(chunk) != expected:
                raise ValueError("a published KV chunk is missing or truncated")
            chunks.append(chunk)
        data = b"".join(chunks)
        if len(data) != info["bytes"] or _hash(data) != info["sha256"]:
            raise ValueError("KV tensor checksum mismatch")
        return data

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SharedStorageConnectorMetadata):
            raise TypeError("unexpected connector metadata")
        requests = [request for request in metadata.requests if not request.is_store]
        if not requests:
            return
        if forward_context.attn_metadata is None or len(requests) != 1:
            raise ValueError("one active attention request is required")

        request = requests[0]
        if len(request.token_ids) != len(request.slot_mapping):
            raise ValueError("partial prefill is not supported")
        prefix = self._prefix_hash(request.token_ids.tolist(), request.mm_hashes)
        manifest = self._manifest(prefix, len(request.token_ids))
        if manifest is None:
            raise ValueError("published KV manifest disappeared")

        layers = {}
        for name, layer in forward_context.no_compile_layers.items():
            caches = getattr(layer, "kv_cache", None)
            if caches is not None:
                layers[name] = caches[forward_context.virtual_engine]
        if set(layers) != set(manifest["layers"]):
            raise ValueError("model attention layers do not match the KV manifest")

        prefetched = None
        if self._prefetch_workers == 2:
            names = list(layers)

            def read_group(group):
                client = Client(self._socket_path)
                try:
                    return {name: self._read_layer(client, prefix, name,
                                                   manifest["layers"][name])
                            for name in group}
                finally:
                    client.close()

            with ThreadPoolExecutor(max_workers=2) as pool:
                groups = list(pool.map(read_group, (names[::2], names[1::2])))
            prefetched = {name: data for group in groups for name, data in group.items()}

        for layer_name, destination in layers.items():
            info = manifest["layers"][layer_name]
            data = (prefetched[layer_name] if prefetched is not None else
                    self._read_layer(self._client(), prefix, layer_name, info))
            source = safetensors.torch.load(data)["kv_cache"]
            if destination.ndim != 5 or destination.shape[0] != 2 or not destination.is_contiguous():
                raise ValueError("unsupported attention KV layout")
            flat = destination.reshape(2, destination.shape[1] * destination.shape[2], -1)
            slots = request.slot_mapping.to(device=destination.device)
            if source.dtype != destination.dtype or tuple(source.shape) != (
                    2, len(slots), flat.shape[-1]):
                raise ValueError("KV tensor shape or dtype mismatch")
            flat[:, slots, :] = source.to(device=destination.device)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if self._read_only:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SharedStorageConnectorMetadata):
            raise TypeError("unexpected connector metadata")
        for request in metadata.requests:
            if not request.is_store or len(request.token_ids) == 0:
                continue
            if len(request.token_ids) != len(request.slot_mapping):
                raise ValueError("partial prefill is not supported")
            if kv_layer.ndim != 5 or kv_layer.shape[0] != 2:
                raise ValueError("unsupported attention KV layout")
            prefix = self._prefix_hash(request.token_ids.tolist(), request.mm_hashes)
            flat = kv_layer.reshape(2, kv_layer.shape[1] * kv_layer.shape[2], -1)
            slots = request.slot_mapping.to(device=kv_layer.device)
            tensor = flat[:, slots, :].detach().cpu().contiguous()
            data = safetensors.torch.save({"kv_cache": tensor})
            chunk_sizes = []
            for index, offset in enumerate(range(0, len(data), CHUNK_BYTES)):
                chunk = data[offset:offset + CHUNK_BYTES]
                self._client().put(self._space, self._chunk_key(prefix, layer_name, index),
                                   chunk, prefix_group=prefix)
                chunk_sizes.append(len(chunk))
            layers = self._pending.setdefault(prefix, {})
            if layer_name in layers:
                raise ValueError("duplicate layer in one save operation")
            layers[layer_name] = {"bytes": len(data), "sha256": _hash(data),
                                  "chunks": chunk_sizes}

    def wait_for_save(self):
        if self._read_only:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SharedStorageConnectorMetadata):
            raise TypeError("unexpected connector metadata")
        for request in metadata.requests:
            if not request.is_store or len(request.token_ids) == 0:
                continue
            prefix = self._prefix_hash(request.token_ids.tolist(), request.mm_hashes)
            layers = self._pending.pop(prefix, {})
            if len(layers) != self._layer_count:
                raise ValueError("incomplete KV save; manifest was not published")
            manifest = {"format": FORMAT, "space": self._space,
                        "prefix_hash": prefix, "num_tokens": len(request.token_ids),
                        "layers": layers}
            self._client().put(self._space, self._manifest_key(prefix),
                               _canonical(manifest), prefix_group=prefix)
