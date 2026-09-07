import contextlib
import math
import os
import typing
from collections import deque
from zlib import adler32

import numpy as np
import torch
import torch_npu

with contextlib.suppress(ImportError):
    # we should remove this after memfabric.offload is merged to master and available in ci machine.
    from memfabric_hybrid import offload  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.ascend_config import SparseKVOffloadConfig, get_ascend_config
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.nano_cache import (
    KV_COMPONENTS,
    TAIL_BLOCKS,
    CopyDescriptors,
    prepare_resident_gather,
    prepare_tail_copy,
)

# Main BF16 cache:
# [k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v].
# Sparse LI C8 indexer caches are separate and
# remain device-resident, so they are not registered with this manager.
OFFLOAD_KV_CACHE_TUPLE_LEN = 6
OFFLOAD_K_CACHE_NPU_INDEX = 0
OFFLOAD_V_CACHE_NPU_INDEX = 1
OFFLOAD_K_CACHE_CPU_INDEX = 2
OFFLOAD_V_CACHE_CPU_INDEX = 3
OFFLOAD_TOPK_BUFFER_K_INDEX = 4
OFFLOAD_TOPK_BUFFER_V_INDEX = 5


_SUBSCRIBED_COMPUTE_STREAMS: set[object] = set()


def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


def get_host_device_memory_usage_ratio(kv_cache_specs: dict[str, KVCacheSpec]) -> float:
    page_size_bytes_host = 0
    page_size_bytes_device = 0
    for kv_cache_spec in kv_cache_specs.values():
        assert isinstance(kv_cache_spec, KVCacheSpec)
        if getattr(kv_cache_spec, "store_on_host", False):
            page_size_bytes_host += kv_cache_spec.page_size_bytes
        else:
            page_size_bytes_device += kv_cache_spec.page_size_bytes

    assert page_size_bytes_device > 0, "Case of no device kv cache is not considered."
    return page_size_bytes_host / page_size_bytes_device


def allocate_kv_offload_topk_buffer_pair(
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_width = 1
    if vllm_config.speculative_config is not None:
        decode_width += vllm_config.speculative_config.num_speculative_tokens

    topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
    num_kv_heads = 1  # sparse kv offload only support sfa(mla) now.
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    max_num_topk_rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.scheduler_config.max_num_seqs * decode_width,
    )
    topk_buffer_k_shape = [max_num_topk_rows, topk_buffer_size, num_kv_heads, k_dim]
    topk_buffer_v_shape = [max_num_topk_rows, topk_buffer_size, num_kv_heads, v_dim]
    if sparse_kv_offload_config.fused_op_type == "nano":
        block_size = vllm_config.cache_config.block_size
        tail_block_num = 2
        topk_buffer_k_shape = [max_num_topk_rows, topk_buffer_size + tail_block_num * block_size, num_kv_heads, k_dim]
        topk_buffer_v_shape = [max_num_topk_rows, topk_buffer_size + tail_block_num * block_size, num_kv_heads, v_dim]
    topk_buffer_k_size_bytes = math.prod(topk_buffer_k_shape) * torch.bfloat16.itemsize
    topk_buffer_v_size_bytes = math.prod(topk_buffer_v_shape) * torch.bfloat16.itemsize
    # NOTE make sure to allocate k+v together and split them after allocate.
    # Refer to the comment in empty_aligned_int8_cpu_tensors for reason.
    topk_buffer_raw = torch.empty([topk_buffer_k_size_bytes + topk_buffer_v_size_bytes], dtype=torch.int8, device="npu")
    topk_buffer_k = topk_buffer_raw[:topk_buffer_k_size_bytes].view(torch.bfloat16).view(topk_buffer_k_shape)
    topk_buffer_v = (
        topk_buffer_raw[topk_buffer_k_size_bytes : topk_buffer_k_size_bytes + topk_buffer_v_size_bytes]
        .view(torch.bfloat16)
        .view(topk_buffer_v_shape)
    )
    if getattr(sparse_kv_offload_config, "generalized_mtp", False):
        # Graph padding owns rows beyond the live request pool. Its LIM state
        # is non-offload, so no first-fill or tail H2D will initialize these
        # buffers. Initialize both hot KV and tails once, outside replay.
        first_padding_row = vllm_config.scheduler_config.max_num_seqs
        topk_buffer_k[first_padding_row:].zero_()
        topk_buffer_v[first_padding_row:].zero_()
    return (topk_buffer_k, topk_buffer_v)


def allocate_kv_offload_topk_profile_buffers(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    num_offload_layers = sum(bool(getattr(spec, "store_on_host", False)) for spec in kv_cache_spec.values())
    if num_offload_layers == 0:
        raise RuntimeError("Sparse KV offload profile did not find any host-resident SFA KV cache layers.")

    buffers = [
        allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config) for _ in range(num_offload_layers)
    ]
    buffer_bytes = sum(tensor.numel() * tensor.element_size() for buffer_pair in buffers for tensor in buffer_pair)
    logger.info_once(
        "Sparse KV offload reserved %.2f GiB of KV offload topk buffers across %d layers for profile run.",
        buffer_bytes / (1 << 30),
        num_offload_layers,
    )
    return buffers


_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024


def empty_aligned_int8_cpu_tensors(
    sizes: list[int],
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> list[torch.Tensor]:
    """
    Allocate multiple int8 tensors with specified sizes,
    each aligned to specified alignment,
    and minimize the gap between each tensor's data_ptr.
    This is used for GLM-5.2 indexer reuse optimize:
    make sure that delta_k_cache_addrs and delta_v_cache_addrs
    between each two layers are the same,
    so we only need to add one delta_addr to the addr tensor of sparse_copy
    without need to mask k and v separately.
    Same reason that we allocate topk_buffer_k & topk_buffer_v together.
    """
    chunk_nums = [cdiv(size, alignment) for size in sizes]
    total_chunk_num = 1 + sum(chunk_nums)
    raw_tensor = offload.empty([total_chunk_num * alignment], dtype=torch.int8, pin_memory=True)
    base_addr = raw_tensor.data_ptr()
    if base_addr % alignment:
        base_addr = (base_addr // alignment + 1) * alignment
    base_offset = base_addr - raw_tensor.data_ptr()
    allocate_tensors = []
    for size, chunk_num in zip(sizes, chunk_nums):
        allocate_tensors.append(raw_tensor[base_offset : base_offset + size])
        base_offset += chunk_num * alignment
    return allocate_tensors


def allocate_kv_cache_tensors_for_sparse_kv_offload(
    k_tensor_size: int,
    v_tensor_size: int,
    alignment: int,
    tp_rank: int,
    keep_device_kv_cache: bool,
    npu_kv_cache_allocate_func: typing.Callable,
):
    if tp_rank == 0:
        [k_tensor_cpu, v_tensor_cpu] = empty_aligned_int8_cpu_tensors(
            [k_tensor_size, v_tensor_size],
            alignment,
        )
    else:
        k_tensor_cpu = None
        v_tensor_cpu = None

    if keep_device_kv_cache:
        k_tensor = npu_kv_cache_allocate_func(
            k_tensor_size,
            alignment,
        )
        v_tensor = npu_kv_cache_allocate_func(
            v_tensor_size,
            alignment,
        )
    else:
        k_tensor = None
        v_tensor = None

    return (k_tensor, v_tensor, k_tensor_cpu, v_tensor_cpu, k_tensor_size + v_tensor_size)


def reshape_kv_cache_tensors_for_sparse_kv_offload(
    raw_cache_tensors: tuple,
    current_kv_cache_spec: AttentionSpec,
    attn_backend: type[AttentionBackend],
    tp_rank: int,
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    raw_k_tensor, raw_v_tensor, raw_k_tensor_cpu, raw_v_tensor_cpu, sum_page_size_bytes = raw_cache_tensors
    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
    kv_cache_shape = attn_backend.get_kv_cache_shape(
        num_blocks,
        current_kv_cache_spec.block_size,
        current_kv_cache_spec.num_kv_heads,
        current_kv_cache_spec.head_size,
    )
    mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    k_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        k_dim,
    )
    v_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        v_dim,
    )
    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype

    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape) if raw_k_tensor is not None else None
    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape) if raw_v_tensor is not None else None

    if tp_rank == 0:
        k_cache_cpu = raw_k_tensor_cpu.view(k_cache_dtype).view(k_shape)
        v_cache_cpu = raw_v_tensor_cpu.view(v_cache_dtype).view(v_shape)
    else:
        k_cache_cpu = None
        v_cache_cpu = None

    topk_buffer_k, topk_buffer_v = allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config)
    return (k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v)


def prepare_sparse_kv_offload_mtp_dummy_metadata(
    num_tokens: int,
    num_reqs: int,
    query_start_loc_cpu: torch.Tensor,
    req_ids_buffer: CpuGpuBuffer,
    token_to_req_buffer: CpuGpuBuffer,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not get_ascend_config().sparse_kv_offload_config.enabled:
        return None, None
    if req_ids_buffer is None or token_to_req_buffer is None:
        raise RuntimeError("Sparse KV offload metadata buffers are not initialized")

    query_lens = np.diff(query_start_loc_cpu[: num_reqs + 1].numpy()).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        token_to_req = np.pad(token_to_req, (0, num_tokens - token_to_req.shape[0]))

    req_ids_buffer.np[:num_reqs] = np.arange(1, num_reqs + 1, dtype=np.int64)
    req_ids_buffer.copy_to_gpu(num_reqs)
    token_to_req_buffer.np[:num_tokens] = token_to_req[:num_tokens]
    token_to_req_buffer.copy_to_gpu(num_tokens)
    return (
        req_ids_buffer.gpu[:num_reqs],
        token_to_req_buffer.gpu[:num_tokens],
    )


class TopkBufferSlotManager:
    """
    A manager to allocate one request to a specific topk_buffer slot,
    each request will stay in this allocated slot until it's finished.
    """

    def __init__(
        self,
        topk_buffer_rows: int,
    ):
        self.slots_pool = deque(range(topk_buffer_rows))
        self.req2slot: dict[str, int] = {}

    def _allocate_slot_for_req(self, req_id):
        if req_id in self.req2slot:
            return
        if len(self.slots_pool) <= 0:
            raise ValueError(f"No enough topk buffer slot to allocate for new req {req_id}.")
        new_slot = self.slots_pool.popleft()
        self.req2slot[req_id] = new_slot

    def _free_slot_of_req(self, req_id: str):
        if req_id not in self.req2slot:
            return
        old_slot = self.req2slot.pop(req_id)
        self.slots_pool.append(old_slot)

    def allocate_topk_buffer_slots(self, req_ids: list[str]) -> tuple[list[int], list[bool]]:
        """
        First free unused slots (for request which is finished),
        then allocate slots for new requests.
        Return: slot_id of each request, and whether this request is allocated to a new slot (first decode)
        """
        # first free unused slots
        req_ids_set = set(req_ids)
        to_free_req_ids = [old_req_id for old_req_id in self.req2slot if old_req_id not in req_ids_set]
        for to_free_req_id in to_free_req_ids:
            self._free_slot_of_req(to_free_req_id)
        # then allocate slots for new req
        to_allocate_req_ids = [req_id for req_id in req_ids if req_id not in self.req2slot]
        for to_allocate_req_id in to_allocate_req_ids:
            self._allocate_slot_for_req(to_allocate_req_id)
        # finally return the cached or new slot_ids of each request, and corresponding first allocate flags
        slot_ids = []
        first_allocate = []
        for req_id in req_ids:
            slot_ids.append(self.req2slot[req_id])
            first_allocate.append(req_id in to_allocate_req_ids)
        return slot_ids, first_allocate


class SparseKVOffloadManager:
    """
    A manager responsible to the Sparse KV cache Offload.
    It enlarge the available memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        sparse_kv_offload_config: SparseKVOffloadConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.sparse_kv_offload_config = sparse_kv_offload_config

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)
        self.topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
        self.topk = sparse_kv_offload_config.topk

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        max_block_num = cdiv(self.max_model_len, self.block_size)
        self.block_table_cpu = torch.zeros(
            [self.max_num_reqs, max_block_num],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.block_table_expanded_cpu = torch.empty(
            [self.max_num_topk_rows, max_block_num],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self._npu_runtime = torch_npu.npu

        self._build_cpp()

        logger.info(
            "SparseKVOffloadManager start init CPU KV pool with %s "
            "GB dram per dp group, it might be time consuming, please wait.",
            sparse_kv_offload_config.dram_size_per_dp_GB,
        )
        config = offload.OffloadConfig()
        config.device_id = torch_npu.npu.current_device()
        config.reserve_size = sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30)
        config.alloc_size = sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30) if self.tp_rank == 0 else 0
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        config.scene = offload.Scene.SHARED
        assert offload.initialize(config) == 0, "Sparse KV offload offload.initialize failed."
        self.tp_group.barrier()

        self.topk_buffer_slot_manager = TopkBufferSlotManager(self.max_num_reqs)
        self.nano_mtp_slot_generations: dict[int, int] = {}

    def _build_cpp(self):
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        npu_include_path = os.path.join(ascend_home, "include")
        npu_lib_path = os.path.join(ascend_home, "lib64")
        if not os.path.exists(npu_lib_path):
            npu_lib_path = os.path.join(ascend_home, "lib")
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        torch_npu_include = os.path.join(torch_npu_path, "include")
        torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
        os.environ["CXX"] = "clang++"
        os.environ["CC"] = "clang"
        abs_path = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(abs_path, "sparse_kv_offload.cpp")
        logger.info_once(f"Sparse KV offload build cpp utils from src: {src_path}")
        self.sparse_kv_offload_cpp = torch.utils.cpp_extension.load(
            name="sparse_kv_offload",
            sources=[src_path],
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-fopenmp",
                "-march=armv8.2-a+sve+fp16+bf16",
                "-fPIC",
                f"-I{npu_include_path}",
                f"-I{torch_npu_include}",
            ],
            extra_ldflags=[
                "-fopenmp",
                f"-L{npu_lib_path}",
                "-lascendcl",
                f"-L{torch_npu_lib_path}",
                "-ltorch_npu",
            ],
            verbose=True,
        )

    def _infer_group_block_sizes(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> int:
        assert len(kv_cache_config.kv_cache_groups) == 1, "Hybrid KV is not supported."
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        return kv_cache_spec.block_size

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.offload_layer_names = [layer_name for layer_name in kv_caches if "indexer" not in layer_name]
        if not self.offload_layer_names:
            raise ValueError("Sparse KV offload did not find SFA KV cache layers.")

        self.num_layers = len(self.offload_layer_names)
        self.layer_name_to_offload_id = {
            layer_name: layer_id for layer_id, layer_name in enumerate(self.offload_layer_names)
        }

        logger.info_once(
            "Sparse KV offload registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        self.mtp_layer_id = self.num_layers - 1 if self.num_layers != self.num_target_layers else -1
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("Sparse KV offload layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "Sparse KV offload layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def _use_nano_fused_op(self) -> bool:
        return self.sparse_kv_offload_config.fused_op_type == "nano"

    def _restore_bfloat16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = self.sparse_kv_offload_cpp.restore_bfloat16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_bfloat16_tensor returned a tensor with unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                f"restore_bfloat16_tensor returned unexpected shape: expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.bfloat16:
            raise RuntimeError(f"restore_bfloat16_tensor returned unexpected dtype: {view.dtype}")
        if not view.is_contiguous():
            raise RuntimeError("restore_bfloat16_tensor requires a contiguous view")
        return view

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        self._register_offload_layers(kv_caches)

        # register topk_buffer and cpu kv_cache
        self.topk_buffers_k: list[torch.Tensor] = []
        self.topk_buffers_v: list[torch.Tensor] = []
        self.k_caches_cpu: list[torch.Tensor] = []
        self.v_caches_cpu: list[torch.Tensor] = []
        for layer_name in self.offload_layer_names:
            cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
            tuple_len = len(cache_or_caches)
            if tuple_len not in [OFFLOAD_KV_CACHE_TUPLE_LEN]:
                raise ValueError(
                    f"Sparse KV offload layer {layer_name}: expected tuple length "
                    f"{OFFLOAD_KV_CACHE_TUPLE_LEN}, got {tuple_len}"
                )
            self.topk_buffers_k.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_K_INDEX])
            self.topk_buffers_v.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_V_INDEX])
            if self.tp_rank == 0:
                self.k_caches_cpu.append(cache_or_caches[OFFLOAD_K_CACHE_CPU_INDEX])
                self.v_caches_cpu.append(cache_or_caches[OFFLOAD_V_CACHE_CPU_INDEX])

        kv_head_num = self.topk_buffers_k[0].size(-2)
        head_dim_k = self.topk_buffers_k[0].size(-1)
        head_dim_v = self.topk_buffers_v[0].size(-1)
        dtype = self.topk_buffers_k[0].dtype
        assert kv_head_num == 1, "Sparse KV offload only support sfa(mla)"
        if dtype != torch.bfloat16:
            raise ValueError(
                "Sparse KV offload requires a BF16 main SFA cache; sparse LI "
                "C8 is supported only for the device-resident indexer cache."
            )
        self.token_size_bytes_k = kv_head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = kv_head_num * head_dim_v * dtype.itemsize
        if self.topk_buffer_size % self.block_size != 0:
            raise ValueError(
                "Sparse KV offload topk_buffer_size must be divisible by "
                f"block_size, got {self.topk_buffer_size} and {self.block_size}"
            )

        # D2H uses a separate descriptor set from the shared H2D buffers below.
        # Both prefill (colocate debug only, gated by keep_device_kv_cache)
        # and decode can produce up to max_num_tokens rows.
        d2h_descriptor_rows = self.max_num_tokens * 2
        device = self.topk_buffers_k[0].device
        self.d2h_src_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_dst_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_lengths_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int32, device=device)
        self.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
        self.d2h_token_indices_npu = torch.arange(self.max_num_tokens, dtype=torch.int64, device=device)
        self._nano_tail_copies: dict[tuple[str, int], CopyDescriptors] = {}
        self._nano_resident_copy = None
        if getattr(self.sparse_kv_offload_config, "generalized_mtp", False):
            self._nano_resident_copy = CopyDescriptors(KV_COMPONENTS * self.max_num_topk_rows * self.topk, device)
            logger.info("Generalized nano host-tail H2D enabled: two tail blocks per request, TP rank=%s", self.tp_rank)
        self._nano_copy_fence = torch.zeros((), dtype=torch.int8, device=device)

        pages_per_row = self.topk_buffer_size // self.block_size
        self.current_slots_npu = torch.empty(
            (self.max_num_topk_rows, self.topk),
            dtype=torch.int32,
            device=device,
        )
        self.resident_block_table_npu = torch.arange(
            self.max_num_topk_rows * pages_per_row,
            dtype=torch.int32,
            device=device,
        ).view(self.max_num_topk_rows, pages_per_row)
        self.resident_query_lens_npu = torch.arange(1, self.max_num_topk_rows + 1, dtype=torch.int32, device=device)
        self.resident_seq_lens_npu = torch.full(
            (self.max_num_topk_rows,),
            self.topk_buffer_size,
            dtype=torch.int32,
            device=device,
        )

        # sparse_copy related addrs and buffers
        self.addr_k_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_k]
        self.addr_v_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_v]
        self.gvas_k_bases: list[int] = []
        self.gvas_v_bases: list[int] = []
        self.cpu_block_lens: list[tuple[int, int]] = []
        use_nano = self._use_nano_fused_op()
        gvas_k_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        gvas_v_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        cpu_block_lens_tensor = torch.zeros([self.num_layers, 2], dtype=torch.int64, device="npu")
        # MLA CPU pools are always 4D: [num_blocks, block_size, num_kv_heads, head_dim].
        shape_k_tensor = torch.zeros([4], dtype=torch.int64, device="npu")
        shape_v_tensor = torch.zeros([4], dtype=torch.int64, device="npu")
        if self.tp_rank == 0:
            for layer_id in range(self.num_layers):
                k_cpu = self.k_caches_cpu[layer_id]
                v_cpu = self.v_caches_cpu[layer_id]
                gvas_k_tensor[layer_id] = k_cpu.data_ptr()
                gvas_v_tensor[layer_id] = v_cpu.data_ptr()
                cpu_block_lens_tensor[layer_id, 0] = (
                    k_cpu.numel() * k_cpu.element_size() // self.kv_cache_config.num_blocks
                )
                cpu_block_lens_tensor[layer_id, 1] = (
                    v_cpu.numel() * v_cpu.element_size() // self.kv_cache_config.num_blocks
                )
            if use_nano:
                if len(self.k_caches_cpu[0].shape) != 4:
                    raise RuntimeError(
                        "Sparse KV offload nano restore expects a 4D CPU K cache, "
                        f"got shape={tuple(self.k_caches_cpu[0].shape)}"
                    )
                if len(self.v_caches_cpu[0].shape) != 4:
                    raise RuntimeError(
                        "Sparse KV offload nano restore expects a 4D CPU V cache, "
                        f"got shape={tuple(self.v_caches_cpu[0].shape)}"
                    )
                shape_k_tensor.copy_(torch.tensor(self.k_caches_cpu[0].shape, dtype=torch.int64, device="npu"))
                shape_v_tensor.copy_(torch.tensor(self.v_caches_cpu[0].shape, dtype=torch.int64, device="npu"))
        self.tp_group.broadcast(gvas_k_tensor, src=0)
        self.tp_group.broadcast(gvas_v_tensor, src=0)
        self.tp_group.broadcast(cpu_block_lens_tensor, src=0)
        if use_nano:
            self.tp_group.broadcast(shape_k_tensor, src=0)
            self.tp_group.broadcast(shape_v_tensor, src=0)
        for layer_id in range(self.num_layers):
            self.gvas_k_bases.append(gvas_k_tensor[layer_id].item())
            self.gvas_v_bases.append(gvas_v_tensor[layer_id].item())
            self.cpu_block_lens.append(
                (
                    cpu_block_lens_tensor[layer_id, 0].item(),
                    cpu_block_lens_tensor[layer_id, 1].item(),
                )
            )

        # fused_copy_sfa reads DRAM KV from host tensors on every TP rank.
        # TP0 owns the shared host pool; other ranks wrap the broadcast GVA as
        # CPU views (same physical storage).
        if use_nano:
            if self.tp_rank != 0:
                cpu_k_shape = [int(x) for x in shape_k_tensor.tolist()]
                cpu_v_shape = [int(x) for x in shape_v_tensor.tolist()]
                if any(dim <= 0 for dim in cpu_k_shape + cpu_v_shape):
                    raise RuntimeError(
                        "Sparse KV offload nano restore received invalid CPU cache "
                        f"shapes: k={cpu_k_shape}, v={cpu_v_shape}"
                    )
                if any(ptr == 0 for ptr in self.gvas_k_bases + self.gvas_v_bases):
                    raise RuntimeError(
                        f"Sparse KV offload nano restore received a zero GVA pointer on tp_rank={self.tp_rank}"
                    )
                self.k_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_k_shape) for ptr in self.gvas_k_bases]
                self.v_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_v_shape) for ptr in self.gvas_v_bases]
                logger.info(
                    "Sparse KV offload restored shared CPU KV views on tp_rank=%s layer_count=%s k_shape=%s v_shape=%s",
                    self.tp_rank,
                    self.num_layers,
                    cpu_k_shape,
                    cpu_v_shape,
                )

        gvas_buffer_offset = 0
        gvas_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8  # 2: k+v, 8: int64
        addr_buffer_offset = gvas_buffer_offset + gvas_buffer_size_bytes
        addr_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8
        size_buffer_offset = addr_buffer_offset + addr_buffer_size_bytes
        size_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 4  # 2: k+v, 4: int32
        num_tokens_buffer_offset = size_buffer_offset + size_buffer_size_bytes
        num_tokens_buffer_size_bytes = 4  # 1 * int32
        sparse_copy_args_buffer_size_bytes = (
            gvas_buffer_size_bytes + addr_buffer_size_bytes + size_buffer_size_bytes + num_tokens_buffer_size_bytes
        )
        self.sparse_copy_args_buffer_cpu = torch.zeros(
            [sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device="cpu", pin_memory=True
        )
        self.sparse_copy_args_buffer_npu = torch.zeros(
            [sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device="npu"
        )

        self.gvas_buffer_cpu = self.sparse_copy_args_buffer_cpu[
            gvas_buffer_offset : gvas_buffer_offset + gvas_buffer_size_bytes
        ].view(torch.int64)
        self.addr_buffer_cpu = self.sparse_copy_args_buffer_cpu[
            addr_buffer_offset : addr_buffer_offset + addr_buffer_size_bytes
        ].view(torch.int64)
        self.size_buffer_cpu = self.sparse_copy_args_buffer_cpu[
            size_buffer_offset : size_buffer_offset + size_buffer_size_bytes
        ].view(torch.int32)
        self.num_tokens_buffer_cpu = self.sparse_copy_args_buffer_cpu[
            num_tokens_buffer_offset : num_tokens_buffer_offset + num_tokens_buffer_size_bytes
        ].view(torch.int32)
        assert self.gvas_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.addr_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.size_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.num_tokens_buffer_cpu.shape == torch.Size([1])

        self.gvas_buffer_npu = self.sparse_copy_args_buffer_npu[
            gvas_buffer_offset : gvas_buffer_offset + gvas_buffer_size_bytes
        ].view(torch.int64)
        self.addr_buffer_npu = self.sparse_copy_args_buffer_npu[
            addr_buffer_offset : addr_buffer_offset + addr_buffer_size_bytes
        ].view(torch.int64)
        self.size_buffer_npu = self.sparse_copy_args_buffer_npu[
            size_buffer_offset : size_buffer_offset + size_buffer_size_bytes
        ].view(torch.int32)
        self.num_tokens_buffer_npu = self.sparse_copy_args_buffer_npu[
            num_tokens_buffer_offset : num_tokens_buffer_offset + num_tokens_buffer_size_bytes
        ].view(torch.int32)
        assert self.gvas_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.addr_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.size_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.num_tokens_buffer_npu.shape == torch.Size([1])

        # topk cache reuse related
        self.lru_workspace_threads = 8
        self.lru_topk_indices_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_token_to_req_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_to_token_cpu_list = [
            torch.full(
                [self.max_num_topk_rows, self.topk_buffer_size],
                -1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_slots_cpu_list = [
            torch.arange(
                self.topk_buffer_size,
                dtype=torch.int32,
                device="cpu",
            )
            .view(1, -1)
            .repeat(self.max_num_topk_rows, 1)
            .pin_memory()
            for _ in range(self.num_layers)
        ]
        self.lru_current_slots_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_count_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_tokens_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_slots_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device="cpu", pin_memory=True)
        self.lru_stable_prefix_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_last_req_ids_cpu_list = [
            torch.full(
                [self.max_num_topk_rows],
                -1,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_token_mark_workspace = torch.zeros(
            [self.lru_workspace_threads, self.max_model_len],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_token_pos_workspace = torch.full(
            [self.lru_workspace_threads, self.max_model_len],
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk_buffer_size * 3],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_position_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_epochs = torch.zeros(
            [self.lru_workspace_threads],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
        self.lru_stable_prefix_lens_ptr = self.lru_stable_prefix_lens_cpu.data_ptr()
        self.lru_last_req_ids_ptrs = [
            lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list
        ]
        self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
        self.lru_token_to_req_ptr = self.lru_token_to_req_cpu.data_ptr()
        self.lru_slot_to_token_ptrs = [
            lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list
        ]
        self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
        self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
        self.lru_miss_count_ptrs = [
            lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list
        ]
        self.lru_miss_tokens_ptrs = [
            lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list
        ]
        self.lru_miss_slots_ptrs = [
            lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list
        ]
        self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
        self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
        self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
        self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
        self.lru_epochs_ptr = self.lru_epochs.data_ptr()

        if use_nano:
            self._init_nano_fused_workspaces(device)

    def _init_nano_fused_workspaces(self, device: torch.device) -> None:
        """Preallocate LIM / fused_copy_sfa buffers with stable addresses."""
        nano_fused_topk = 2048
        if self.topk != nano_fused_topk:
            raise ValueError(
                f"nano fused ops require sparse_kv_offload_config.topk == {nano_fused_topk}, got {self.topk}"
            )
        rows = self.max_num_topk_rows
        self.lim_topk_src_ids = torch.empty(
            (rows, 1, nano_fused_topk),
            dtype=torch.int32,
            device=device,
        )
        self.lim_topk_dst_slots = torch.empty(
            (rows, 1, nano_fused_topk),
            dtype=torch.int32,
            device=device,
        )
        self.lim_miss_counts = torch.empty(
            (rows,),
            dtype=torch.int32,
            device=device,
        )
        self.lim_query_lens = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        self.lim_num_cache_tokens = torch.full(
            (rows,),
            self.topk_buffer_size,
            dtype=torch.int32,
            device=device,
        )
        self.lim_actual_seq_lengths_kv = torch.empty(
            (rows,),
            dtype=torch.int32,
            device=device,
        )
        self._fused_attention_out: dict[tuple[int, ...], torch.Tensor] = {}
        self._last_lim_batch_size = 0
        # PD first-decode prefix fill: pool entries / batch rows for this step.
        self._nano_init_pool_entries: list[int] = []
        self._nano_init_req_rows: list[int] = []
        self._nano_init_slots_written = False
        self._nano_init_h2d_layers: set[str] = set()
        self._offload_cache_state_cpu: torch.Tensor | None = None
        self._nano_debug_decode_logged = False
        self._nano_debug_prefix_logged = False
        self._nano_debug_meta_logged = False
        self._nano_debug_prefill_logged = False

    def nano_debug_enabled(self) -> bool:
        from vllm_ascend import envs as ascend_envs

        return bool(ascend_envs.VLLM_ASCEND_SPARSE_KV_OFFLOAD_DEBUG)

    def clear_nano_init_step(self) -> None:
        self._nano_init_pool_entries = []
        self._nano_init_req_rows = []
        self._nano_init_slots_written = False
        self._nano_init_h2d_layers = set()
        self._nano_debug_decode_logged = False
        self._nano_debug_prefix_logged = False
        self._nano_debug_meta_logged = False
        self._nano_debug_prefill_logged = False

    def set_nano_init_step(self, req_rows: list[int], pool_entries: list[int]) -> None:
        if len(req_rows) != len(pool_entries):
            raise ValueError(
                f"nano init req_rows and pool_entries length mismatch: {len(req_rows)} vs {len(pool_entries)}"
            )
        self._nano_init_req_rows = list(req_rows)
        self._nano_init_pool_entries = list(pool_entries)
        self._nano_init_slots_written = False
        self._nano_init_h2d_layers = set()

    def has_nano_init_step(self) -> bool:
        return bool(self._nano_init_pool_entries)

    def initialize_nano_prefix_topk_buffer(
        self,
        layer_name: str,
        block_table: torch.Tensor,
        cache_slots_pool: torch.Tensor,
        cache_state: torch.Tensor | None,
        offload_seq_lengths_key: torch.Tensor,
        seq_lengths_key: torch.Tensor,
        *,
        write_slots: bool,
    ) -> None:
        """First-decode: fill hot slots ``0..C-1`` and dense tail blocks.

        LIM plus fused_copy_sfa will rescore and H2D misses, so the initial
        resident set need not be the true top-C. It only needs to be full and
        consistent with ``cache_slots_pool`` (identity ``token i -> slot i``).
        """
        if not self._nano_init_pool_entries:
            return
        if not self._use_nano_fused_op():
            raise RuntimeError("initialize_nano_prefix_topk_buffer requires fused_op_type=nano")

        layer_id = self._get_offload_layer_id(layer_name)
        cache_tokens = self.topk_buffer_size
        block_size = self.block_size
        row_stride_tokens = cache_tokens + 2 * block_size
        token_size_k = self.token_size_bytes_k
        token_size_v = self.token_size_bytes_v
        block_bytes_k = self.cpu_block_lens[layer_id][0]
        block_bytes_v = self.cpu_block_lens[layer_id][1]
        gvas_k_base = self.gvas_k_bases[layer_id]
        gvas_v_base = self.gvas_v_bases[layer_id]
        addr_k_base = self.addr_k_bases[layer_id]
        addr_v_base = self.addr_v_bases[layer_id]
        max_blocks = block_table.size(1)

        offload_lens_cpu = offload_seq_lengths_key.detach().to(device="cpu", dtype=torch.int32)
        seq_lens_cpu = seq_lengths_key.detach().to(device="cpu", dtype=torch.int32)
        for req_row, pool_entry in zip(self._nano_init_req_rows, self._nano_init_pool_entries):
            candidate_tokens = int(offload_lens_cpu[req_row].item())
            if candidate_tokens < cache_tokens:
                raise RuntimeError(
                    "nano first-decode prefix init requires offloaded candidates "
                    f">= topk_buffer_size ({cache_tokens}), got {candidate_tokens} "
                    f"for pool_entry={pool_entry}, req_row={req_row}"
                )
            if candidate_tokens % block_size != 0:
                raise RuntimeError(
                    "nano first-decode candidates must be complete blocks, "
                    f"got candidate_tokens={candidate_tokens}, block_size={block_size}"
                )

        if write_slots and not self._nano_init_slots_written:
            for req_row, pool_entry in zip(self._nano_init_req_rows, self._nano_init_pool_entries):
                slots_row = cache_slots_pool[pool_entry]
                slots_row.fill_(CACHE_SLOTS_POOL_INVALID_ID)
                slots_row[:cache_tokens] = torch.arange(
                    cache_tokens,
                    dtype=torch.int32,
                    device=slots_row.device,
                )
                offload_len = int(offload_lens_cpu[req_row].item())
                seq_len = int(seq_lens_cpu[req_row].item())
                if seq_len > offload_len:
                    tail_tokens = torch.arange(
                        offload_len,
                        seq_len,
                        dtype=torch.int32,
                        device=slots_row.device,
                    )
                    slots_row[offload_len:seq_len] = cache_tokens + (tail_tokens % (2 * block_size))
                if cache_state is not None:
                    cache_state[pool_entry] = CACHE_STATE_OFFLOAD_READY
                if self._offload_cache_state_cpu is not None:
                    self._offload_cache_state_cpu[pool_entry] = CACHE_STATE_OFFLOAD_READY
            self._nano_init_slots_written = True

        if layer_name in self._nano_init_h2d_layers:
            return
        self._nano_init_h2d_layers.add(layer_name)

        max_tokens_per_copy = self.max_num_topk_rows * self.topk
        if max_tokens_per_copy <= 0:
            raise RuntimeError("sparse_copy descriptor capacity is zero")

        pending: list[tuple[int, int, int, int, int, int | None]] = []
        # Hot region: tokens 0..C-1 -> identity slots 0..C-1.
        for req_row, pool_entry in zip(self._nano_init_req_rows, self._nano_init_pool_entries):
            remaining = cache_tokens
            token_start = 0
            while remaining > 0:
                chunk = min(remaining, max_tokens_per_copy)
                pending.append((pool_entry, req_row, token_start, chunk, 0, None))
                token_start += chunk
                remaining -= chunk
            # Dense tail: tokens [offload_len, seq_len) -> slots C + (t % (2*block_size)).
            offload_len = int(offload_lens_cpu[req_row].item())
            seq_len = int(seq_lens_cpu[req_row].item())
            if seq_len > offload_len:
                remaining = seq_len - offload_len
                token_start = offload_len
                while remaining > 0:
                    chunk = min(remaining, max_tokens_per_copy)
                    pending.append((pool_entry, req_row, token_start, chunk, cache_tokens, 2 * block_size))
                    token_start += chunk
                    remaining -= chunk

        block_table_cpu = block_table.detach().to(device="cpu", dtype=torch.int32)
        for pool_entry, req_row, token_start, chunk, slot_base, slot_mod in pending:
            self._sparse_copy_prefix_chunk(
                pool_entry=pool_entry,
                req_row=req_row,
                token_start=token_start,
                token_count=chunk,
                slot_base=slot_base,
                slot_mod=slot_mod,
                block_table_cpu=block_table_cpu,
                max_blocks=max_blocks,
                block_size=block_size,
                row_stride_tokens=row_stride_tokens,
                token_size_k=token_size_k,
                token_size_v=token_size_v,
                block_bytes_k=block_bytes_k,
                block_bytes_v=block_bytes_v,
                gvas_k_base=gvas_k_base,
                gvas_v_base=gvas_v_base,
                addr_k_base=addr_k_base,
                addr_v_base=addr_v_base,
            )
        if self.nano_debug_enabled() and not self._nano_debug_prefix_logged:
            self._nano_debug_prefix_logged = True
            logger.info(
                "nano debug prefix-init layer=%s write_slots=%s pools=%s rows=%s C=%s candidates=%s seqs=%s",
                layer_name,
                write_slots,
                self._nano_init_pool_entries,
                self._nano_init_req_rows,
                cache_tokens,
                [int(offload_lens_cpu[r].item()) for r in self._nano_init_req_rows],
                [int(seq_lens_cpu[r].item()) for r in self._nano_init_req_rows],
            )
            _pe = self._nano_init_pool_entries[0]
            _rr = self._nano_init_req_rows[0]
            _buf_k = self.topk_buffers_k[layer_id]
            _hot = _buf_k[_pe, 0:cache_tokens]
            _off_len = int(offload_lens_cpu[_rr].item())
            _seq_len = int(seq_lens_cpu[_rr].item())
            _tail_len = max(_seq_len - _off_len, 0)
            _tail = _buf_k[_pe, cache_tokens : cache_tokens + _tail_len]
            _tail_full = _buf_k[_pe, cache_tokens : cache_tokens + 2 * block_size]
            _cpu_nz = -1.0
            if self.tp_rank == 0 and self.k_caches_cpu:
                _cpu_nz = float((self.k_caches_cpu[layer_id] != 0).float().mean().item())
            logger.info(
                "nano debug prefix-data layer=%s pool=%s hot_nz=%.4f hot_abs=%.2f "
                "tail_len=%d tail_nz=%.4f tail_abs=%.2f tail_full_nz=%.4f cpu_nz=%.4f k_shape=%s",
                layer_name,
                _pe,
                float((_hot != 0).float().mean().item()),
                float(_hot.float().abs().sum().item()),
                _tail_len,
                float((_tail != 0).float().mean().item()) if _tail.numel() else -1.0,
                float(_tail.float().abs().sum().item()) if _tail.numel() else -1.0,
                float((_tail_full != 0).float().mean().item()),
                _cpu_nz,
                tuple(_buf_k.shape),
            )

    def _sparse_copy_prefix_chunk(
        self,
        *,
        pool_entry: int,
        req_row: int,
        token_start: int,
        token_count: int,
        slot_base: int = 0,
        slot_mod: int | None = None,
        block_table_cpu: torch.Tensor,
        max_blocks: int,
        block_size: int,
        row_stride_tokens: int,
        token_size_k: int,
        token_size_v: int,
        block_bytes_k: int,
        block_bytes_v: int,
        gvas_k_base: int,
        gvas_v_base: int,
        addr_k_base: int,
        addr_v_base: int,
    ) -> None:
        gvas = self.gvas_buffer_cpu
        addr = self.addr_buffer_cpu
        size = self.size_buffer_cpu
        num_tokens_buf = self.num_tokens_buffer_cpu
        gvas.fill_(0)
        addr.fill_(0)
        size.fill_(0)

        for i in range(token_count):
            token = token_start + i
            if slot_mod is None:
                slot = token  # hot region: identity destination
            else:
                slot = slot_base + (token % slot_mod)  # dense tail: ring slot
            block_id = token // block_size
            offset = token % block_size
            if block_id < 0 or block_id >= max_blocks:
                raise RuntimeError(f"nano prefix init token {token} maps to invalid block_id={block_id}")
            block_indice = int(block_table_cpu[req_row, block_id].item())
            if block_indice < 0:
                raise RuntimeError(
                    f"nano prefix init missing DRAM block for token={token}, req_row={req_row}, block_id={block_id}"
                )
            gvas_k = gvas_k_base + block_indice * block_bytes_k + offset * token_size_k
            gvas_v = gvas_v_base + block_indice * block_bytes_v + offset * token_size_v
            addr_k = addr_k_base + (pool_entry * row_stride_tokens + slot) * token_size_k
            addr_v = addr_v_base + (pool_entry * row_stride_tokens + slot) * token_size_v
            gvas[i] = gvas_k
            gvas[token_count + i] = gvas_v
            addr[i] = addr_k
            addr[token_count + i] = addr_v
            size[i] = token_size_k
            size[token_count + i] = token_size_v

        num_tokens_buf[0] = token_count * 2
        self.sparse_copy_args_buffer_npu.copy_(self.sparse_copy_args_buffer_cpu, non_blocking=False)
        if self.tp_size > 1:
            self.tp_group.broadcast(torch.empty([], dtype=torch.int8, device="npu"), src=0)
        result = offload.sparse_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"memfabric H2D sparse_copy for nano prefix init failed with result={result}")

    def get_lim_output_buffers(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size <= 0 or batch_size > self.max_num_topk_rows:
            raise ValueError(f"LIM batch_size out of range: {batch_size}, capacity={self.max_num_topk_rows}")
        return (
            self.lim_topk_src_ids[:batch_size],
            self.lim_topk_dst_slots[:batch_size],
            self.lim_miss_counts[:batch_size],
        )

    def mark_lim_outputs_ready(self, batch_size: int) -> None:
        self._last_lim_batch_size = batch_size

    def require_lim_outputs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._last_lim_batch_size < batch_size:
            raise RuntimeError(
                "nano fused decode shared/skip_topk layer ran before its owner "
                f"LIM update: need batch_size={batch_size}, "
                f"have={self._last_lim_batch_size}"
            )
        return self.get_lim_output_buffers(batch_size)

    def get_fused_attention_out(self, ql_nope: torch.Tensor) -> torch.Tensor:
        key = tuple(ql_nope.shape)
        out = self._fused_attention_out.get(key)
        if out is None or out.dtype != ql_nope.dtype or out.device != ql_nope.device:
            out = torch.empty_like(ql_nope)
            self._fused_attention_out[key] = out
        return out

    def dram_kv_pair_for_fused(self, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Host DRAM CKV/KPE as ``[blocks, block_size, dim]`` for fused_copy_sfa."""
        layer_id = self._get_offload_layer_id(layer_name)
        if layer_id >= len(self.k_caches_cpu) or layer_id >= len(self.v_caches_cpu):
            raise RuntimeError(
                "nano fused_copy_sfa requires host DRAM KV views on all TP ranks, "
                f"layer={layer_name}, tp_rank={self.tp_rank}"
            )
        k_dram = self.k_caches_cpu[layer_id]
        v_dram = self.v_caches_cpu[layer_id]
        if k_dram.ndim == 4 and k_dram.size(2) == 1:
            k_dram = k_dram.squeeze(2)
        if v_dram.ndim == 4 and v_dram.size(2) == 1:
            v_dram = v_dram.squeeze(2)
        if k_dram.ndim != 3 or v_dram.ndim != 3:
            raise RuntimeError(
                f"nano DRAM KV must be 3D after squeeze, got k={tuple(k_dram.shape)}, v={tuple(v_dram.shape)}"
            )
        if not k_dram.is_contiguous() or not v_dram.is_contiguous():
            raise RuntimeError("nano DRAM KV views must stay contiguous non-owning aliases")
        return k_dram, v_dram

    def hbm_kv_pair_for_fused(self, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Paged HBM views ``[blocks, block_size, 1, dim]`` over topk_buffer(+tail)."""
        layer_id = self._get_offload_layer_id(layer_name)
        buffer_k = self.topk_buffers_k[layer_id]
        buffer_v = self.topk_buffers_v[layer_id]
        block_size = self.block_size
        if buffer_k.size(-3) % block_size != 0:
            raise RuntimeError(
                "topk_buffer token dim must be divisible by block_size for "
                f"fused_copy_sfa, got {buffer_k.shape} block_size={block_size}"
            )
        hbm_k = buffer_k.view(-1, block_size, buffer_k.size(-2), buffer_k.size(-1))
        hbm_v = buffer_v.view(-1, block_size, buffer_v.size(-2), buffer_v.size(-1))
        return hbm_k, hbm_v

    def allocate_topk_buffer_slots(self, new_req_ids: list[str]):
        return self.topk_buffer_slot_manager.allocate_topk_buffer_slots(new_req_ids)

    def _nano_copy_geometry(self, layer_name):
        layer = self._get_offload_layer_id(layer_name)
        return dict(
            block_size=self.block_size,
            hot_tokens=self.topk_buffer_size,
            source_bases=(self.gvas_k_bases[layer], self.gvas_v_bases[layer]),
            destination_bases=(self.addr_k_bases[layer], self.addr_v_bases[layer]),
            token_bytes=(self.token_size_bytes_k, self.token_size_bytes_v),
            source_block_capacity=self.k_caches_cpu[layer].shape[0],
        )

    def _copy_nano_kv(self, descriptors, device):
        # TP0 commits new KV to the shared host pool. Every rank must observe
        # that write before reading its own HBM tail/resident inputs.
        if self.tp_size > 1:
            self.tp_group.broadcast(self._nano_copy_fence, src=0)
        result = offload.sparse_copy(*descriptors.args(), device)
        if result not in (None, 0):
            raise RuntimeError(f"memfabric nano H2D sparse_copy failed with result={result}")

    def restore_nano_tail(self, layer_name, batch):
        """Run before copy-SFA in eager execution and on every graph replay."""
        graph = batch.graph_buffers
        if graph is not None:
            descriptors = graph.tail_copies[layer_name]
            active = graph.active_mask
        else:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Nano tail descriptors must be allocated before capture")
            key = (layer_name, batch.seq_lens.numel())
            descriptors = self._nano_tail_copies.get(key)
            if descriptors is None:
                descriptors = CopyDescriptors(
                    KV_COMPONENTS * TAIL_BLOCKS * batch.seq_lens.numel(),
                    batch.seq_lens.device,
                )
                self._nano_tail_copies[key] = descriptors
            active = None
        prepare_tail_copy(descriptors, batch, active=active, **self._nano_copy_geometry(layer_name))
        self._copy_nano_kv(descriptors, batch.seq_lens.device)

    def gather_nano_fallback(self, layer_name, block_table, selected, token_to_request, visible_lengths):
        """Bounded eager fallback for short or mixed-query PD decode batches."""
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Nano short-query fallback must remain eager")
        if self._nano_resident_copy is None or selected.shape[0] > self.max_num_topk_rows:
            raise ValueError("Nano fallback exceeds the allocated query rows")
        slots, table = prepare_resident_gather(
            self._nano_resident_copy,
            block_table=block_table,
            selected=selected,
            token_to_request=token_to_request,
            visible_lengths=visible_lengths,
            **self._nano_copy_geometry(layer_name),
        )
        self._copy_nano_kv(self._nano_resident_copy, selected.device)
        return slots, table

    def offload_new_kv(
        self,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,  # NPU source cache: cache_npu[src_slot] -> cache_cpu[dst_slot]
        v_cache_npu: torch.Tensor | None,
        k: torch.Tensor | None,  # legacy decode: k/v rows -> cache_cpu[slot]
        v: torch.Tensor | None,
        has_prefill: bool = False,
        capturing: bool = False,
        src_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        # the has_prefill path (NPU paged cache -> CPU pool D2H) only exists
        # for single-node PD-colocate debug.
        if self.tp_rank != 0:
            # Decode-produced K/V is replicated across TP ranks, so TP0 alone
            # writes new decode tokens. PD pull fills disjoint parts of this
            # shared pool from all TP ranks through the broadcast GVA.
            return
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("Sparse KV offload TP0 CPU cache is not registered")
        if has_prefill and not self.sparse_kv_offload_config.keep_device_kv_cache:
            raise RuntimeError(
                "Sparse KV offload prefill offload requires "
                "keep_device_kv_cache=True; a PD-disaggregated decode node "
                "never stages prefill KV in an NPU paged cache"
            )

        copy_from_npu_cache = has_prefill or src_slot_mapping is not None
        if copy_from_npu_cache:
            if k_cache_npu is None or v_cache_npu is None:
                raise ValueError("NPU-cache D2H requires K/V device caches")
            device = k_cache_npu.device
        else:
            if k is None or v is None:
                raise ValueError("decode offload requires current-token K/V")
            device = k.device

        dst_slots = slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
        token_count = dst_slots.numel()
        if src_slot_mapping is not None:
            src_slots = src_slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
            if src_slots.numel() != token_count:
                raise ValueError(
                    "src_slot_mapping and slot_mapping must have the same length, "
                    f"got src={src_slots.numel()}, dst={token_count}"
                )
        else:
            src_slots = dst_slots
        if token_count > self.max_num_tokens:
            raise ValueError(
                "Sparse KV offload rows exceed D2H descriptor capacity, "
                f"got {token_count}, capacity={self.max_num_tokens}"
            )

        num_k_slots = k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        num_v_slots = v_cache_cpu.numel() * v_cache_cpu.element_size() // self.token_size_bytes_v
        if num_k_slots != num_v_slots or num_k_slots <= 0:
            raise ValueError(
                f"Sparse KV offload CPU K/V pools have incompatible token capacities: k={num_k_slots}, v={num_v_slots}"
            )
        dst_valid = (dst_slots >= 0) & (dst_slots < num_k_slots)
        safe_dst_slots = dst_slots.clamp(min=0, max=num_k_slots - 1)

        if copy_from_npu_cache:
            assert k_cache_npu is not None and v_cache_npu is not None
            num_src_k_slots = k_cache_npu.numel() * k_cache_npu.element_size() // self.token_size_bytes_k
            num_src_v_slots = v_cache_npu.numel() * v_cache_npu.element_size() // self.token_size_bytes_v
            if num_src_k_slots != num_src_v_slots or num_src_k_slots <= 0:
                raise ValueError(
                    "Sparse KV offload NPU K/V caches have incompatible token "
                    f"capacities: k={num_src_k_slots}, v={num_src_v_slots}"
                )
            src_valid = (src_slots >= 0) & (src_slots < num_src_k_slots)
            valid = dst_valid & src_valid
            safe_src_slots = src_slots.clamp(min=0, max=num_src_k_slots - 1)
            src_k = int(k_cache_npu.data_ptr()) + safe_src_slots * self.token_size_bytes_k
            src_v = int(v_cache_npu.data_ptr()) + safe_src_slots * self.token_size_bytes_v
        else:
            assert k is not None and v is not None
            k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
            v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
            if k_rows.shape[0] != token_count or v_rows.shape[0] != token_count:
                raise ValueError("decode K/V row counts must match slot_mapping")
            if not k_rows.is_contiguous():
                k_rows = k_rows.contiguous()
            if not v_rows.is_contiguous():
                v_rows = v_rows.contiguous()
            token_indices = self.d2h_token_indices_npu[:token_count]
            valid = dst_valid
            src_k = int(k_rows.data_ptr()) + token_indices * self.token_size_bytes_k
            src_v = int(v_rows.data_ptr()) + token_indices * self.token_size_bytes_v

        dst_k = int(k_cache_cpu.data_ptr()) + safe_dst_slots * self.token_size_bytes_k
        dst_v = int(v_cache_cpu.data_ptr()) + safe_dst_slots * self.token_size_bytes_v
        self.d2h_src_ptrs_npu[:token_count].copy_(src_k)
        self.d2h_src_ptrs_npu[token_count : 2 * token_count].copy_(src_v)
        self.d2h_dst_ptrs_npu[:token_count].copy_(dst_k)
        self.d2h_dst_ptrs_npu[token_count : 2 * token_count].copy_(dst_v)
        self.d2h_lengths_npu[:token_count].fill_(self.token_size_bytes_k)
        self.d2h_lengths_npu[token_count : 2 * token_count].fill_(self.token_size_bytes_v)
        self.d2h_lengths_npu[:token_count].masked_fill_(~valid, 0)
        self.d2h_lengths_npu[token_count : 2 * token_count].masked_fill_(~valid, 0)
        self.d2h_size_npu.fill_(2 * token_count)

        result = offload.sparse_copy(
            self.d2h_src_ptrs_npu,
            self.d2h_dst_ptrs_npu,
            self.d2h_lengths_npu,
            self.d2h_size_npu,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"memfabric D2H sparse_copy failed with result={result}")

    def onload_topk_kv(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        block_table: torch.Tensor,
        topk_indices_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        token_to_req_npu: torch.Tensor | None = None,
        capturing: bool = False,
        skip_topk: bool = False,
    ):
        layer_id = self._get_offload_layer_id(layer_name)
        if num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "Sparse KV offload topk rows exceed configured workspace, "
                f"num_tokens={num_tokens}, max_num_topk_rows={self.max_num_topk_rows}"
            )
        if layer_id in [0, self.mtp_layer_id]:
            # metadata which are same across all layers, only compute/copy once in first layer.
            # last layer (mtp layer) may have different metadata, do not skip.
            if token_to_req_npu is not None:
                # spec decode case, expand block_table to actual num decode tokens.
                token_to_req_cpu = self.lru_token_to_req_cpu[:num_tokens]
                token_to_req_cpu.copy_(token_to_req_npu[:num_tokens], non_blocking=capturing)
                block_table_expanded = torch.index_select(block_table, 0, token_to_req_npu[:num_tokens].to(torch.int64))
                self.block_table_expanded_cpu[:num_tokens].copy_(block_table_expanded, non_blocking=capturing)
            else:
                self.block_table_cpu[:num_reqs].copy_(block_table, non_blocking=capturing)
            self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu[:num_tokens], non_blocking=capturing)
            self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(
                stable_prefix_lens_npu[:num_tokens],
                non_blocking=capturing,
            )

        if skip_topk:
            assert layer_id > 0, "No previous layer to reuse."
            gvas_offset = self.gvas_k_bases[layer_id] - self.gvas_k_bases[layer_id - 1]
            addr_offset = self.addr_k_bases[layer_id] - self.addr_k_bases[layer_id - 1]
            assert self.gvas_v_bases[layer_id] - self.gvas_v_bases[layer_id - 1] == gvas_offset, (
                "k/v gvas base delta mismatch."
            )
            assert self.addr_v_bases[layer_id] - self.addr_v_bases[layer_id - 1] == addr_offset, (
                "k/v addr base delta mismatch."
            )
            self.gvas_buffer_npu += gvas_offset
            self.addr_buffer_npu += addr_offset
        else:
            if token_to_req_npu is not None:
                block_table_cpu = self.block_table_expanded_cpu[:num_tokens]
            else:
                block_table_cpu = self.block_table_cpu[:num_reqs]
            topk_indices_cpu = self.lru_topk_indices_cpu[:num_tokens]
            topk_indices_cpu.copy_(topk_indices_npu[:num_tokens], non_blocking=capturing)

            args = (
                num_tokens,
                self.lru_miss_count_cpu_list[layer_id][:num_tokens],
                self.lru_miss_tokens_cpu_list[layer_id][:num_tokens],
                self.lru_miss_slots_cpu_list[layer_id][:num_tokens],
                self.lru_req_ids_ptr,
                self.lru_last_req_ids_ptrs[layer_id],
                self.lru_topk_indices_ptr,
                self.lru_stable_prefix_lens_ptr,
                self.lru_slot_to_token_ptrs[layer_id],
                self.lru_slots_ptrs[layer_id],
                self.lru_current_slots_ptr,
                self.lru_miss_count_ptrs[layer_id],
                self.lru_miss_tokens_ptrs[layer_id],
                self.lru_miss_slots_ptrs[layer_id],
                block_table_cpu,
                self.block_size,
                self.token_size_bytes_k,
                self.token_size_bytes_v,
                self.gvas_k_bases[layer_id],
                self.gvas_v_bases[layer_id],
                self.addr_k_bases[layer_id],
                self.addr_v_bases[layer_id],
                self.lru_token_mark_workspace_ptr,
                self.lru_token_pos_workspace_ptr,
                self.lru_slot_workspace_ptr,
                self.lru_miss_position_workspace_ptr,
                self.lru_epochs_ptr,
                self.gvas_buffer_cpu,
                self.addr_buffer_cpu,
                self.size_buffer_cpu,
                self.num_tokens_buffer_cpu,
                layer_id,
            )

            if capturing:
                current_compute_stream = torch_npu.npu.current_stream()
                subscribed_compute_streams = get_subscribed_compute_streams()
                if current_compute_stream not in subscribed_compute_streams:
                    torch_npu.npu._subscribe_report(current_compute_stream)
                    subscribed_compute_streams.add(current_compute_stream)
                torch_npu.npu._launch_host_func(
                    current_compute_stream,
                    self._onload_topk_kv_cpu,
                    args,
                )
            else:
                self._onload_topk_kv_cpu(args)

            self.sparse_copy_args_buffer_npu.copy_(self.sparse_copy_args_buffer_cpu, non_blocking=capturing)

        if self.tp_size > 1:
            # Make sure that tp0 d2h is finished before other tp's h2d.
            # NOTE we can't use barrier since it can't be captured in graph.
            self.tp_group.broadcast(torch.empty([], dtype=torch.int8, device="npu"), src=0)
        offload.sparse_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )

        current_slots_cpu = self.lru_current_slots_cpu[:num_tokens]
        current_slots_npu[:num_tokens].copy_(current_slots_cpu, non_blocking=capturing)

    def _onload_topk_kv_cpu(self, args):
        # code that is incompatible with graph mode, compute here outside graph
        (
            num_reqs,
            miss_count,
            miss_tokens,
            miss_slots,
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_stable_prefix_lens_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
            layer_id,
        ) = args
        self.sparse_kv_offload_cpp.lru_resident_compact(
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_stable_prefix_lens_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            num_reqs,
            self.topk,
            self.topk_buffer_size,
            self.max_model_len,
            self.lru_workspace_threads,
            self.lru_workspace_threads,
        )
        self.sparse_kv_offload_cpp.compute_lru_resident_addrs(
            miss_count,
            miss_tokens,
            miss_slots,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            self.topk_buffer_size,
            self.lru_workspace_threads,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
        )


_SPARSE_KV_OFFLOAD_MANAGER: SparseKVOffloadManager | None = None


def init_sparse_kv_offload_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    global _SPARSE_KV_OFFLOAD_MANAGER
    if _SPARSE_KV_OFFLOAD_MANAGER is None:
        _SPARSE_KV_OFFLOAD_MANAGER = SparseKVOffloadManager(
            vllm_config,
            kv_cache_config,
            sparse_kv_offload_config,
        )
    return _SPARSE_KV_OFFLOAD_MANAGER


def get_sparse_kv_offload_manager():
    assert _SPARSE_KV_OFFLOAD_MANAGER is not None, "KV offload manager is not initialized."
    return _SPARSE_KV_OFFLOAD_MANAGER


CACHE_STATE_NO_OFFLOAD = -3
CACHE_STATE_OFFLOAD_INIT = -2
CACHE_STATE_OFFLOAD_READY = 0
# fused_li_manage treats negative entries as "not resident"; use -1 to match
# the nano LIM contract (not a sentinel outside int16-ish ranges).
CACHE_SLOTS_POOL_INVALID_ID = -1


def update_sparse_kv_offload_metadata(
    sparse_kv_offload_config: SparseKVOffloadConfig,
    num_tokens: int,
    num_reqs: int,
    num_tokens_padded: int,
    num_reqs_padded: int,
    req_ids: list[str],
    query_start_loc: CpuGpuBuffer,
    num_scheduled_tokens_np: np.array,
    block_size: int,
    decode_threshold: int,
    positions_np: np.array,
    seq_lens_cpu_tensor: torch.Tensor,
    offload_req_ids_tensor: CpuGpuBuffer,
    offload_token_to_req: CpuGpuBuffer,
    offload_req_topk_buffer_slots: CpuGpuBuffer | None,
    offload_device_slot_mapping: CpuGpuBuffer | None,
    offload_device_block_table: CpuGpuBuffer | None,
    offload_seq_lengths_key: CpuGpuBuffer | None,
    offload_cache_state: CpuGpuBuffer | None,
    offload_cache_slots_pool: CpuGpuBuffer | None,
) -> None:
    """Populate per-request identity tensors for the Sparse KV offload LRU.

    Request ids are adler32 hashes of the scheduler request id,
    rows are reset whenever the occupying request changes.
    """
    offload_req_ids_tensor.np[:num_reqs_padded].fill(0)
    effective_num_reqs = min(num_reqs, len(req_ids))
    req_id_values = np.asarray(
        [
            adler32(req_id.encode("utf-8")) if isinstance(req_id, str) else row + 1
            for row, req_id in enumerate(req_ids[:effective_num_reqs])
        ],
        dtype=np.int64,
    )
    offload_req_ids_tensor.np[:effective_num_reqs] = req_id_values
    offload_req_ids_tensor.copy_to_gpu(num_reqs_padded)

    query_start_loc_cpu = query_start_loc.cpu[: num_reqs + 1]
    query_lens = np.diff(query_start_loc_cpu.numpy()).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        raise RuntimeError(
            "KV offload token_to_req metadata is shorter than the scheduled token batch: "
            f"metadata={token_to_req.shape[0]}, tokens={num_tokens}"
        )
    offload_token_to_req.np[:num_tokens] = token_to_req[:num_tokens]
    if num_tokens_padded > num_tokens:
        offload_token_to_req.np[num_tokens:num_tokens_padded].fill(0)
    offload_token_to_req.copy_to_gpu(num_tokens_padded)

    if getattr(sparse_kv_offload_config, "generalized_mtp", False) and effective_num_reqs == 0:
        # Startup graph dummy batches have query rows but no scheduler-owned
        # requests. The graph builder supplies isolated synthetic LIM inputs;
        # never allocate real request slots from this dummy token count.
        offload_req_topk_buffer_slots.np[:num_reqs_padded].fill(-1)
        offload_req_topk_buffer_slots.copy_to_gpu(num_reqs_padded)
        offload_device_slot_mapping.np[:num_tokens_padded].fill(-1)
        offload_device_slot_mapping.copy_to_gpu(num_tokens_padded)
        offload_seq_lengths_key.np[:num_reqs_padded].fill(0)
        offload_seq_lengths_key.copy_to_gpu(num_reqs_padded)
        return

    if sparse_kv_offload_config.fused_op_type == "nano":
        manager = get_sparse_kv_offload_manager()
        manager.clear_nano_init_step()
        manager._offload_cache_state_cpu = offload_cache_state.cpu
        tail_block_num = 2
        topk_buffer_size = sparse_kv_offload_config.topk_buffer_size

        # Allocate pool rows for every request in the step. Decode-only PD
        # disagg batches are unchanged (all reqs are decode). Colocate also
        # allocates during prefill so dual-write and later decode share one row.
        all_req_ids = list(req_ids[:effective_num_reqs])
        topk_buffer_slots, first_allocate = manager.allocate_topk_buffer_slots(all_req_ids)
        if getattr(sparse_kv_offload_config, "generalized_mtp", False):
            for slot, allocated in zip(topk_buffer_slots, first_allocate):
                if allocated:
                    manager.nano_mtp_slot_generations[slot] = manager.nano_mtp_slot_generations.get(slot, 0) + 1
        offload_req_topk_buffer_slots.np[:effective_num_reqs] = np.asarray(
            topk_buffer_slots, dtype=offload_req_topk_buffer_slots.np.dtype
        )
        offload_req_topk_buffer_slots.copy_to_gpu(effective_num_reqs)

        # device_slot_mapping for all tokens: dense-tail slot within each pool row.
        # Used by decode append and by colocate prefill dual-write; PD-disagg D
        # still uses the same mapping for decode token writes into the tail that
        # arrived from P.
        token_pool_entries = np.repeat(
            np.asarray(topk_buffer_slots, dtype=np.int64),
            query_lens[:effective_num_reqs],
        )
        if token_pool_entries.shape[0] < num_tokens:
            raise RuntimeError(
                "KV offload device_slot_mapping metadata is shorter than the "
                f"scheduled token batch: metadata={token_pool_entries.shape[0]}, "
                f"tokens={num_tokens}"
            )
        token_pool_entries = token_pool_entries[:num_tokens]
        positions = positions_np[:num_tokens].astype(np.int64, copy=False)
        offset_in_tail_blocks = positions % (tail_block_num * block_size)
        row_stride = topk_buffer_size + tail_block_num * block_size
        device_slot_mapping = token_pool_entries * row_stride + topk_buffer_size + offset_in_tail_blocks

        # Prefill dual-write / P→D tail transfer only needs the single incomplete
        # last block (at most block_size tokens). Complete blocks stay on DRAM.
        # Decode tokens always keep a valid dense-tail slot for append.
        num_decode_reqs = int((num_scheduled_tokens_np[:effective_num_reqs] <= decode_threshold).sum())
        num_decode_tokens = int(query_lens[:num_decode_reqs].sum()) if num_decode_reqs > 0 else 0
        seq_lens_np = np.asarray(seq_lens_cpu_tensor[:effective_num_reqs].numpy(), dtype=np.int64)
        incomplete_start = (seq_lens_np // block_size) * block_size
        token_req = token_to_req[:num_tokens].astype(np.int64, copy=False)
        token_incomplete_start = incomplete_start[token_req]
        keep_dense = positions >= token_incomplete_start
        if num_decode_tokens > 0:
            keep_dense[:num_decode_tokens] = True
        device_slot_mapping = np.where(keep_dense, device_slot_mapping, -1)

        offload_device_slot_mapping.np[:num_tokens] = device_slot_mapping.astype(
            offload_device_slot_mapping.np.dtype, copy=False
        )
        if num_tokens_padded > num_tokens:
            # -1: ignored by dual-write mask; never treat pad as absolute slot 0.
            offload_device_slot_mapping.np[num_tokens:num_tokens_padded].fill(-1)
        offload_device_slot_mapping.copy_to_gpu(num_tokens_padded)

        if num_decode_reqs > 0:
            # Decode-only metadata: LIM candidates / device block table / PD init.
            decode_query_lens = query_lens[:num_decode_reqs]
            start_positions = positions_np[query_start_loc.np[:num_decode_reqs]]
            end_positions = start_positions + decode_query_lens - 1
            dense_start_block = (start_positions // block_size).astype(np.int64, copy=False)
            dense_end_block = (end_positions // block_size).astype(np.int64, copy=False)
            dense_span = dense_end_block - dense_start_block
            if np.any(dense_span >= tail_block_num):
                raise RuntimeError(
                    "nano dense tail exceeds circular capacity: "
                    f"dense_span={dense_span.tolist()}, tail_block_num={tail_block_num}, "
                    f"start_positions={start_positions.tolist()}, "
                    f"end_positions={end_positions.tolist()}"
                )
            offloaded_block_num = dense_start_block
            # Write via the numpy view: assigning ndarray into torch ``.cpu`` fails.
            offload_lens_np = (offloaded_block_num * block_size).astype(offload_seq_lengths_key.np.dtype, copy=False)
            offload_seq_lengths_key.np[:num_decode_reqs] = offload_lens_np
            offload_seq_lengths_key.copy_to_gpu(num_decode_reqs)

            # device_block_table (matches fused_copy_sfa / nanovllm layout):
            # - offload_len > 0: [hot identity | dense circular], tail starts at C
            # - offload_len == 0: dense circular at the front (num_cache_tokens=0
            #   makes the kernel read logical slots 0.. from block_table[0])
            # Dense physical blocks always sit after the hot arena in the pool
            # row; only the *logical* column placement changes for C=0.
            device_block_table_np = offload_device_block_table.np[:num_decode_reqs]
            device_block_table_np.fill(0)
            topk_buffer_block_num = topk_buffer_size // block_size
            decode_slots = np.asarray(topk_buffer_slots[:num_decode_reqs], dtype=np.int64)
            pool_blocks_per_row = topk_buffer_block_num + tail_block_num
            block_offset_of_req = decode_slots * pool_blocks_per_row

            dense_local = np.empty((num_decode_reqs, tail_block_num), dtype=np.int64)
            dense_local[:, 0] = dense_start_block % tail_block_num
            for tail_block_offset in range(1, tail_block_num):
                dense_local[:, tail_block_offset] = (dense_local[:, tail_block_offset - 1] + 1) % tail_block_num
            dense_physical = dense_local + topk_buffer_block_num

            zero_offload = offload_lens_np <= 0
            nonzero_offload = ~zero_offload
            if np.any(nonzero_offload):
                n_idx = np.nonzero(nonzero_offload)[0]
                device_block_table_np[n_idx, :topk_buffer_block_num] = np.arange(
                    topk_buffer_block_num, dtype=device_block_table_np.dtype
                )
                device_block_table_np[n_idx, topk_buffer_block_num : topk_buffer_block_num + tail_block_num] = (
                    dense_physical[n_idx]
                )
            if np.any(zero_offload):
                z_idx = np.nonzero(zero_offload)[0]
                # Kernel tailSlotStart=0: logical blocks 0.. map to dense.
                device_block_table_np[z_idx, :tail_block_num] = dense_physical[z_idx]

            device_block_table_np[..., :pool_blocks_per_row] += block_offset_of_req[:, None]
            offload_device_block_table.copy_to_gpu()

        # Reset cache_slots for newly allocated pool rows (prefill or decode)
        # before deciding prefix init, so same-step PD allocate sees INIT.
        first_allocate_flags = np.asarray(first_allocate, dtype=bool)
        new_req_slot_ids = np.asarray(topk_buffer_slots, dtype=np.int32)[first_allocate_flags]
        for new_req_slot_id in new_req_slot_ids:
            offload_cache_state.cpu[new_req_slot_id] = CACHE_STATE_OFFLOAD_INIT
            offload_cache_state.gpu[new_req_slot_id].copy_(
                offload_cache_state.cpu[new_req_slot_id],
                non_blocking=True,
            )
            offload_cache_slots_pool.cpu[new_req_slot_id].fill_(CACHE_SLOTS_POOL_INVALID_ID)
            offload_cache_slots_pool.gpu[new_req_slot_id].copy_(
                offload_cache_slots_pool.cpu[new_req_slot_id],
                non_blocking=True,
            )

        if num_decode_reqs > 0:
            # Hot-region prefix init whenever cache is still INIT and there are
            # enough DRAM candidates to fill C. Colocate allocates the pool in
            # prefill (decode first_allocate=False) but leaves state INIT until
            # this first long decode; PD-disagg typically allocates on decode.
            # Dense-tail-only rows (offload_len < C) skip — LIM is not used yet.
            decode_slots_i32 = np.asarray(topk_buffer_slots[:num_decode_reqs], dtype=np.int32)
            cache_states = offload_cache_state.cpu[decode_slots_i32.tolist()].detach().to(dtype=torch.int32).numpy()
            need_prefix_init = (cache_states == CACHE_STATE_OFFLOAD_INIT) & (offload_lens_np >= topk_buffer_size)
            decode_new_slot_ids = decode_slots_i32[need_prefix_init]
            decode_new_rows = np.nonzero(need_prefix_init)[0].astype(np.int32)
            if decode_new_slot_ids.size > 0:
                manager.set_nano_init_step(
                    req_rows=[int(x) for x in decode_new_rows.tolist()],
                    pool_entries=[int(x) for x in decode_new_slot_ids.tolist()],
                )
            if manager.nano_debug_enabled() and not manager._nano_debug_meta_logged:
                manager._nano_debug_meta_logged = True
                dense_cols = device_block_table_np[
                    0, topk_buffer_block_num : topk_buffer_block_num + tail_block_num
                ].tolist()
                logger.info(
                    "nano debug meta decode_reqs=%s C=%s offload_lens=%s "
                    "start_pos=%s dense_start_block=%s prefix_init=%s "
                    "pools=%s device_bt_dense0=%s first_allocate=%s",
                    num_decode_reqs,
                    int(topk_buffer_size),
                    offload_lens_np.tolist(),
                    start_positions.tolist(),
                    dense_start_block.tolist(),
                    bool(decode_new_slot_ids.size > 0),
                    [int(x) for x in decode_slots_i32.tolist()],
                    dense_cols,
                    [bool(x) for x in first_allocate_flags[:num_decode_reqs].tolist()],
                )
