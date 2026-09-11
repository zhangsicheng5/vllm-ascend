"""Standalone SFA backend for Sparse KV offload.

All Sparse KV offload related attention logic lives in this module and is selected by
``AscendSFABackend.get_impl_cls()`` / ``get_builder_cls()`` when
``sparse_kv_offload_config.enabled`` is set, keeping ``sfa_v1.py`` clean.

Data plane (see zsc-sfa-kv-offload-merge-plan.md):

- prefill (debug intermediate state, only reachable with
  ``keep_device_kv_cache=True``): ``exec_kv`` writes the NPU paged main
  cache as usual, then the layer's cache rows are committed D2H
  (  ``cache_cpu[slot] = cache_npu[slot]``) through the manager. With
  ``fused_op_type=nano`` and ``keep_device_kv_cache`` (PD colocate), tokens
  are also dual-written into the topk buffer dense-tail slots
  (``pos % (2*block_size)``). PD-disagg D does not prefill here — its
  incomplete-block dense tail is transferred from P;
- decode (``fused_op_type=default``): no NPU main K/V cache at all
  (indexer K cache only). The current token's K/V is produced compute-only
  and committed D2H directly; top-k misses are loaded H2D into the resident
  (topk) buffer and a single resident SFA attention runs.
- decode (``fused_op_type=nano``): write the current token into the two
  HBM tail blocks of the topk buffer, then D2H from those tail slots into
  the CPU paged pool; decode attention uses ``npu_fused_li_manage`` +
  ``npu_fused_copy_sfa`` (``torch.ops._C_ascend``). When there are no
  offloaded LIM candidates (``offload_seq_lengths_key==0``), attention is
  dense-tail only (``num_cache_tokens=0``) for both colocate and disagg.
"""

from dataclasses import dataclass, replace
from typing import Any, TypeVar

import torch
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

import vllm_ascend.vllm_ascend_C  # noqa: E402,F401  (register torch.ops._C_ascend fused LIM/SFA ops)
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    PreprocessType,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    build_valid_topk_mask,
    split_decodes_and_prefills,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp import (
    GeneralizedMtpRuntime,
    MtpBatch,
    make_mtp_batch,
    prepare_copy_sfa_queries,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_sparse_kv_offload_manager,
)
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.utils import enable_dsa_cp

NANO_FUSED_TOPK = 2048

M = TypeVar("M", bound=AscendSFAMetadata)


def _generalized_mtp_enabled() -> bool:
    return getattr(get_ascend_config().sparse_kv_offload_config, "generalized_mtp", False)


def _mtp_runtime(manager) -> GeneralizedMtpRuntime:
    if not hasattr(manager, "generalized_mtp_runtime"):
        manager.generalized_mtp_runtime = GeneralizedMtpRuntime(manager)
    return manager.generalized_mtp_runtime


@dataclass
class AscendSFAKVOffloadMetadata(AscendSFAMetadata):
    mtp_batch: MtpBatch | None = None
    mtp_graph_capture: bool = False
    mtp_graph_dummy: bool = False
    main_slot_mapping: torch.Tensor | None = None
    indexer_block_table: torch.Tensor | None = None


def _check_device_kv_cache_exist() -> None:
    # prefill/mixed handling only exists for single-node PD-colocate debug;
    # a PD-disaggregated decode node never receives prefill batches.
    if not get_ascend_config().sparse_kv_offload_config.keep_device_kv_cache:
        raise RuntimeError(
            "Sparse KV offload received a prefill/mixed batch without "
            "keep_device_kv_cache=True; a PD-disaggregated decode node "
            "only accepts decode requests"
        )


class AscendSFAKVOffloadMetadataBuilder(AscendSFAMetadataBuilder):
    """Fills the offload-specific SFA metadata (decode split + request ids)."""

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls or AscendSFAKVOffloadMetadata,
            supports_dcp_with_varlen,
        )
        kv_transfer_config = vllm_config.kv_transfer_config
        self.is_pd_decode_consumer = (
            kv_transfer_config is not None
            and kv_transfer_config.is_kv_consumer
            and not kv_transfer_config.is_kv_producer
        )
        self._mtp_capture_width = (
            1 + vllm_config.speculative_config.num_speculative_tokens
            if getattr(vllm_config, "speculative_config", None) is not None
            else 1
        )

    def _populate_offload_metadata(
        self,
        metadata: AscendSFAMetadata,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        for_graph_capture: bool = False,
        draft_index: int = 0,
    ) -> AscendSFAMetadata:
        is_dummy = getattr(common_attn_metadata, "offload_is_dummy", False)
        if _generalized_mtp_enabled() and not for_graph_capture and not is_dummy:
            # FIA/FlashComm padding adds artificial requests. Ownership and
            # the generalized ABI describe only the scheduled query rows.
            starts = common_attn_metadata.query_start_loc_cpu[: common_attn_metadata.num_reqs]
            ends = common_attn_metadata.query_start_loc_cpu[1 : common_attn_metadata.num_reqs + 1]
            actual_reqs = int(((ends > starts) & (ends <= common_attn_metadata.num_actual_tokens)).sum())
            common_attn_metadata = common_attn_metadata.unpadded(
                common_attn_metadata.num_actual_tokens,
                actual_reqs,
            )
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            # The D node has already loaded the prompt KV from P. vLLM still
            # marks the one-token boundary step as prefilling because its
            # computed-token count is N - 1, but SFA must execute it through
            # the decode-offload path. Keep colocated producer/debug behavior
            # unchanged so genuine short prefills still populate the cache.
            treat_short_extends_as_decodes=self.is_pd_decode_consumer,
        )
        metadata.num_decodes = num_decodes
        metadata.num_prefills = num_prefills
        metadata.num_decode_tokens = num_decode_tokens
        metadata.req_ids_tensor = common_attn_metadata.req_ids_tensor
        metadata.token_to_req = common_attn_metadata.token_to_req
        if _generalized_mtp_enabled():
            # The base builder reuses its length buffers while preparing all
            # draft steps before any of them execute. The eager fallback also
            # needs step-owned lengths: later Q1 drafts must not overwrite the
            # first step's multi-query boundaries or causal sequence lengths.
            metadata.cum_query_lens = metadata.cum_query_lens.clone()
            metadata.seq_lens = metadata.seq_lens.clone()
            # Draft metadata must carry request pool ownership, but all lengths
            # and tail locations are rebuilt after speculative rejection.
            metadata.req_topk_buffer_slots = common_attn_metadata.req_topk_buffer_slots
            manager = get_sparse_kv_offload_manager()
            _mtp_runtime(manager)
            if for_graph_capture or is_dummy:
                from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp_graph import (
                    make_dummy_batch,
                )

                metadata.mtp_graph_capture = for_graph_capture
                metadata.mtp_graph_dummy = is_dummy
                metadata.slot_mapping.fill_(-1)
                metadata.mtp_batch = make_dummy_batch(
                    metadata,
                    manager,
                    self._mtp_capture_width if draft_index == 0 else 1,
                )
            else:
                # The base SFA builder may retain padded prefix buffers.
                actual = replace(
                    metadata,
                    cum_query_lens=common_attn_metadata.query_start_loc[1:],
                    seq_lens=common_attn_metadata.seq_lens,
                    block_table=common_attn_metadata.block_table_tensor,
                )
                host_lengths = getattr(common_attn_metadata, "_seq_lens_cpu", None)
                if host_lengths is None:
                    host_lengths = getattr(common_attn_metadata, "seq_lens_cpu", None)
                metadata.mtp_batch = make_mtp_batch(
                    actual,
                    manager,
                    query_ends_cpu=common_attn_metadata.query_start_loc_cpu[1:],
                    seq_lens_cpu=host_lengths,
                    pool_rows_cpu=getattr(common_attn_metadata, "req_topk_buffer_slots_cpu", None),
                )
            if metadata.mtp_batch is None:
                _mtp_runtime(manager).invalidate()
            return metadata
        if get_ascend_config().sparse_kv_offload_config.fused_op_type == "nano":
            num_reqs = int(common_attn_metadata.num_reqs)
            num_actual_tokens = int(common_attn_metadata.num_actual_tokens)
            # Pool rows / device_slot_mapping cover the whole step so both
            # PD-disagg decode and colocate prefill+decode share one layout.
            metadata.req_topk_buffer_slots = common_attn_metadata.req_topk_buffer_slots[:num_reqs]
            metadata.device_slot_mapping = common_attn_metadata.device_slot_mapping[:num_actual_tokens]
            metadata.device_block_table = (
                common_attn_metadata.device_block_table[:num_decodes] if num_decodes > 0 else None
            )
            metadata.offload_seq_lengths_key = (
                common_attn_metadata.offload_seq_lengths_key[:num_decodes] if num_decodes > 0 else None
            )
            metadata.cache_state = common_attn_metadata.cache_state
            metadata.cache_slots_pool = common_attn_metadata.cache_slots_pool
        return metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> AscendSFAMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build, **kwargs)
        return self._populate_offload_metadata(metadata, common_attn_metadata)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build_for_graph_capture(common_attn_metadata, common_attn_metadata.attn_state)

    def build_for_graph_capture(self, common_attn_metadata, attn_state, *, draft_index=0):
        if not _generalized_mtp_enabled():
            return super().build_for_graph_capture(common_attn_metadata, attn_state)
        metadata = super()._build(common_attn_metadata)
        metadata.attn_state = attn_state
        # Dummy queries must not write the retained device or host history.
        metadata.slot_mapping.fill_(-1)
        return self._populate_offload_metadata(
            metadata,
            common_attn_metadata,
            for_graph_capture=True,
            draft_index=draft_index,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        **kwargs: Any,
    ) -> AscendSFAMetadata:
        metadata = super().build_for_drafting(
            common_attn_metadata,
            draft_index,
            **kwargs,
        )
        return self._populate_offload_metadata(metadata, common_attn_metadata)


class AscendSFAKVOffloadImpl(AscendSFAImpl):
    """SFA implementation that routes main MLA K/V through the CPU pool."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )
        if enable_dsa_cp():
            raise NotImplementedError("Sparse KV offload currently requires TP without context parallelism")
        if self.enable_sparse_sfa_c8:
            raise NotImplementedError(
                "Sparse KV offload does not support the sparse SFA C8 main "
                "cache; sparse LI C8 is supported for the device-resident "
                "indexer cache."
            )
        self._current_layer_name: str | None = None

    def _resolve_preprocess_type(self, act_dtype: torch.dtype) -> PreprocessType:
        logger.warning_once(
            "Sparse KV offload requires the native SFA preprocessing path; sfa_prolog_v3/mlapo is disabled."
        )
        return PreprocessType.NATIVE

    @staticmethod
    def _use_nano_fused_op() -> bool:
        return get_ascend_config().sparse_kv_offload_config.fused_op_type == "nano"

    @staticmethod
    def _cpu_cache_pair(manager, layer_name: str):
        layer_id = manager._get_offload_layer_id(layer_name)
        # default: only TP0 allocates CPU pools. nano: non-0 ranks restore
        # shared GVA views in register_kv_caches for fused_copy_sfa.
        if layer_id >= len(manager.k_caches_cpu) or layer_id >= len(manager.v_caches_cpu):
            return None, None
        return manager.k_caches_cpu[layer_id], manager.v_caches_cpu[layer_id]

    @staticmethod
    def _topk_buffer_pair(manager, layer_name: str):
        layer_id = manager._get_offload_layer_id(layer_name)
        return manager.topk_buffers_k[layer_id], manager.topk_buffers_v[layer_id]

    @staticmethod
    def _scatter_kv_to_device_slots(
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        buffer_k: torch.Tensor,
        buffer_v: torch.Tensor,
        device_slots: torch.Tensor,
    ) -> None:
        slot_index = device_slots.reshape(-1, 1).to(dtype=torch.int64)
        torch_npu.npu_scatter_nd_update_(
            buffer_k.reshape(-1, k_nope.shape[-1]),
            slot_index,
            k_nope.reshape(-1, k_nope.shape[-1]),
        )
        torch_npu.npu_scatter_nd_update_(
            buffer_v.reshape(-1, k_pe.shape[-1]),
            slot_index,
            k_pe.reshape(-1, k_pe.shape[-1]),
        )

    @staticmethod
    def _resident_views(manager, layer_name: str, rows: int):
        layer_id = manager._get_offload_layer_id(layer_name)
        buffer_k = manager.topk_buffers_k[layer_id]
        buffer_v = manager.topk_buffers_v[layer_id]
        pages_per_row = manager.topk_buffer_size // manager.block_size
        resident_pages = rows * pages_per_row
        resident_k = buffer_k[:rows].view(
            resident_pages,
            manager.block_size,
            buffer_k.shape[-2],
            buffer_k.shape[-1],
        )
        resident_v = buffer_v[:rows].view(
            resident_pages,
            manager.block_size,
            buffer_v.shape[-2],
            buffer_v.shape[-1],
        )
        return (
            resident_k,
            resident_v,
            manager.current_slots_npu[:rows],
            manager.resident_block_table_npu[:rows],
            manager.resident_query_lens_npu[:rows],
            manager.resident_seq_lens_npu[:rows],
        )

    def _offload_layer_name(self) -> str:
        layer_name = self.layer_name or self._current_layer_name
        if layer_name is None:
            raise RuntimeError("Sparse KV offload requires a bound attention layer name")
        return layer_name

    @staticmethod
    def _is_decode_only(attn_metadata: M) -> bool:
        return (
            attn_metadata.attn_state in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
            and int(getattr(attn_metadata, "num_prefills", 0) or 0) == 0
            and int(getattr(attn_metadata, "num_decodes", 0) or 0) > 0
        )

    @staticmethod
    def _pad_to_input_tokens(
        attn_output: torch.Tensor,
        num_input_tokens: int,
    ) -> torch.Tensor:
        if attn_output.shape[0] >= num_input_tokens:
            return attn_output
        padded = attn_output.new_zeros(num_input_tokens, *attn_output.shape[1:])
        padded[: attn_output.shape[0]] = attn_output
        return padded

    @staticmethod
    def _in_graph_runtime() -> bool:
        if not is_forward_context_available():
            return False
        forward_context = get_forward_context()
        runtime_mode = getattr(
            forward_context,
            "cudagraph_runtime_mode",
            CUDAGraphMode.NONE,
        )
        return forward_context.capturing or runtime_mode not in (
            None,
            CUDAGraphMode.NONE,
        )

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._current_layer_name = layer_name
        try:
            if _generalized_mtp_enabled() and self.has_indexer and attn_metadata is not None:
                indexer_name = self.indexer.k_cache.prefix
                context = get_forward_context()
                indexer_metadata = context.attn_metadata.get(indexer_name)
                if indexer_metadata is not None:
                    attn_metadata = replace(
                        attn_metadata,
                        main_slot_mapping=attn_metadata.slot_mapping,
                        slot_mapping=indexer_metadata.slot_mapping,
                        indexer_block_table=indexer_metadata.block_table,
                    )
                else:
                    # Draft builders omit cache-only indexer layers. Aliasing
                    # their IDs is safe only for an explicitly shared group.
                    groups = get_sparse_kv_offload_manager().kv_cache_config.kv_cache_groups
                    group_ids = {name: gid for gid, group in enumerate(groups) for name in group.layer_names}
                    if group_ids.get(layer_name) is None or group_ids.get(layer_name) != group_ids.get(indexer_name):
                        raise RuntimeError(
                            "Nano draft attention requires explicit indexer metadata for separate KV groups"
                        )
            return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output)
        finally:
            self._current_layer_name = None

    def _get_sfa_kv_slot_mapping(self, attn_metadata):
        main_slots = getattr(attn_metadata, "main_slot_mapping", None)
        return attn_metadata.slot_mapping if main_slots is None else main_slots

    def _compute_kv_only(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode-only KV generation that never touches an NPU paged cache."""
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        assert self.kv_a_layernorm is not None, "kv_a_layernorm must be initialized"
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        rms_in, rope_in = kv_no_split.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_nope_flat, _ = torch_npu.npu_rms_norm(
            rms_in.view(-1, self.kv_lora_rank),
            self.kv_a_layernorm.weight,
            epsilon=self.kv_a_layernorm.variance_epsilon,
        )
        k_nope = k_nope_flat.view(B, N, S, self.kv_lora_rank)
        k_pe = torch_npu.npu_interleave_rope(
            rope_in,
            cos,
            sin,
        )
        return k_nope, k_pe

    def _pd_decode_consumer(self) -> bool:
        # Kept for callers that distinguish PD-disagg decode; nano prefix
        # init itself is no longer gated on this (colocate also needs it).
        return bool(self.is_kv_consumer and not self.is_kv_producer)

    def _maybe_nano_prefix_init(
        self,
        attn_metadata: M,
        manager,
        layer_name: str,
        seq_lengths_key: torch.Tensor,
    ) -> None:
        if not manager.has_nano_init_step():
            return
        if self._in_graph_runtime():
            raise RuntimeError("nano first-decode topk-buffer prefix init must remain eager")
        num_decodes = int(attn_metadata.num_decodes or 0)
        if attn_metadata.cache_slots_pool is None or attn_metadata.offload_seq_lengths_key is None:
            raise RuntimeError("nano prefix init requires cache_slots_pool and offload_seq_lengths_key")
        manager.initialize_nano_prefix_topk_buffer(
            layer_name=layer_name,
            block_table=attn_metadata.block_table[:num_decodes],
            cache_slots_pool=attn_metadata.cache_slots_pool,
            cache_state=attn_metadata.cache_state,
            offload_seq_lengths_key=attn_metadata.offload_seq_lengths_key[:num_decodes],
            seq_lengths_key=seq_lengths_key[:num_decodes],
            write_slots=not self.skip_topk,
        )

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ):
        mtp_enabled = _generalized_mtp_enabled()
        if (mtp_enabled and attn_metadata.mtp_batch is None) or not (
            self._use_nano_fused_op() and self._is_decode_only(attn_metadata)
        ):
            indexer_table = getattr(attn_metadata, "indexer_block_table", None)
            indexer_metadata = (
                attn_metadata if indexer_table is None else replace(attn_metadata, block_table=indexer_table)
            )
            return super().indexer_select_post_process(
                x,
                q_c,
                kv_cache,
                indexer_metadata,
                cos,
                sin,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
            )
        return self._nano_fused_li_manage(
            x,
            q_c,
            kv_cache,
            attn_metadata,
            cos,
            sin,
            actual_seq_lengths_key,
        )

    def _nano_fused_li_manage(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_indexer:
            raise RuntimeError(f"nano fused_li_manage requires an indexer. layer_name={self.layer_name}.")
        if self.enable_sparse_li_c8:
            raise NotImplementedError("nano fused_li_manage does not support sparse LI C8")
        assert self.wk_weights_proj is not None
        assert self.wq_b is not None

        num_decodes = int(attn_metadata.num_decodes or 0)
        num_decode_tokens = int(attn_metadata.num_decode_tokens or 0)
        mtp_batch = getattr(attn_metadata, "mtp_batch", None)
        if num_decodes <= 0:
            raise RuntimeError("nano fused_li_manage requires decode requests")
        if mtp_batch is None and num_decode_tokens != num_decodes:
            raise NotImplementedError(
                "nano fused_li_manage does not support MTP multi-token decode yet "
                f"(num_decode_tokens={num_decode_tokens}, num_decodes={num_decodes})"
            )
        if attn_metadata.req_topk_buffer_slots is None:
            raise RuntimeError("nano fused_li_manage requires req_topk_buffer_slots")
        if mtp_batch is None and attn_metadata.cache_slots_pool is None:
            raise RuntimeError("nano fused_li_manage requires cache_slots_pool")
        if mtp_batch is None and attn_metadata.offload_seq_lengths_key is None:
            raise RuntimeError("nano fused_li_manage requires offload_seq_lengths_key")

        manager = get_sparse_kv_offload_manager()
        layer_name = self._offload_layer_name()
        if mtp_batch is None:
            self._maybe_nano_prefix_init(attn_metadata, manager, layer_name, actual_seq_lengths_key)
            topk_src_ids, topk_dst_slots, miss_counts = manager.get_lim_output_buffers(num_decodes)
            offload_lens = attn_metadata.offload_seq_lengths_key[:num_decodes]
        # No complete offloaded blocks yet: dense-tail only (typical short
        # colocate prompt). Skip LIM and leave miss_counts at 0.
        if mtp_batch is None and bool(torch.all(offload_lens <= 0).item()):
            topk_src_ids.zero_()
            topk_dst_slots.zero_()
            miss_counts.zero_()
            manager.mark_lim_outputs_ready(num_decodes)
            return topk_src_ids

        kw, _ = self.wk_weights_proj(x)
        weights = kw[:, self.head_dim :]
        if isinstance(q_c, tuple):
            q_c_tensor, q_c_scale = q_c
            q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
            quant_matmul_kwargs = dict(
                bias=None,
                output_dtype=x.dtype,
            )
            if q_c_tensor.dtype == torch.float8_e4m3fn:
                if q_c_scale.dim() == 2:
                    q_c_scale = q_c_scale.view(q_c_scale.shape[0], -1, 2)
                quant_matmul_kwargs.update(
                    scale_dtype=torch_npu.float8_e8m0fnu,
                    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                    group_sizes=[1, 1, getattr(self.wq_b.quant_method.quant_method, "group_size", 32)],
                )
            elif q_c_scale.dim() > 1 and q_c_scale.shape[-1] == 1:
                q_c_scale = q_c_scale.squeeze(dim=-1)
            q_li = torch_npu.npu_quant_matmul(
                q_c_tensor,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_c_scale,
                **quant_matmul_kwargs,
            )
        else:
            q_li, _ = self.wq_b(q_c)
        q_li = q_li.view(-1, self.n_head, self.head_dim)
        if HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            q_li_pe, q_li_nope = torch.split(
                q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )
            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

        query_rows = mtp_batch.num_tokens if mtp_batch is not None else num_decodes
        q_li = q_li[:query_rows]
        weights = weights[:query_rows]
        if q_li.shape[1] not in (32, 64):
            raise RuntimeError(f"nano fused_li_manage expects 32 or 64 index heads, got {q_li.shape[1]}")
        if q_li.shape[-1] != 128:
            raise RuntimeError(f"nano fused_li_manage expects head_dim=128, got {q_li.shape[-1]}")

        index_key = kv_cache[self.kv_cache_indexer_k_idx]
        block_size = manager.block_size
        # LIM requires [blocks, 128, 1, 128].
        if index_key.ndim == 4 and index_key.size(1) == block_size and index_key.size(2) == 1:
            index_key_cache = index_key
        else:
            index_key_cache = index_key.view(-1, block_size, 1, self.head_dim)

        block_table = attn_metadata.block_table[:num_decodes]
        if mtp_batch is not None:
            indexer_table = getattr(attn_metadata, "indexer_block_table", None)
            block_table = mtp_batch.source_block_table if indexer_table is None else indexer_table[:num_decodes]
            runtime = _mtp_runtime(manager)
            mapping, states, outputs = runtime.prepare_lim(
                layer_name,
                mtp_batch,
                block_table.size(1) * block_size,
                q_li.device,
            )
            # The pinned floating-point LIM ABI reserves scale inputs; the
            # kernel does not read their values (this is not LI C8 support).
            query_scale = torch.empty(q_li.shape[:2], dtype=torch.float32, device=q_li.device)
            key_scale = torch.empty(index_key_cache.shape[:3], dtype=torch.float32, device=q_li.device)
            torch.ops._C_ascend.npu_fused_li_manage_mtp(
                weights.contiguous(),
                query_scale,
                q_li.contiguous(),
                key_scale,
                index_key_cache.contiguous(),
                block_table.contiguous(),
                mtp_batch.query_ends,
                mtp_batch.seq_lens,
                mtp_batch.offload_lens,
                mtp_batch.cache_tokens,
                states,
                mtp_batch.pool_entries,
                mapping,
                *outputs,
            )
            if (
                mtp_batch.graph_buffers is None
                and manager.nano_debug_enabled()
                and not getattr(manager, "_nano_debug_li_steps", 0)
            ):
                manager._nano_debug_li_steps = 1
                logger.info(
                    "generalized MTP LIM executed: layer=%s T=%s B=%s prefix=%s C=%s request_misses=%s",
                    layer_name,
                    query_rows,
                    num_decodes,
                    mtp_batch.prefix_lengths,
                    mtp_batch.cache_sizes,
                    outputs[-1].cpu().tolist(),
                )
            return outputs[0]
        _li_layer_id = manager._get_offload_layer_id(layer_name)
        _li_steps = getattr(manager, "_nano_debug_li_steps", 0)
        if _li_layer_id == 0 and manager.nano_debug_enabled() and _li_steps < 6:
            manager._nano_debug_li_steps = _li_steps + 1
            _idx = index_key_cache.detach()
            _cand = (
                attn_metadata.offload_seq_lengths_key[:num_decodes]
                .detach()
                .to(device="cpu", dtype=torch.int32)
                .tolist()
            )
            _bt = block_table.detach().to(device="cpu", dtype=torch.int32).tolist()
            logger.info(
                "nano debug li_indexer layer=%s step=%d indexer_shape=%s nz_ratio=%.4f "
                "abs_sum=%.2f candidates=%s block_table=%s",
                layer_name,
                _li_steps,
                tuple(index_key_cache.shape),
                float((_idx != 0).float().mean().item()),
                float(_idx.float().abs().sum().item()),
                _cand,
                _bt,
            )
        expected_slots_width = block_table.size(1) * block_size
        cache_slots_pool = attn_metadata.cache_slots_pool
        if cache_slots_pool.size(1) != expected_slots_width:
            raise RuntimeError(
                "cache_slots_pool width must equal block_table_cols * block_size for LIM, "
                f"got pool={cache_slots_pool.size(1)}, expected={expected_slots_width}"
            )

        # Candidates = fully offloaded blocks only; dense tail is attended by SFA.
        num_candidate_tokens = offload_lens.to(dtype=torch.int32)
        num_cache_tokens = manager.lim_num_cache_tokens[:num_decodes]
        req_pool_entries = attn_metadata.req_topk_buffer_slots[:num_decodes].to(dtype=torch.int32)

        if not cache_slots_pool.is_contiguous():
            raise RuntimeError("cache_slots_pool must be contiguous for in-place LIM updates")
        if not block_table.is_contiguous():
            block_table = block_table.contiguous()
        if not num_candidate_tokens.is_contiguous():
            num_candidate_tokens = num_candidate_tokens.contiguous()
        if not req_pool_entries.is_contiguous():
            req_pool_entries = req_pool_entries.contiguous()
        if not index_key_cache.is_contiguous():
            index_key_cache = index_key_cache.contiguous()

        torch.ops._C_ascend.npu_fused_li_manage(
            q_li.contiguous(),
            weights.contiguous(),
            index_key_cache,
            block_table,
            num_candidate_tokens,
            num_cache_tokens,
            req_pool_entries,
            cache_slots_pool,
            topk_src_ids,
            topk_dst_slots,
            miss_counts,
        )
        manager.mark_lim_outputs_ready(num_decodes)

        # IndexShare / skip_topk consumers reuse topk_indices_buffer.
        # Shape matches lightning_indexer: [T, 1, topk].
        return topk_src_ids

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        if _generalized_mtp_enabled():
            if not get_ascend_config().sparse_kv_offload_config.keep_device_kv_cache:
                if not self._is_decode_only(attn_metadata):
                    _check_device_kv_cache_exist()
                k_nope, k_pe = self._compute_kv_only(kv_no_split, cos, sin)
                manager = get_sparse_kv_offload_manager()
                k_cpu, v_cpu = self._cpu_cache_pair(manager, self._offload_layer_name())
                manager.offload_new_kv(
                    slot_mapping=slots,
                    k_cache_cpu=k_cpu,
                    v_cache_cpu=v_cpu,
                    k_cache_npu=None,
                    v_cache_npu=None,
                    k=k_nope,
                    v=k_pe,
                    has_prefill=False,
                    capturing=self._in_graph_runtime(),
                )
                return k_pe, k_nope
            # Retain a complete device cache for colocated prefill/short-batch
            # fallback, and update the host pool used by nano's tail restore.
            result = super().exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
            manager = get_sparse_kv_offload_manager()
            k_cpu, v_cpu = self._cpu_cache_pair(manager, self._offload_layer_name())
            manager.offload_new_kv(
                slot_mapping=slots,
                k_cache_cpu=k_cpu,
                v_cache_cpu=v_cpu,
                k_cache_npu=kv_cache[0],
                v_cache_npu=kv_cache[1],
                k=None,
                v=None,
                has_prefill=True,
                capturing=self._in_graph_runtime(),
            )
            return result
        if self._is_decode_only(attn_metadata):
            k_nope, k_pe = self._compute_kv_only(kv_no_split, cos, sin)
            manager = get_sparse_kv_offload_manager()
            layer_name = self._offload_layer_name()
            k_cache_cpu, v_cache_cpu = self._cpu_cache_pair(manager, layer_name)
            if self._use_nano_fused_op():
                device_slots = attn_metadata.device_slot_mapping
                if device_slots is None:
                    raise RuntimeError("nano fused decode requires device_slot_mapping metadata")
                token_count = device_slots.numel()
                k_nope = k_nope[:token_count]
                k_pe = k_pe[:token_count]
                dst_slots = slots.reshape(-1)[:token_count]
                buffer_k, buffer_v = self._topk_buffer_pair(manager, layer_name)
                self._scatter_kv_to_device_slots(
                    k_nope,
                    k_pe,
                    buffer_k,
                    buffer_v,
                    device_slots,
                )
                manager.offload_new_kv(
                    slot_mapping=dst_slots,
                    k_cache_cpu=k_cache_cpu,
                    v_cache_cpu=v_cache_cpu,
                    k_cache_npu=buffer_k,
                    v_cache_npu=buffer_v,
                    k=None,
                    v=None,
                    has_prefill=False,
                    capturing=self._in_graph_runtime(),
                    src_slot_mapping=device_slots,
                )
            else:
                manager.offload_new_kv(
                    slot_mapping=slots,
                    k_cache_cpu=k_cache_cpu,
                    v_cache_cpu=v_cache_cpu,
                    k_cache_npu=None,
                    v_cache_npu=None,
                    k=k_nope,
                    v=k_pe,
                    has_prefill=False,
                    capturing=self._in_graph_runtime(),
                )
            return k_pe, k_nope

        # Prefill / mixed batch (colocate debug only): stage in the NPU paged
        # main cache as usual, then commit the written rows D2H into the
        # shared CPU pool. Complete blocks live on DRAM for later LIM. Nano
        # colocate additionally dual-writes into the dense tail below;
        # PD-disagg D never enters this path (no local prefill).
        _check_device_kv_cache_exist()
        result = super().exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
        manager = get_sparse_kv_offload_manager()
        layer_name = self._offload_layer_name()

        k_cache_cpu, v_cache_cpu = self._cpu_cache_pair(manager, layer_name)
        manager.offload_new_kv(
            slot_mapping=slots,
            k_cache_cpu=k_cache_cpu,
            v_cache_cpu=v_cache_cpu,
            k_cache_npu=kv_cache[0],
            v_cache_npu=kv_cache[1],
            k=None,
            v=None,
            has_prefill=True,
            capturing=self._in_graph_runtime(),
        )
        if self._use_nano_fused_op() and get_ascend_config().sparse_kv_offload_config.keep_device_kv_cache:
            # Colocate only: dual-write the single incomplete last block into
            # dense tail. Complete-block tokens have device_slot_mapping=-1.
            # PD-disagg D gets that one-block tail from P (not handled here).
            device_slots = attn_metadata.device_slot_mapping
            if device_slots is None:
                raise RuntimeError("nano colocate prefill requires device_slot_mapping for dense-tail dual-write")
            token_count = min(int(slots.numel()), int(device_slots.numel()))
            if token_count > 0:
                src_slots = slots.reshape(-1)[:token_count].to(dtype=torch.int64)
                dst_slots = device_slots.reshape(-1)[:token_count]
                tail_mask = dst_slots >= 0
                if bool(tail_mask.any().item()):
                    src_slots = src_slots[tail_mask]
                    dst_slots = dst_slots[tail_mask]
                    buffer_k, buffer_v = self._topk_buffer_pair(manager, layer_name)
                    k_dim = buffer_k.shape[-1]
                    v_dim = buffer_v.shape[-1]
                    k_rows = kv_cache[0].reshape(-1, k_dim).index_select(0, src_slots)
                    v_rows = kv_cache[1].reshape(-1, v_dim).index_select(0, src_slots)
                    self._scatter_kv_to_device_slots(
                        k_rows,
                        v_rows,
                        buffer_k,
                        buffer_v,
                        dst_slots,
                    )
                    if manager.nano_debug_enabled() and not manager._nano_debug_prefill_logged:
                        manager._nano_debug_prefill_logged = True
                        logger.info(
                            "nano debug prefill dual-write layer=%s n_tail=%s dst_slot[0]=%s src_slot[0]=%s",
                            layer_name,
                            int(dst_slots.numel()),
                            int(dst_slots[0].item()),
                            int(src_slots[0].item()),
                        )
        return result

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        num_decodes = int(getattr(attn_metadata, "num_decodes", 0) or 0)
        num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0) or 0)
        num_prefills = int(getattr(attn_metadata, "num_prefills", 0) or 0)
        manager = get_sparse_kv_offload_manager()
        layer_name = self._offload_layer_name()

        if _generalized_mtp_enabled():
            batch = attn_metadata.mtp_batch
            if batch is None:
                if not get_ascend_config().sparse_kv_offload_config.keep_device_kv_cache:
                    return self._generalized_pd_fallback(
                        ql_nope,
                        q_pe,
                        topk_indices,
                        attn_metadata,
                        manager,
                        layer_name,
                    )
                return super()._execute_sparse_flash_attention_process(
                    ql_nope,
                    q_pe,
                    kv_cache,
                    topk_indices,
                    attn_metadata,
                    actual_seq_lengths_query,
                    actual_seq_lengths_key,
                    block_table=block_table,
                )
            return self._pad_to_input_tokens(
                self._generalized_mtp_copy(ql_nope, q_pe, kv_cache, manager, layer_name, batch, attn_metadata),
                ql_nope.shape[0],
            )

        if num_decode_tokens == 0:
            # Pure prefill batch (colocate debug only).
            _check_device_kv_cache_exist()
            return super()._execute_sparse_flash_attention_process(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table=block_table,
            )

        if self._use_nano_fused_op():
            decode_attn_output = self._nano_fused_copy_sfa(
                ql_nope=ql_nope,
                q_pe=q_pe,
                attn_metadata=attn_metadata,
                actual_seq_lengths_key=actual_seq_lengths_key,
                manager=manager,
                layer_name=layer_name,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )
            if num_prefills == 0:
                return self._pad_to_input_tokens(decode_attn_output, ql_nope.shape[0])
            _check_device_kv_cache_exist()
            prefill_query_offset = actual_seq_lengths_query[num_decodes - 1]
            prefill_query_lens = actual_seq_lengths_query[num_decodes:] - prefill_query_offset
            prefill_block_table = attn_metadata.block_table[num_decodes : num_decodes + num_prefills]
            prefill_attn_output = super()._execute_sparse_flash_attention_process(
                ql_nope[num_decode_tokens:],
                q_pe[num_decode_tokens:],
                kv_cache,
                topk_indices[num_decode_tokens:],
                attn_metadata,
                prefill_query_lens,
                actual_seq_lengths_key[num_decodes:],
                block_table=prefill_block_table,
            )
            attn_output = torch.cat([decode_attn_output, prefill_attn_output], dim=0)
            return self._pad_to_input_tokens(attn_output, ql_nope.shape[0])

        if attn_metadata.req_ids_tensor is None or attn_metadata.token_to_req is None:
            raise RuntimeError("Sparse KV offload requires req_ids_tensor/token_to_req metadata")
        token_to_req = attn_metadata.token_to_req[:num_decode_tokens]
        row_to_req = token_to_req.to(dtype=torch.int64)
        decode_seq_lens = torch.index_select(
            actual_seq_lengths_key[:num_decodes],
            0,
            row_to_req,
        )
        decode_cum_query_lens = actual_seq_lengths_query[:num_decodes]
        decode_query_lens = decode_cum_query_lens.clone()
        if num_decodes > 1:
            decode_query_lens[1:] -= decode_cum_query_lens[:-1]
        # Only the query span can be rewritten by the next MTP step.
        stable_prefix_lens = (actual_seq_lengths_key[:num_decodes] - decode_query_lens).clamp_min_(0)
        decode_stable_prefix_lens = torch.index_select(
            stable_prefix_lens,
            0,
            row_to_req,
        )
        decode_topk = topk_indices[:num_decode_tokens]
        seq_len_thresholds = decode_seq_lens.view(
            decode_seq_lens.shape[0],
            *([1] * (decode_topk.ndim - 1)),
        )
        valid_topk = build_valid_topk_mask(decode_topk, seq_len_thresholds)
        decode_topk = torch.where(
            valid_topk,
            decode_topk,
            torch.full_like(decode_topk, -1),
        )
        if decode_topk.ndim == 3 and decode_topk.shape[1] == 1:
            decode_topk = decode_topk.squeeze(1)
        if decode_topk.ndim != 2:
            raise ValueError("Sparse KV offload top-k must have [tokens, topk] shape")

        (
            resident_k,
            resident_v,
            resident_slot_indices,
            resident_block_table,
            resident_query_lens,
            resident_seq_lens,
        ) = self._resident_views(manager, layer_name, num_decode_tokens)
        decode_req_ids = torch.index_select(
            attn_metadata.req_ids_tensor[:num_decodes],
            0,
            row_to_req,
        )
        manager.onload_topk_kv(
            layer_name,
            num_decode_tokens,
            num_decodes,
            attn_metadata.block_table[:num_decodes],
            decode_topk,
            resident_slot_indices,
            decode_req_ids,
            decode_stable_prefix_lens,
            token_to_req,
            capturing=self._in_graph_runtime(),
            skip_topk=self.skip_topk,
        )
        decode_attn_output = DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope[:num_decode_tokens],
            q_pe[:num_decode_tokens],
            (resident_k, resident_v),
            resident_slot_indices.unsqueeze(1),
            attn_metadata,
            resident_query_lens,
            resident_seq_lens,
            block_table=resident_block_table,
        )
        if num_prefills == 0:
            return self._pad_to_input_tokens(decode_attn_output, ql_nope.shape[0])

        # Mixed batch (colocate debug only): prefill rows still attend the NPU
        # paged cache. The cumulative query lengths are rebased to the first
        # prefill request.
        _check_device_kv_cache_exist()
        prefill_query_offset = actual_seq_lengths_query[num_decodes - 1]
        prefill_query_lens = actual_seq_lengths_query[num_decodes:] - prefill_query_offset
        prefill_block_table = attn_metadata.block_table[num_decodes : num_decodes + num_prefills]
        prefill_attn_output = super()._execute_sparse_flash_attention_process(
            ql_nope[num_decode_tokens:],
            q_pe[num_decode_tokens:],
            kv_cache,
            topk_indices[num_decode_tokens:],
            attn_metadata,
            prefill_query_lens,
            actual_seq_lengths_key[num_decodes:],
            block_table=prefill_block_table,
        )
        attn_output = torch.cat([decode_attn_output, prefill_attn_output], dim=0)
        return self._pad_to_input_tokens(attn_output, ql_nope.shape[0])

    def _generalized_pd_fallback(self, q, q_rope, topk_indices, metadata, manager, layer_name):
        if self._in_graph_runtime():
            raise RuntimeError("Generalized PD short-query attention must remain eager")
        if not self._is_decode_only(metadata):
            _check_device_kv_cache_exist()
        requests, tokens = metadata.num_decodes, metadata.num_decode_tokens
        ends = metadata.cum_query_lens[:requests].to(torch.int64)
        widths = torch.diff(ends, prepend=ends.new_zeros(1))
        token_to_request = torch.repeat_interleave(
            torch.arange(requests, device=q.device),
            widths,
            output_size=tokens,
        )
        visible = metadata.seq_lens[:requests].index_select(0, token_to_request) - (
            ends.index_select(0, token_to_request) - torch.arange(tokens, device=q.device) - 1
        )
        selected = topk_indices[:tokens].reshape(tokens, -1)
        if selected.shape[1] != NANO_FUSED_TOPK:
            raise ValueError("Generalized PD fallback requires 2048 TopK entries per query")
        # The fallback uses query rows rather than persistent request rows.
        # Invalidate once per fallback batch in the builder before any layer
        # overwrites the hot arena; the next fused batch must first-fill.
        slots, table = manager.gather_nano_fallback(
            layer_name,
            metadata.block_table[:requests],
            selected,
            token_to_request,
            visible,
        )
        hbm_k, hbm_v = manager.hbm_kv_pair_for_fused(layer_name)
        output = DeviceOperator.execute_sparse_flash_attention_process(
            self,
            q[:tokens],
            q_rope[:tokens],
            (hbm_k, hbm_v),
            slots.unsqueeze(1),
            metadata,
            torch.arange(1, tokens + 1, dtype=torch.int32, device=q.device),
            torch.full((tokens,), NANO_FUSED_TOPK, dtype=torch.int32, device=q.device),
            block_table=table,
        )
        return self._pad_to_input_tokens(output, q.shape[0])

    def _generalized_mtp_copy(self, q, q_rope, kv_cache, manager, layer_name, batch, metadata):
        if self._in_graph_runtime() and batch.graph_buffers is None:
            raise RuntimeError("Generalized MTP graph inputs were not prepared before capture")
        runtime = _mtp_runtime(manager)
        src, dst, topk_misses, miss_src, miss_dst, misses = runtime.copy_metadata(batch)
        hbm_kv, hbm_rope = manager.hbm_kv_pair_for_fused(layer_name)
        dram_kv, dram_rope = manager.dram_kv_pair_for_fused(layer_name)
        # RD2H populates the CPU main pool, not nano's rank-local HBM tails.
        # Restore only [L,S), after the current query's D2H, on every replay.
        manager.restore_nano_tail(layer_name, batch)
        query_heads = q.shape[1]
        query, query_rope = prepare_copy_sfa_queries(
            q[: batch.num_tokens],
            q_rope[: batch.num_tokens],
        )
        out = manager.get_fused_attention_out(query)
        if batch.graph_buffers is None and manager.nano_debug_enabled() and not manager._nano_debug_decode_logged:
            logger.info(
                "generalized MTP copy-SFA inputs: layer=%s query_shape=%s query_ends=%s "
                "pool_rows=%s hbm_shape=%s dram_shape=%s",
                layer_name,
                tuple(query.shape),
                batch.query_ends_cpu,
                batch.pool_rows,
                tuple(hbm_kv.shape),
                tuple(dram_kv.shape),
            )
        torch.ops._C_ascend.npu_fused_copy_sfa_mtp(
            query_rope,
            query,
            batch.query_ends,
            batch.cache_tokens + batch.seq_lens - batch.offload_lens,
            batch.cache_tokens,
            dst,
            src,
            topk_misses,
            miss_src,
            miss_dst,
            misses,
            batch.hbm_block_table,
            batch.source_block_table,
            hbm_rope,
            hbm_kv,
            dram_rope,
            dram_kv,
            float(self.scale),
            out,
        )
        if batch.graph_buffers is None and manager.nano_debug_enabled() and not manager._nano_debug_decode_logged:
            manager._nano_debug_decode_logged = True
            logger.info(
                "generalized MTP copy-SFA submitted: layer=%s T=%s B=%s DRAM_device=%s query_heads=%s kernel_heads=%s",
                layer_name,
                batch.num_tokens,
                len(batch.pool_rows),
                dram_kv.device,
                query_heads,
                query.shape[1],
            )
        return out[:, :query_heads].contiguous()

    def _nano_fused_copy_sfa(
        self,
        *,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: M,
        actual_seq_lengths_key: torch.Tensor,
        manager,
        layer_name: str,
        num_decodes: int,
        num_decode_tokens: int,
    ) -> torch.Tensor:
        if num_decode_tokens != num_decodes:
            raise NotImplementedError(
                "nano fused_copy_sfa does not support MTP multi-token decode yet "
                f"(num_decode_tokens={num_decode_tokens}, num_decodes={num_decodes})"
            )
        if attn_metadata.device_block_table is None:
            raise RuntimeError("nano fused_copy_sfa requires device_block_table")
        if attn_metadata.offload_seq_lengths_key is None:
            raise RuntimeError("nano fused_copy_sfa requires offload_seq_lengths_key")

        # skip_topk shared layers never enter LIM; still need per-layer H2D fill.
        self._maybe_nano_prefix_init(attn_metadata, manager, layer_name, actual_seq_lengths_key)

        if self.skip_topk:
            topk_src_ids, topk_dst_slots, miss_counts = manager.require_lim_outputs(num_decodes)
        else:
            topk_src_ids, topk_dst_slots, miss_counts = manager.get_lim_output_buffers(num_decodes)
            if manager._last_lim_batch_size < num_decodes:
                raise RuntimeError(f"nano fused_copy_sfa expected fused_li_manage outputs for batch_size={num_decodes}")

        q = ql_nope[:num_decode_tokens]
        q_rope = q_pe[:num_decode_tokens]
        if q.ndim != 3 or q.size(-1) != self.kv_lora_rank:
            raise RuntimeError(f"nano fused_copy_sfa expects ql_nope [B,N,{self.kv_lora_rank}], got {tuple(q.shape)}")
        if q_rope.ndim != 3 or q_rope.size(-1) != self.qk_rope_head_dim:
            raise RuntimeError(
                f"nano fused_copy_sfa expects q_pe [B,N,{self.qk_rope_head_dim}], got {tuple(q_rope.shape)}"
            )

        # HBM: buffer_k = CKV (kv_lora), buffer_v = KPE (rope)
        hbm_kv_cache, hbm_k_rope = manager.hbm_kv_pair_for_fused(layer_name)
        # DRAM: host-side offload pool (k_cpu = CKV, v_cpu = KPE)
        dram_kv_cache, dram_k_rope = manager.dram_kv_pair_for_fused(layer_name)

        offload_lens = attn_metadata.offload_seq_lengths_key[:num_decodes].to(dtype=torch.int32)
        seq_lens = actual_seq_lengths_key[:num_decodes].to(dtype=torch.int32)
        tail_lens = (seq_lens - offload_lens).clamp_min(0)
        # No LIM candidates => dense-tail only (num_cache_tokens=0). Metadata
        # places dense physical blocks at the front of device_block_table for
        # those rows so the kernel's tailSlotStart=0 maps to real KV.
        num_cache_tokens = torch.where(
            offload_lens > 0,
            manager.lim_num_cache_tokens[:num_decodes],
            torch.zeros_like(offload_lens),
        )
        actual_seq_lengths_kv = num_cache_tokens + tail_lens
        actual_seq_lengths_query = manager.lim_query_lens[:num_decodes]

        if manager.nano_debug_enabled() and not manager._nano_debug_decode_logged:
            manager._nano_debug_decode_logged = True
            miss_cpu = miss_counts.detach().to(device="cpu")
            offload_cpu = offload_lens.detach().to(device="cpu")
            seq_cpu = seq_lens.detach().to(device="cpu")
            tail_cpu = tail_lens.detach().to(device="cpu")
            cache_cpu = num_cache_tokens.detach().to(device="cpu")
            logger.info(
                "nano debug fused_copy layer=%s skip_topk=%s prefix_init=%s "
                "offload=%s seq=%s tail=%s C=%s miss=%s src0=%s dst0=%s",
                layer_name,
                self.skip_topk,
                manager.has_nano_init_step(),
                offload_cpu.tolist(),
                seq_cpu.tolist(),
                tail_cpu.tolist(),
                cache_cpu.tolist(),
                miss_cpu.tolist(),
                topk_src_ids[0, 0, :8].detach().to(device="cpu").tolist(),
                topk_dst_slots[0, 0, :8].detach().to(device="cpu").tolist(),
            )

        attention_out = manager.get_fused_attention_out(q)
        torch.ops._C_ascend.npu_fused_copy_sfa(
            q_rope.contiguous(),
            q.contiguous(),
            actual_seq_lengths_query.contiguous(),
            actual_seq_lengths_kv.contiguous(),
            num_cache_tokens.contiguous(),
            topk_dst_slots,
            topk_src_ids.view(num_decodes, NANO_FUSED_TOPK),
            miss_counts,
            attn_metadata.device_block_table[:num_decodes].contiguous(),
            attn_metadata.block_table[:num_decodes].contiguous(),
            hbm_k_rope,
            hbm_kv_cache,
            dram_k_rope,
            dram_kv_cache,
            float(self.scale),
            attention_out,
        )
        return attention_out
