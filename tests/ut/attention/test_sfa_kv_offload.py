"""Regression tests for SFA KV-offload attention metadata."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.attention.attention_v1 import AscendAttentionState  # noqa: E402
from vllm_ascend.attention.sfa_kv_offload import (  # noqa: E402
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadata,
    AscendSFAKVOffloadMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder  # noqa: E402


@pytest.fixture(autouse=True)
def offload_config():
    # These boundary-classification tests construct a builder without the
    # model/config initialization performed by an actual worker.
    config = SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(
            generalized_mtp=False,
            fused_op_type="default",
        )
    )
    with patch("vllm_ascend.attention.sfa_kv_offload.get_ascend_config", return_value=config):
        yield config


def _make_boundary_decode_metadata():
    return SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=1,
        num_reqs=1,
        num_actual_tokens=1,
        query_start_loc_cpu=torch.tensor([0, 1]),
        is_prefilling=torch.tensor([True]),
        req_ids_tensor=torch.tensor([7]),
        token_to_req=torch.tensor([0]),
    )


@pytest.mark.parametrize(
    ("kv_transfer_config", "expected"),
    [
        (None, False),
        (SimpleNamespace(is_kv_consumer=False, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=False), True),
    ],
)
def test_pd_decode_consumer_is_derived_from_kv_role(kv_transfer_config, expected):
    vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    with patch.object(AscendSFAMetadataBuilder, "__init__", return_value=None):
        builder = AscendSFAKVOffloadMetadataBuilder(
            kv_cache_spec=None,
            layer_names=[],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

    assert builder.is_pd_decode_consumer is expected


@pytest.mark.parametrize(
    ("is_pd_decode_consumer", "expected_decodes", "expected_prefills"),
    [
        (True, 1, 0),
        (False, 0, 1),
    ],
)
def test_boundary_token_classification_depends_on_pd_decode_role(
    is_pd_decode_consumer,
    expected_decodes,
    expected_prefills,
):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = is_pd_decode_consumer
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, _make_boundary_decode_metadata())

    assert metadata.num_decodes == expected_decodes
    assert metadata.num_prefills == expected_prefills
    assert metadata.num_decode_tokens == expected_decodes
    assert metadata.req_ids_tensor.tolist() == [7]
    assert metadata.token_to_req.tolist() == [0]
    assert AscendSFAKVOffloadImpl._is_decode_only(metadata) is is_pd_decode_consumer


def test_pd_decode_consumer_still_rejects_long_prefill_classification():
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = True
    metadata = SimpleNamespace()
    common_metadata = _make_boundary_decode_metadata()
    common_metadata.max_query_len = 2
    common_metadata.num_actual_tokens = 2
    common_metadata.query_start_loc_cpu = torch.tensor([0, 2])

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, common_metadata)

    assert metadata.num_decodes == 0
    assert metadata.num_prefills == 1
    assert metadata.num_decode_tokens == 0


@pytest.mark.parametrize("widths", [(4, 4), (4, 1)])
def test_generalized_fallback_lengths_survive_later_draft_builds(offload_config, widths):
    """Preparing later Q1 drafts must not change an earlier fallback step."""
    from vllm_ascend.attention.sfa_v1 import DeviceOperator

    @dataclass
    class Metadata:
        cum_query_lens: torch.Tensor
        seq_lens: torch.Tensor
        block_table: torch.Tensor
        attn_state: AscendAttentionState = AscendAttentionState.SpecDecoding

    offload_config.sparse_kv_offload_config.generalized_mtp = True
    offload_config.sparse_kv_offload_config.keep_device_kv_cache = False
    tokens = sum(widths)
    shared_ends = torch.tensor([widths[0], tokens], dtype=torch.int32)
    shared_lengths = torch.tensor([10575, 22], dtype=torch.int32)
    table = torch.zeros((2, 128), dtype=torch.int32)
    metadata = Metadata(shared_ends, shared_lengths, table)
    starts = torch.tensor([0, widths[0], tokens], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=tokens,
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        seq_lens=shared_lengths,
        block_table_tensor=table,
        req_ids_tensor=torch.tensor([7, 8]),
        token_to_req=None,
        req_topk_buffer_slots=torch.tensor([0, 1]),
    )
    common.unpadded = lambda *_: common
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.decode_threshold = 4
    builder.is_pd_decode_consumer = True
    module = "vllm_ascend.attention.sfa_kv_offload."
    with (
        patch(module + "split_decodes_and_prefills", return_value=(2, 0, tokens, 0)),
        patch(module + "get_sparse_kv_offload_manager"),
        patch(module + "_mtp_runtime"),
        patch(module + "make_mtp_batch", return_value=None),
    ):
        builder._populate_offload_metadata(metadata, common)

    # The proposer prepares Q1 step metadata before executing the first step.
    shared_ends.copy_(torch.tensor([1, 2], dtype=torch.int32))
    shared_lengths.add_(2)
    assert metadata.cum_query_lens.tolist() == [widths[0], tokens]
    assert metadata.seq_lens.tolist() == [10575, 22]

    def gather(layer_name, block_table, selected, token_to_request, visible):
        assert token_to_request.tolist() == [0] * widths[0] + [1] * widths[1]
        assert visible.tolist() == (list(range(10576 - widths[0], 10576)) + list(range(23 - widths[1], 23)))
        return torch.zeros((tokens, 2048), dtype=torch.int32), table

    manager = SimpleNamespace(gather_nano_fallback=gather, hbm_kv_pair_for_fused=lambda _: (None, None))
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    query = torch.zeros((tokens, 4, 512))
    with (
        patch.object(impl, "_in_graph_runtime", return_value=False),
        patch.object(DeviceOperator, "execute_sparse_flash_attention_process", return_value=query),
    ):
        output = impl._generalized_pd_fallback(
            query,
            torch.zeros((tokens, 4, 64)),
            torch.zeros((tokens, 1, 2048), dtype=torch.int32),
            metadata,
            manager,
            "test.layer",
        )
    assert output.shape == query.shape


@pytest.mark.parametrize("is_dummy", [False, True])
def test_only_explicit_runtime_dummy_gets_synthetic_graph_metadata(offload_config, is_dummy):
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp_graph import eligible_graph_steps

    offload_config.sparse_kv_offload_config.generalized_mtp = True
    lengths = torch.tensor([4], dtype=torch.int32)
    starts = torch.tensor([0, 4], dtype=torch.int32)
    table = torch.zeros((1, 128), dtype=torch.int32)
    metadata = AscendSFAKVOffloadMetadata(
        num_actual_tokens=4,
        num_input_tokens=4,
        slot_mapping=torch.arange(4),
        seq_lens=lengths,
        seq_lens_cpu=lengths,
        cum_query_lens=starts[1:],
        block_table=table,
        sin=torch.zeros(4, 64),
        cos=torch.ones(4, 64),
    )
    common = SimpleNamespace(
        num_reqs=1, num_actual_tokens=4, query_start_loc_cpu=starts, query_start_loc=starts,
        seq_lens=lengths, _seq_lens_cpu=lengths, block_table_tensor=table,
        req_topk_buffer_slots=torch.tensor([-1 if is_dummy else 0], dtype=torch.int32),
        req_topk_buffer_slots_cpu=torch.tensor([-1 if is_dummy else 0], dtype=torch.int32),
        req_ids_tensor=None, token_to_req=None, offload_is_dummy=is_dummy,
    )
    common.unpadded = lambda *_: common
    manager = SimpleNamespace(block_size=128, topk_buffer_size=8192, max_num_reqs=1)
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.decode_threshold = 4
    builder.is_pd_decode_consumer = True
    builder._mtp_capture_width = 4
    module = "vllm_ascend.attention.sfa_kv_offload."
    with (
        patch(module + "split_decodes_and_prefills", return_value=(1, 0, 4, 0)),
        patch(module + "get_sparse_kv_offload_manager", return_value=manager),
    ):
        builder._populate_offload_metadata(metadata, common)
    assert metadata.mtp_graph_capture is False
    assert metadata.mtp_graph_dummy is is_dummy
    assert eligible_graph_steps([{"owner": metadata}]) is is_dummy
    if is_dummy:
        assert metadata.mtp_batch.seq_lens_cpu == (8196,)
        assert metadata.slot_mapping.tolist() == [-1] * 4
    else:
        assert metadata.mtp_batch is None
# ---------------------------------------------------------------------------
# Sparse LI C8 dispatch contract in _nano_fused_li_manage (MTP path):
# CPU-mockable tests for the guard, op selection, and int8/fp16 tensor
# preparation. NPU-side top-k precision lives in tests/ut/ops.
# ---------------------------------------------------------------------------

_LIM_N_HEAD = 32
_LIM_HEAD_DIM = 128
_LIM_NUM_DECODES = 2
_LIM_TOKENS_PER_REQ = 7
_LIM_T = _LIM_NUM_DECODES * _LIM_TOKENS_PER_REQ
_LIM_BLOCKS = 8


def _lim_impl(*, enable_c8: bool):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.has_indexer = True
    impl.enable_sparse_li_c8 = enable_c8
    impl.enable_sparse_sfa_c8 = False  # default layout: indexer_k=2, scale=3
    impl.n_head = _LIM_N_HEAD
    impl.head_dim = _LIM_HEAD_DIM
    impl.qk_rope_head_dim = 64
    impl.is_rope_neox_style = False
    impl.c8_k_cache_dtype = torch.int8
    impl.c8_k_scale_cache_dtype = torch.float16
    impl.layer_name = "test.layer"
    impl._offload_layer_name = lambda: "test.layer"
    return impl


def _lim_mtp_metadata():
    block_table = torch.zeros((_LIM_NUM_DECODES, _LIM_BLOCKS), dtype=torch.int32)
    mtp_batch = SimpleNamespace(
        num_tokens=_LIM_T,
        source_block_table=block_table,
        query_ends=torch.tensor([_LIM_TOKENS_PER_REQ, _LIM_T], dtype=torch.int32),
        seq_lens=torch.full((_LIM_NUM_DECODES,), 8192, dtype=torch.int32),
        offload_lens=torch.full((_LIM_NUM_DECODES,), 4096, dtype=torch.int32),
        cache_tokens=torch.full((_LIM_NUM_DECODES,), 2048, dtype=torch.int32),
        pool_entries=torch.arange(_LIM_NUM_DECODES, dtype=torch.int32),
        prefix_lengths=[4096] * _LIM_NUM_DECODES,
        cache_sizes=[2048] * _LIM_NUM_DECODES,
        graph_buffers=None,
    )
    return SimpleNamespace(
        num_decodes=_LIM_NUM_DECODES,
        num_decode_tokens=_LIM_T,
        num_prefills=0,
        mtp_batch=mtp_batch,
        block_table=block_table,
        req_topk_buffer_slots=torch.arange(_LIM_NUM_DECODES, dtype=torch.int32),
        indexer_block_table=None,
    )


def _lim_kv_cache():
    index_key = torch.zeros(
        (_LIM_BLOCKS, _LIM_HEAD_DIM, 1, _LIM_HEAD_DIM), dtype=torch.int8
    )
    index_key_scale = torch.zeros(
        (_LIM_BLOCKS, _LIM_HEAD_DIM, 1, 1), dtype=torch.float16
    )
    return (None, None, index_key, index_key_scale)


def _run_lim_dispatch(*, enable_c8: bool):
    """Drive _nano_fused_li_manage end-to-end with mocked deps; return records."""
    import torch_npu
    from contextlib import ExitStack
    from unittest.mock import MagicMock

    from vllm_ascend.attention import sfa_kv_offload as mod
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    impl = _lim_impl(enable_c8=enable_c8)
    metadata = _lim_mtp_metadata()
    kv_cache = _lim_kv_cache()

    torch.manual_seed(0)
    q_li_flat = torch.randn((_LIM_T, _LIM_N_HEAD * _LIM_HEAD_DIM), dtype=torch.bfloat16)
    kw = torch.randn((_LIM_T, _LIM_HEAD_DIM + _LIM_N_HEAD), dtype=torch.bfloat16)
    impl.wq_b = MagicMock(return_value=(q_li_flat, None))
    impl.wk_weights_proj = MagicMock(return_value=(kw, None))
    q_li_raw = q_li_flat.view(_LIM_T, _LIM_N_HEAD, _LIM_HEAD_DIM)

    # Non-identity orthogonal stand-in for the shared Hadamard matrix: a
    # column-reversing permutation. Exact (no float error), so a forgotten
    # rotation (input == raw) is distinguishable from an applied one.
    hadamard = torch.zeros((_LIM_HEAD_DIM, _LIM_HEAD_DIM), dtype=torch.bfloat16)
    for i in range(_LIM_HEAD_DIM):
        hadamard[i, _LIM_HEAD_DIM - 1 - i] = 1.0

    m_c8 = MagicMock()
    m_mtp = MagicMock()
    quant_inputs = []

    def fake_quant(inp, dst_type=None):
        quant_inputs.append(inp.detach().clone())
        return inp.to(torch.int8), torch.ones(inp.shape[0], dtype=torch.float32)

    manager = SimpleNamespace(
        block_size=_LIM_HEAD_DIM, nano_debug_enabled=lambda: False
    )
    runtime = MagicMock()
    mapping = torch.zeros(
        (_LIM_NUM_DECODES, _LIM_BLOCKS * _LIM_HEAD_DIM), dtype=torch.int32
    )
    states = torch.zeros((_LIM_NUM_DECODES,), dtype=torch.int32)
    outputs = (
        torch.zeros((_LIM_T, 1, 2048), dtype=torch.int32),
        torch.zeros((_LIM_T, 1, 2048), dtype=torch.int32),
        torch.zeros((_LIM_T,), dtype=torch.int32),
        torch.zeros((_LIM_NUM_DECODES, 32768), dtype=torch.int32),
        torch.zeros((_LIM_NUM_DECODES, 32768), dtype=torch.int32),
        torch.zeros((_LIM_NUM_DECODES,), dtype=torch.int32),
    )
    runtime.prepare_lim.return_value = (mapping, states, outputs)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(torch.ops._C_ascend, "npu_fused_li_manage_mtp_c8", m_c8, create=True)
        )
        stack.enter_context(
            patch.object(torch.ops._C_ascend, "npu_fused_li_manage_mtp", m_mtp, create=True)
        )
        stack.enter_context(patch.object(torch_npu, "npu_dynamic_quant", fake_quant))
        stack.enter_context(patch.object(AscendSFAImpl, "q_hadamard", hadamard))
        stack.enter_context(patch.object(mod, "HAS_TRITON", True))
        stack.enter_context(patch.object(mod, "rope_forward_triton_siso", lambda *a, **k: a[0]))
        stack.enter_context(patch.object(mod, "get_sparse_kv_offload_manager", return_value=manager))
        stack.enter_context(patch.object(mod, "_mtp_runtime", return_value=runtime))
        result = impl._nano_fused_li_manage(
            torch.zeros((_LIM_T, 512), dtype=torch.bfloat16),
            torch.zeros((_LIM_T, 512), dtype=torch.bfloat16),
            kv_cache,
            metadata,
            torch.zeros(64, dtype=torch.bfloat16),
            torch.zeros(64, dtype=torch.bfloat16),
            torch.zeros((_LIM_NUM_DECODES,), dtype=torch.int32),
        )

    return {
        "result": result,
        "m_c8": m_c8,
        "m_mtp": m_mtp,
        "quant_inputs": quant_inputs,
        "q_li_raw": q_li_raw,
        "hadamard": hadamard,
        "outputs": outputs,
    }


def test_li_c8_non_mtp_path_still_raises():
    impl = _lim_impl(enable_c8=True)
    metadata = SimpleNamespace(num_decodes=1, num_decode_tokens=1)
    with pytest.raises(NotImplementedError, match="non-MTP"):
        impl._nano_fused_li_manage(None, None, None, metadata, None, None, None)


def test_li_c8_mtp_dispatches_to_c8_op_with_int8_inputs():
    rec = _run_lim_dispatch(enable_c8=True)
    assert rec["m_c8"].called, "C8 path must call npu_fused_li_manage_mtp_c8"
    assert not rec["m_mtp"].called, "C8 path must not call the bf16 mtp op"

    args = rec["m_c8"].call_args.args
    weights, q_scale, query, key_scale, index_key_cache = args[:5]
    assert weights.dtype == torch.bfloat16, "index_weights must be bf16"
    assert q_scale.dtype == torch.float16, "query_dequant_scale must be fp16"
    assert tuple(q_scale.shape) == (_LIM_T, _LIM_N_HEAD)
    assert query.dtype == torch.int8, "query must be int8 after Hadamard+quant"
    assert tuple(query.shape) == (_LIM_T, _LIM_N_HEAD, _LIM_HEAD_DIM)
    assert key_scale.dtype == torch.float16, "index_key_dequant_scale must be fp16"
    assert key_scale.ndim == 3, "key dequant scale must be 3D [blocks,128,1]"
    assert tuple(key_scale.shape) == (_LIM_BLOCKS, _LIM_HEAD_DIM, 1)
    assert index_key_cache.dtype == torch.int8, "index_key_cache must be int8"

    # Hadamard was actually applied before quant (forgotten rotation would make
    # the quant input equal the raw q_li, which differs from q_li @ hadamard).
    assert rec["quant_inputs"], "npu_dynamic_quant must be called once"
    expected = (rec["q_li_raw"] @ rec["hadamard"]).reshape(-1, _LIM_HEAD_DIM)
    assert torch.equal(rec["quant_inputs"][0], expected), (
        "query must be Hadamard-rotated before int8 quant"
    )
    assert rec["result"] is rec["outputs"][0]


def test_li_non_c8_mtp_still_uses_bf16_mtp_op():
    rec = _run_lim_dispatch(enable_c8=False)
    assert rec["m_mtp"].called, "non-C8 path must call npu_fused_li_manage_mtp"
    assert not rec["m_c8"].called, "non-C8 path must not call the c8 op"
    assert not rec["quant_inputs"], "non-C8 path must not quantize the query"

    args = rec["m_mtp"].call_args.args
    # weights, placeholder query_scale, q_li, placeholder key_scale, index_key
    weights, query_scale, query, key_scale, _ = args[:5]
    assert query_scale.dtype == torch.float32, "non-C8 uses fp32 placeholder scale"
    assert key_scale.dtype == torch.float32, "non-C8 uses fp32 placeholder scale"
    assert query.dtype == torch.bfloat16, "non-C8 query stays bf16"
