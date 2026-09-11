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
