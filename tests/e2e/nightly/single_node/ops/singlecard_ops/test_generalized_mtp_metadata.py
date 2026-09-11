# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU checks for serving metadata, cache ownership, and shared miss buffers."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp import (
    GeneralizedMtpRuntime,
    make_mtp_batch,
    prepare_copy_sfa_queries,
)


def fixture():
    device = "npu:0"
    manager = SimpleNamespace(
        block_size=128,
        topk_buffer_size=8192,
        max_num_reqs=3,
        topk_buffer_slot_manager=SimpleNamespace(req2slot={"request-a": 2, "request-b": 0}),
        nano_mtp_slot_generations={2: 1, 0: 1},
    )
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=2,
        cum_query_lens=torch.tensor([1, 5], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([8450, 8195], dtype=torch.int32, device=device),
        req_topk_buffer_slots=torch.tensor([2, 0], dtype=torch.int32, device=device),
        block_table=torch.arange(256, dtype=torch.int32, device=device).reshape(2, 128),
    )
    return manager, metadata


def test_causal_prefix_variable_budget_and_tail_rollover():
    manager, metadata = fixture()
    batch = make_mtp_batch(metadata, manager)
    assert batch.prefix_lengths == [8448, 8064]
    assert batch.cache_sizes == [8192, 8064]
    assert batch.num_tokens == 5
    # Request B crosses a block boundary with Q=4, so its 131-token tail
    # occupies both circular blocks after the physically fixed hot arena.
    assert batch.tail_sources.cpu().tolist() == list(range(8448, 8450)) + list(range(16384 + 8064, 16384 + 8195))
    stride = 8192 + 256
    expected = [2 * stride + 8192 + p % 256 for p in range(8448, 8450)]
    expected += [8192 + p % 256 for p in range(8064, 8195)]
    assert batch.tail_destinations.cpu().tolist() == expected
    tables = batch.hbm_block_table.cpu()
    assert tables[0, 64:66].tolist() == [196, 197]
    assert tables[1, 63:65].tolist() == [65, 64]
    metadata.seq_lens[1] = 1000
    assert make_mtp_batch(metadata, manager) is None


def test_layer_ownership_request_reuse_and_shared_miss_buffers():
    manager, metadata = fixture()
    batch = make_mtp_batch(metadata, manager)
    runtime = GeneralizedMtpRuntime(manager)
    map_a, state, outputs = runtime.prepare_lim("owner-a", batch, 16384, metadata.seq_lens.device)
    assert state.cpu().tolist() == [-2, -2]
    map_a[2, 17] = 23
    map_b, state_b, _ = runtime.prepare_lim("owner-b", batch, 16384, metadata.seq_lens.device)
    assert state_b.cpu().tolist() == [-2, -2]
    assert map_b[2, 17].item() == -(1 << 31)
    _, steady, outputs = runtime.prepare_lim("owner-a", batch, 16384, metadata.seq_lens.device)
    assert steady.cpu().tolist() == [-1, -1]
    outputs[3].fill_(17)
    outputs[4].fill_(23)
    shared = runtime.copy_metadata(batch)
    assert shared is outputs
    for i in (3, 4):
        assert shared[i].shape == (2, 32768) and shared[i].is_contiguous()
        assert shared[i].data_ptr() == outputs[i].data_ptr()
        torch.testing.assert_close(shared[i], outputs[i], rtol=0, atol=0)
    # Even an identical external request ID must first-fill after row reuse.
    manager.nano_mtp_slot_generations[2] += 1
    _, reused, _ = runtime.prepare_lim("owner-a", batch, 16384, metadata.seq_lens.device)
    assert reused.cpu().tolist() == [-2, -1]
    runtime.invalidate()
    _, invalidated, _ = runtime.prepare_lim("owner-a", batch, 16384, metadata.seq_lens.device)
    assert invalidated.cpu().tolist() == [-2, -2]


def test_batch_metadata_survives_builder_buffer_reuse():
    manager, metadata = fixture()
    batch = make_mtp_batch(metadata, manager)
    expected_blocks = metadata.block_table.clone()
    metadata.cum_query_lens.copy_(torch.tensor([1, 2], dtype=torch.int32, device="npu:0"))
    metadata.seq_lens.zero_()
    metadata.req_topk_buffer_slots.fill_(1)
    metadata.block_table.zero_()

    # The queued batch remains internally consistent after the next build
    # repurposes every source buffer. In particular, final prefix still T.
    assert batch.query_ends.cpu().tolist() == [1, 5]
    assert batch.num_tokens == 5
    assert batch.seq_lens.cpu().tolist() == [8450, 8195]
    assert batch.pool_entries.cpu().tolist() == [2, 0]
    torch.testing.assert_close(batch.source_block_table, expected_blocks, rtol=0, atol=0)


def test_host_snapshots_remove_readbacks_and_survive_reuse(monkeypatch):
    manager, metadata = fixture()
    ends = metadata.cum_query_lens.cpu()
    lengths = metadata.seq_lens.cpu()
    pools = metadata.req_topk_buffer_slots.cpu()
    original_cpu = torch.Tensor.cpu

    def reject_device_readback(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu", "unexpected MTP metadata device readback"
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", reject_device_readback)
        batch = make_mtp_batch(
            metadata, manager, query_ends_cpu=ends, seq_lens_cpu=lengths, pool_rows_cpu=pools
        )
    ends.zero_()
    lengths.zero_()
    pools.fill_(-1)
    assert batch.query_ends_cpu == (1, 5)
    assert batch.seq_lens_cpu == (8450, 8195)
    assert batch.pool_rows == [2, 0]
    assert batch.prefix_lengths == [8448, 8064]
    torch.testing.assert_close(batch.seq_lens, metadata.seq_lens)


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
def test_native_head_queries_preserve_contiguous_storage(heads):
    query = torch.randn(5, heads, 512, dtype=torch.bfloat16, device="npu:0")
    rope = torch.randn(5, heads, 64, dtype=torch.bfloat16, device="npu:0")
    prepared, prepared_rope = prepare_copy_sfa_queries(query, rope)
    assert prepared.data_ptr() == query.data_ptr()
    assert prepared_rope.data_ptr() == rope.data_ptr()


@pytest.mark.parametrize("heads", [0, 9, 12, 24, 256])
def test_unsupported_native_head_counts_are_rejected(heads):
    query = torch.empty(5, heads, 512, dtype=torch.bfloat16, device="npu:0")
    rope = torch.empty(5, heads, 64, dtype=torch.bfloat16, device="npu:0")
    with pytest.raises(ValueError, match="query heads per rank"):
        prepare_copy_sfa_queries(query, rope)


@pytest.mark.parametrize("with_offload", [False, True])
def test_unpadded_metadata_preserves_offload_pool_ownership(with_offload):
    manager, metadata = fixture()
    query_start_loc_cpu = torch.tensor([0, 1, 5, 16], dtype=torch.int32)
    common = AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.to("npu:0"),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.cat((metadata.seq_lens, metadata.seq_lens.new_zeros(1))),
        num_reqs=3,
        num_actual_tokens=16,
        num_input_tokens=16,
        max_query_len=11,
        max_seq_len=8450,
        block_table_tensor=torch.cat(
            (
                metadata.block_table,
                metadata.block_table.new_zeros((1, metadata.block_table.shape[1])),
            )
        ),
        slot_mapping=torch.arange(16, dtype=torch.int64, device="npu:0"),
        req_topk_buffer_slots=(torch.tensor([2, 0, -1], dtype=torch.int32, device="npu:0") if with_offload else None),
        req_topk_buffer_slots_cpu=(torch.tensor([2, 0, -1], dtype=torch.int32) if with_offload else None),
    )
    unpadded = common.unpadded(num_actual_tokens=5, num_actual_reqs=2)
    if not with_offload:
        assert unpadded.req_topk_buffer_slots is None
        return

    # Nontrivial pool order must survive token/request padding removal;
    # the padded -1 row must never reach the draft's generalized LIM batch.
    assert unpadded.req_topk_buffer_slots.cpu().tolist() == [2, 0]
    assert unpadded.req_topk_buffer_slots_cpu.tolist() == [2, 0]
    metadata.cum_query_lens = unpadded.query_start_loc[1:]
    metadata.seq_lens = unpadded.seq_lens
    metadata.block_table = unpadded.block_table_tensor
    metadata.req_topk_buffer_slots = unpadded.req_topk_buffer_slots
    batch = make_mtp_batch(metadata, manager)
    assert batch.pool_rows == [2, 0]
    assert batch.num_tokens == 5
