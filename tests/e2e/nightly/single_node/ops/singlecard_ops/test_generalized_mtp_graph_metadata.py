# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent metadata and scratch ownership across NPU graph replays."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadMetadata
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp import (
    GeneralizedMtpRuntime,
    make_mtp_batch,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp_graph import (
    MtpGraphBuffers,
    MtpGraphMetadataSet,
    eligible_graph_steps,
    make_dummy_batch,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    allocate_kv_offload_topk_buffer_pair,
)


def fixture():
    device = torch.device("npu:0")
    manager = SimpleNamespace(
        block_size=128,
        topk_buffer_size=8192,
        max_num_reqs=3,
        max_num_topk_rows=12,
        topk_buffer_slot_manager=SimpleNamespace(req2slot={"a": 2, "b": 0}),
        nano_mtp_slot_generations={2: 1, 0: 1},
    )
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=2,
        cum_query_lens=torch.tensor([4, 8], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([8452, 8196], dtype=torch.int32, device=device),
        req_topk_buffer_slots=torch.tensor([2, 0], dtype=torch.int32, device=device),
        block_table=torch.arange(256, dtype=torch.int32, device=device).reshape(2, 128),
    )
    runtime = GeneralizedMtpRuntime(manager)
    buffers = MtpGraphBuffers(runtime, query_width=4, source_capacity=16384, device=device)
    return manager, metadata, buffers


def test_graph_padding_owns_private_rows_and_fixed_tail_storage():
    manager, metadata, buffers = fixture()
    batch = make_mtp_batch(metadata, manager)
    buffers.update(batch)
    assert buffers.batch.pool_rows == [2, 0, 5]
    assert buffers.batch.tail_sources.numel() == 3 * 256
    assert buffers.tail_valid.cpu().sum().item() == 8
    valid = buffers.tail_valid
    torch.testing.assert_close(buffers.batch.tail_sources[valid], batch.tail_sources)
    torch.testing.assert_close(buffers.batch.tail_destinations[valid], batch.tail_destinations)
    scratch_destinations = buffers.batch.tail_destinations[-256:]
    assert int(scratch_destinations.min().cpu()) >= manager.max_num_reqs * (8192 + 256)
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-2, -2, -3]
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-1, -1, -3]
    manager.nano_mtp_slot_generations[2] += 1
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-2, -1, -3]
    assert set(buffers.runtime.residents["owner"]) == {0, 2}


def test_graph_padding_becomes_a_new_live_request_after_slot_reuse():
    manager, metadata, buffers = fixture()
    buffers.update(make_mtp_batch(metadata, manager))
    buffers.prepare_layer("owner")
    metadata.num_decodes = 1
    buffers.update(make_mtp_batch(metadata, manager))
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-1, -3, -3]
    assert buffers.batch.pool_rows == [2, 4, 5]
    manager.topk_buffer_slot_manager.req2slot = {"a": 2, "new-b": 0}
    manager.nano_mtp_slot_generations[0] += 1
    metadata.num_decodes = 2
    buffers.update(make_mtp_batch(metadata, manager))
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-1, -2, -3]
    # An existing request also resets if its aligned prefix shrinks.
    metadata.seq_lens[0] -= manager.block_size
    buffers.update(make_mtp_batch(metadata, manager))
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-2, -1, -3]


def test_graph_capture_uses_only_private_nonoffload_rows():
    manager, metadata, buffers = fixture()
    batch = make_mtp_batch(metadata, manager)
    buffers.update(batch, is_capture=True)
    buffers.prepare_layer("owner")
    assert buffers.batch.pool_rows == [3, 4, 5]
    assert buffers.layers["owner"][1].cpu().tolist() == [-3, -3, -3]
    assert not buffers.active_mask.any().item()
    assert not buffers.tail_valid.any().item()
    assert buffers.runtime.residents["owner"] == {}
    # Capture must not mark live rows warm before the first actual decode.
    buffers.update(batch)
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-2, -2, -3]


def test_runtime_dummy_keeps_residency_and_graph_checks_on_cpu(monkeypatch):
    manager, metadata, buffers = fixture()
    real = make_mtp_batch(metadata, manager)
    buffers.update(real)
    buffers.prepare_layer("owner")
    residents = dict(buffers.runtime.residents["owner"])
    dummy_metadata = SimpleNamespace(
        num_input_tokens=12, seq_lens=metadata.seq_lens, block_table=metadata.block_table
    )
    original_cpu = torch.Tensor.cpu

    def reject_device_readback(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu", "graph metadata must not read device values"
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", reject_device_readback)
        dummy = make_dummy_batch(dummy_metadata, manager, 4)
        assert eligible_graph_steps([{"owner": SimpleNamespace(mtp_batch=dummy)}])
        assert buffers.accepts(dummy)
        buffers.update(dummy, is_dummy=True)
        buffers.prepare_layer("owner")
    assert buffers.batch.pool_rows == [3, 4, 5]
    assert buffers.active_requests == 0
    assert not buffers.active_mask.any().item()
    assert not buffers.tail_valid.any().item()
    assert buffers.layers["owner"][1].cpu().tolist() == [-3, -3, -3]
    assert buffers.runtime.residents["owner"] == residents
    buffers.update(real)
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-1, -1, -3]


def test_wrapper_retains_full_mode_for_runtime_dummy_and_real_transitions(monkeypatch):
    import vllm_ascend.compilation.acl_graph as graph_module

    manager, source, buffers = fixture()
    manager.generalized_mtp_runtime = buffers.runtime
    batch = make_mtp_batch(source, manager)
    device = source.seq_lens.device
    metadata = AscendSFAKVOffloadMetadata(
        num_actual_tokens=8, num_input_tokens=12,
        slot_mapping=torch.arange(12, dtype=torch.int32, device=device),
        seq_lens=source.seq_lens, seq_lens_cpu=source.seq_lens.cpu(),
        cum_query_lens=source.cum_query_lens, block_table=source.block_table,
        sin=torch.zeros(12, 64, device=device), cos=torch.ones(12, 64, device=device),
        num_decodes=2, num_decode_tokens=8, mtp_batch=batch,
    )
    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        attn_metadata={"owner": metadata}, batch_descriptor="test-batch",
    )
    monkeypatch.setattr(graph_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager.get_sparse_kv_offload_manager",
        lambda: manager,
    )
    wrapper = graph_module.ACLGraphWrapper.__new__(graph_module.ACLGraphWrapper)
    wrapper.generalized_mtp_offload = True
    wrapper.runtime_mode = CUDAGraphMode.FULL
    wrapper.concrete_aclgraph_entries = {}
    wrapper.runnable = Mock(side_effect=AssertionError("unexpected local eager fallback"))
    modes = []

    def graph_call():
        modes.append(context.cudagraph_runtime_mode)
        return context.attn_metadata["owner"].mtp_batch

    wrapper._call_graph = graph_call
    live = wrapper()
    pointer = live.seq_lens.data_ptr()
    metadata.mtp_batch = make_dummy_batch(metadata, manager, 4)
    metadata.mtp_graph_dummy = True
    dummy = wrapper()
    assert dummy.seq_lens.data_ptr() == pointer
    assert dummy.graph_buffers.active_requests == 0
    assert dummy.graph_buffers.layers["owner"][1].cpu().tolist() == [-3] * 3
    metadata.mtp_batch = batch
    metadata.mtp_graph_dummy = False
    metadata.slot_mapping.copy_(torch.arange(12, dtype=torch.int32, device=device))
    restored = wrapper()
    assert restored.seq_lens.data_ptr() == pointer
    assert restored.graph_buffers.layers["owner"][1].cpu().tolist() == [-1, -1, -3]
    assert modes == [CUDAGraphMode.FULL] * 3
    wrapper.runnable.assert_not_called()


def test_padding_cache_and_tail_are_initialized_without_touching_live_rows(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=12, max_num_seqs=3),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)),
        cache_config=SimpleNamespace(block_size=128),
    )
    offload_config = SimpleNamespace(topk_buffer_size=8192, fused_op_type="nano", generalized_mtp=True)
    original_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        # Raw int8 -1 creates BF16 NaNs, exposing uninitialized padding.
        return result.fill_(-1)

    monkeypatch.setattr(torch, "empty", poisoned_empty)
    for tensor in allocate_kv_offload_topk_buffer_pair(config, offload_config):
        assert tensor.shape[:2] == (12, 8192 + 256)
        assert tensor[:3].isnan().all().item()
        assert tensor[3:].count_nonzero().item() == 0


def test_graph_inputs_change_without_replacing_captured_storage():
    manager, metadata, buffers = fixture()
    buffers.update(make_mtp_batch(metadata, manager))
    watched = [
        buffers.batch.seq_lens,
        buffers.batch.pool_entries,
        buffers.batch.tail_sources,
        buffers.batch.tail_destinations,
        buffers.tail_valid,
    ]
    addresses = [tensor.data_ptr() for tensor in watched]
    snapshots = [torch.empty_like(tensor) for tensor in watched]
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        for snapshot, value in zip(snapshots, watched):
            snapshot.copy_(value)
    graph.replay()
    torch.npu.synchronize()
    assert snapshots[0].cpu().tolist() == [8452, 8196, 8196]
    # A smaller next batch changes pool ownership and crosses a tail block.
    metadata.num_decodes = 1
    metadata.seq_lens[0] = 8580
    buffers.update(make_mtp_batch(metadata, manager))
    graph.replay()
    torch.npu.synchronize()
    assert [tensor.data_ptr() for tensor in watched] == addresses
    assert snapshots[0].cpu().tolist() == [8580, 8196, 8196]
    assert snapshots[1].cpu().tolist() == [2, 4, 5]
    assert snapshots[-1].cpu().sum().item() == 4
    for snapshot, current in zip(snapshots, watched):
        torch.testing.assert_close(snapshot, current)


def test_graph_consumers_share_lim_miss_storage_across_replays():
    manager, metadata, buffers = fixture()
    buffers.update(make_mtp_batch(metadata, manager))
    outputs = buffers.outputs
    assert buffers.copy_metadata() is outputs
    assert buffers.runtime.copy_metadata(buffers.batch) is outputs
    for tensor in outputs[3:5]:
        assert tensor.shape == (3, 32768) and tensor.is_contiguous()
    snapshots = [torch.empty_like(value) for value in outputs[3:5]]
    pointers = [value.data_ptr() for value in outputs[3:5]]
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        for snapshot, value in zip(snapshots, buffers.copy_metadata()[3:5]):
            snapshot.copy_(value)
    for count in (2, 1, 2):
        metadata.num_decodes = count
        buffers.update(make_mtp_batch(metadata, manager))
        outputs[3].fill_(count)
        outputs[4].fill_(count + 7)
        graph.replay()
        torch.npu.synchronize()
        assert [value.data_ptr() for value in buffers.copy_metadata()[3:5]] == pointers
        for snapshot, value in zip(snapshots, outputs[3:5]):
            torch.testing.assert_close(snapshot, value, rtol=0, atol=0)


def test_nonuniform_batch_does_not_reuse_uniform_graph():
    manager, metadata, buffers = fixture()
    metadata.cum_query_lens.copy_(torch.tensor([1, 5], dtype=torch.int32, device="npu:0"))
    batch = make_mtp_batch(metadata, manager)
    assert not buffers.accepts(batch)
    with pytest.raises(ValueError, match="uniform-query capacity"):
        buffers.update(batch)


def test_graph_step_uses_owned_batch_after_builder_scratch_is_reused():
    manager, source, buffers = fixture()
    batch = make_mtp_batch(source, manager)
    device = source.seq_lens.device
    metadata = AscendSFAKVOffloadMetadata(
        num_actual_tokens=8,
        num_input_tokens=16,
        slot_mapping=torch.arange(16, dtype=torch.int32, device=device),
        seq_lens=source.seq_lens,
        seq_lens_cpu=source.seq_lens.cpu(),
        cum_query_lens=source.cum_query_lens,
        block_table=source.block_table,
        sin=torch.zeros(16, 64, device=device),
        cos=torch.ones(16, 64, device=device),
        num_decodes=2,
        num_decode_tokens=8,
        mtp_batch=batch,
    )
    graphs = MtpGraphMetadataSet(buffers.runtime, [{"owner": metadata}])
    # The next draft builder reuses these SFA scratch buffers, while the
    # first step's MtpBatch still owns its rejection-adjusted snapshots.
    source.cum_query_lens.zero_()
    source.seq_lens.zero_()
    source.block_table.zero_()
    graphs.update([{"owner": metadata}])
    captured = graphs.steps[0]["owner"]
    assert captured.cum_query_lens.cpu().tolist() == [4, 8, 12]
    assert captured.seq_lens.cpu().tolist() == [8452, 8196, 8196]
    torch.testing.assert_close(captured.block_table[:2], batch.source_block_table)
    assert captured.req_topk_buffer_slots.cpu().tolist() == [2, 0, 5]
    assert captured.slot_mapping.cpu().tolist() == list(range(8)) + [-1] * 8
    # A later shrinking batch must not retain previously live D2H slots.
    source.num_decodes = 1
    source.seq_lens[0] = 8452
    source.cum_query_lens[0] = 4
    metadata.mtp_batch = make_mtp_batch(source, manager)
    graphs.update([{"owner": metadata}])
    assert captured.slot_mapping.cpu().tolist() == list(range(4)) + [-1] * 12


def test_graph_step_keeps_ordinary_attention_metadata():
    manager, source, buffers = fixture()
    batch = make_mtp_batch(source, manager)
    device = source.seq_lens.device
    sparse = AscendSFAKVOffloadMetadata(
        num_actual_tokens=8,
        num_input_tokens=16,
        slot_mapping=torch.arange(16, dtype=torch.int32, device=device),
        seq_lens=source.seq_lens,
        seq_lens_cpu=source.seq_lens.cpu(),
        cum_query_lens=source.cum_query_lens,
        block_table=source.block_table,
        sin=torch.zeros(16, 64, device=device),
        cos=torch.ones(16, 64, device=device),
        num_decodes=2,
        num_decode_tokens=8,
        mtp_batch=batch,
    )
    ordinary = SimpleNamespace(seq_lens=source.seq_lens)
    step = {"ordinary": ordinary, "sparse": sparse}

    assert eligible_graph_steps([step])
    assert not eligible_graph_steps([{"ordinary": ordinary}])
    graphs = MtpGraphMetadataSet(buffers.runtime, [step])
    assert graphs.steps[0]["ordinary"] is ordinary
    graphs.update([step])


def test_graph_owns_separate_indexer_mappings_and_refreshes_them():
    manager, source, buffers = fixture()
    batch = make_mtp_batch(source, manager)
    device = source.seq_lens.device
    metadata = AscendSFAKVOffloadMetadata(
        num_actual_tokens=8,
        num_input_tokens=16,
        slot_mapping=torch.arange(16, dtype=torch.int32, device=device),
        seq_lens=source.seq_lens,
        seq_lens_cpu=source.seq_lens.cpu(),
        cum_query_lens=source.cum_query_lens,
        block_table=source.block_table,
        sin=torch.zeros(16, 64, device=device),
        cos=torch.ones(16, 64, device=device),
        num_decodes=2,
        num_decode_tokens=8,
        mtp_batch=batch,
    )
    indexer = AscendSFAIndexerMetadata(source.block_table + 100, metadata.slot_mapping + 50)
    step = {"main": metadata, "indexer": indexer}
    graphs = MtpGraphMetadataSet(buffers.runtime, [step])
    owned = graphs.steps[0]["indexer"]
    assert owned.block_table.data_ptr() != indexer.block_table.data_ptr()
    pointers = [owned.block_table.data_ptr(), owned.slot_mapping.data_ptr()]
    graphs.update([step])
    snapshots = [torch.empty_like(owned.block_table), torch.empty_like(owned.slot_mapping)]
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        snapshots[0].copy_(owned.block_table)
        snapshots[1].copy_(owned.slot_mapping)
    indexer.block_table.add_(17)
    indexer.slot_mapping.add_(11)
    graphs.update([step])
    graph.replay()
    torch.npu.synchronize()
    assert pointers == [owned.block_table.data_ptr(), owned.slot_mapping.data_ptr()]
    torch.testing.assert_close(snapshots[0][:2], indexer.block_table)
    assert snapshots[0][2].count_nonzero().item() == 0
    torch.testing.assert_close(snapshots[1], indexer.slot_mapping)
    torch.testing.assert_close(graphs.steps[0]["main"].block_table[:2], batch.source_block_table)
