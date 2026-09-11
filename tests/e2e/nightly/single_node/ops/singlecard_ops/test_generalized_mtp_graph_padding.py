# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LIM -> copy-SFA graph replay with real requests and non-offload padding."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from memfabric_hybrid import offload

from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadMetadata
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp import (
    GeneralizedMtpRuntime,
    make_mtp_batch,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp_graph import (
    MtpGraphBuffers,
    MtpGraphMetadataSet,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.nano_cache import prepare_tail_copy
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    allocate_kv_offload_topk_buffer_pair,
)
from vllm_ascend.utils import enable_custom_op

BLOCK = 128
HOT = 8192
PREFIX = 10368
SOURCE_CAPACITY = 16384
REQUESTS = 4
TOPK = 2048


@pytest.fixture(scope="module")
def registered_pool():
    torch.npu.set_device(0)
    enable_custom_op()
    torch.empty(1, device="npu:0")
    config = offload.OffloadConfig()
    config.device_id, config.reserve_size, config.alloc_size = 0, 1 << 30, 1 << 30
    config.world_size, config.rank_id, config.scene = 1, 0, offload.Scene.SHARED
    assert offload.initialize(config) == 0
    yield
    torch.npu.synchronize()
    offload.uninitialize()


def test_dummy_replay_preserves_pd_host_and_indexer_caches(registered_pool):
    """Exercise the actual D2H/scatter writes with independently owned PD pools."""
    device = torch.device("npu:0")
    requests, width, tokens = 2, 4, 8
    manager = SimpleNamespace(
        block_size=BLOCK, topk_buffer_size=HOT, max_num_reqs=requests,
        max_num_topk_rows=8, max_num_tokens=tokens, tp_rank=0,
        topk_buffer_slot_manager=SimpleNamespace(req2slot={"waiting-a": 0, "waiting-b": 1}),
        nano_mtp_slot_generations={0: 1, 1: 1},
        sparse_kv_offload_config=SimpleNamespace(keep_device_kv_cache=False),
        token_size_bytes_k=1024, token_size_bytes_v=128,
        d2h_token_indices_npu=torch.arange(tokens, dtype=torch.int64, device=device),
        d2h_src_ptrs_npu=torch.zeros(2 * tokens, dtype=torch.int64, device=device),
        d2h_dst_ptrs_npu=torch.zeros(2 * tokens, dtype=torch.int64, device=device),
        d2h_lengths_npu=torch.zeros(2 * tokens, dtype=torch.int32, device=device),
        d2h_size_npu=torch.zeros(1, dtype=torch.int32, device=device),
    )
    host = [offload.empty((32, dim), dtype=torch.bfloat16, pin_memory=True) for dim in (512, 64)]
    for tensor in host:
        tensor.fill_(-7)
    index_cache = torch.full((32, 128), -9, dtype=torch.bfloat16, device=device)
    kv = [torch.ones(tokens, dim, dtype=torch.bfloat16, device=device) for dim in (512, 64)]
    index_key = torch.ones(tokens, 128, dtype=torch.bfloat16, device=device)
    slots = torch.arange(tokens, dtype=torch.int64, device=device)
    metadata = AscendSFAKVOffloadMetadata(
        num_actual_tokens=tokens, num_input_tokens=tokens, slot_mapping=slots,
        main_slot_mapping=slots.clone(),
        seq_lens=torch.full((requests,), HOT + width, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.full((requests,), HOT + width, dtype=torch.int32),
        cum_query_lens=torch.tensor([width, tokens], dtype=torch.int32, device=device),
        block_table=torch.zeros(requests, SOURCE_CAPACITY // BLOCK, dtype=torch.int32, device=device),
        sin=torch.zeros(tokens, 64, device=device), cos=torch.ones(tokens, 64, device=device),
        num_decodes=requests, num_decode_tokens=tokens,
        req_topk_buffer_slots=torch.arange(requests, dtype=torch.int32, device=device),
        mtp_graph_capture=True,
    )
    metadata.mtp_batch = make_mtp_batch(metadata, manager)
    indexer = AscendSFAIndexerMetadata(metadata.block_table.clone(), slots + 16)
    step = {"main": metadata, "indexer": indexer}
    graphs = MtpGraphMetadataSet(GeneralizedMtpRuntime(manager), [step])
    graphs.update([step])
    owned = graphs.steps[0]

    def writes():
        SparseKVOffloadManager.offload_new_kv(
            manager, slot_mapping=owned["main"].main_slot_mapping,
            k_cache_cpu=host[0], v_cache_cpu=host[1],
            k_cache_npu=None, v_cache_npu=None, k=kv[0], v=kv[1], capturing=True,
        )
        torch_npu.npu_scatter_nd_update_(index_cache, owned["indexer"].slot_mapping.view(-1, 1), index_key)

    writes()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        writes()
    torch.npu.synchronize()
    for tensor in host:
        assert (tensor == -7).all().item()
    assert (index_cache == -9).all().item()
    pointers = [owned[name].slot_mapping.data_ptr() for name in ("main", "indexer")]
    pointers.append(owned["main"].main_slot_mapping.data_ptr())

    for is_dummy, value in ((False, 2), (True, 3), (False, 4), (True, 5)):
        metadata.mtp_graph_capture = False
        metadata.mtp_graph_dummy = is_dummy
        graphs.update([step])
        for tensor in (*kv, index_key):
            tensor.fill_(value)
        host_before = [tensor.clone() for tensor in host]
        index_before = index_cache.clone()
        graph.replay()
        torch.npu.synchronize()
        assert pointers == [owned[name].slot_mapping.data_ptr() for name in ("main", "indexer")] + [
            owned["main"].main_slot_mapping.data_ptr()
        ]
        if is_dummy:
            assert manager.d2h_lengths_npu.count_nonzero().item() == 0
            for tensor, before in zip(host, host_before):
                torch.testing.assert_close(tensor, before, rtol=0, atol=0)
            torch.testing.assert_close(index_cache, index_before, rtol=0, atol=0)
        else:
            for tensor, before in zip(host, host_before):
                before[:tokens].fill_(value)
                torch.testing.assert_close(tensor, before, rtol=0, atol=0)
            index_before[16 : 16 + tokens].fill_(value)
            torch.testing.assert_close(index_cache, index_before, rtol=0, atol=0)


@pytest.mark.parametrize("query_width", [1, 4])
def test_padded_rows_generate_no_host_transfers_on_graph_replay(registered_pool, query_width):
    torch.manual_seed(20260907)
    device = torch.device("npu:0")
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=REQUESTS),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)),
        cache_config=SimpleNamespace(block_size=BLOCK),
    )
    offload_config = SimpleNamespace(topk_buffer_size=HOT, fused_op_type="nano", generalized_mtp=True)
    manager = SimpleNamespace(
        block_size=BLOCK,
        topk_buffer_size=HOT,
        max_num_reqs=REQUESTS,
        max_num_topk_rows=16,
        topk_buffer_slot_manager=SimpleNamespace(req2slot={str(i): i for i in range(REQUESTS)}),
        nano_mtp_slot_generations={i: 1 for i in range(REQUESTS)},
    )
    runtime = GeneralizedMtpRuntime(manager)
    buffers = MtpGraphBuffers(runtime, query_width=query_width, source_capacity=SOURCE_CAPACITY, device=device)
    source_blocks = SOURCE_CAPACITY // BLOCK
    table = torch.arange(1, source_blocks + 1, dtype=torch.int32, device=device).repeat(REQUESTS, 1)
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=REQUESTS,
        cum_query_lens=torch.arange(1, REQUESTS + 1, dtype=torch.int32, device=device) * query_width,
        seq_lens=torch.full((REQUESTS,), PREFIX + query_width, dtype=torch.int32, device=device),
        req_topk_buffer_slots=torch.arange(REQUESTS, dtype=torch.int32, device=device),
        block_table=table,
    )
    hbm = allocate_kv_offload_topk_buffer_pair(config, offload_config)
    for tensor in hbm:
        tensor[:REQUESTS].fill_(-7)
    paged_hbm = [tensor.view(-1, BLOCK, 1, tensor.shape[-1]) for tensor in hbm]
    host = [offload.empty((source_blocks + 1, BLOCK, dim), dtype=torch.bfloat16, pin_memory=True) for dim in (512, 64)]
    for tensor in host:
        tensor.copy_(torch.randn(tensor.shape, dtype=tensor.dtype))
        # All padded source-table entries point to block zero. Any accidental
        # host gather or first-fill would poison their private cache/output.
        tensor[0].fill_(float("nan"))
    index_key = torch.randn(source_blocks + 1, BLOCK, 1, 128, dtype=torch.bfloat16, device=device)
    tokens = REQUESTS * query_width
    index_query = torch.randn(tokens, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(tokens, 32, dtype=torch.bfloat16, device=device)
    query_scale = torch.ones(tokens, 32, dtype=torch.float32, device=device)
    key_scale = torch.ones(source_blocks + 1, BLOCK, 1, dtype=torch.float32, device=device)
    query = torch.randn(tokens, 8, 512, dtype=torch.bfloat16, device=device)
    query_rope = torch.randn(tokens, 8, 64, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(query)
    scale = 1 / math.sqrt(576)
    buffers.update(make_mtp_batch(metadata, manager), is_capture=True)
    buffers.prepare_layer("owner")
    mapping, states, outputs = buffers.lim_inputs("owner")
    descriptors = buffers.tail_copies["owner"]

    def chain():
        batch = buffers.batch
        torch.ops._C_ascend.npu_fused_li_manage_mtp(
            weights,
            query_scale,
            index_query,
            key_scale,
            index_key,
            batch.source_block_table,
            batch.query_ends,
            batch.seq_lens,
            batch.offload_lens,
            batch.cache_tokens,
            states,
            batch.pool_entries,
            mapping,
            *outputs,
        )
        prepare_tail_copy(
            descriptors,
            batch,
            block_size=BLOCK,
            hot_tokens=HOT,
            source_bases=tuple(t.data_ptr() for t in host),
            destination_bases=tuple(t.data_ptr() for t in hbm),
            token_bytes=(1024, 128),
            source_block_capacity=source_blocks + 1,
            active=buffers.active_mask,
        )
        offload.sparse_copy(*descriptors.args(), device)
        src, dst, topk_misses, miss_src, miss_dst, misses = buffers.copy_metadata()
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
            paged_hbm[1],
            paged_hbm[0],
            host[1],
            host[0],
            scale,
            output,
        )

    def check_padding(count):
        assert outputs[-1][count:].count_nonzero().item() == 0
        assert outputs[2][count * query_width :].count_nonzero().item() == 0
        lengths = descriptors.lengths.view(2, REQUESTS, 2)
        assert lengths[:, count:].count_nonzero().item() == 0
        for tensor in hbm:
            assert tensor[REQUESTS:].count_nonzero().item() == 0
        assert output[count * query_width :].isfinite().all().item()
        assert output[count * query_width :].count_nonzero().item() == 0

    # Warm up and capture only dummy rows, then update persistent metadata.
    chain()
    torch.npu.synchronize()
    check_padding(0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        chain()
    torch.npu.synchronize()
    check_padding(0)
    pointers = [tensor.data_ptr() for tensor in (mapping, states, *outputs, *descriptors.args())]
    for count, reset in ((4, False), (0, False), (4, False), (3, False), (1, False), (4, True), (4, False)):
        # A runtime dummy step reuses the graph and its private inactive rows.
        metadata.num_decodes = count or REQUESTS
        if reset:
            manager.nano_mtp_slot_generations[0] += 1
        buffers.update(make_mtp_batch(metadata, manager), is_dummy=count == 0)
        buffers.prepare_layer("owner")
        current_states = states.cpu().tolist()
        live_before = [tensor[:REQUESTS].clone() for tensor in hbm] if count == 0 else None
        graph.replay()
        torch.npu.synchronize()
        check_padding(count)
        if live_before is not None:
            for tensor, before in zip(hbm, live_before):
                torch.testing.assert_close(tensor[:REQUESTS], before, rtol=0, atol=0)
        assert pointers == [tensor.data_ptr() for tensor in (mapping, states, *outputs, *descriptors.args())]
        for row, state in enumerate(current_states):
            if state == -2:
                assert outputs[-1][row].item() == HOT
            elif state == -1:
                # Identical queries reuse their resident TopK on warm steps.
                assert outputs[-1][row].item() == 0
        # Reference attention reads logical host sources directly, independently
        # of LIM's physical destination mapping and copy-SFA's cache routing.
        src = outputs[0].cpu().reshape(tokens, TOPK).long()
        query_cpu, rope_cpu = query.cpu().float(), query_rope.cpu().float()
        for token in range(count * query_width):
            tail = torch.arange(PREFIX, PREFIX + token % query_width + 1)
            selected = torch.cat((src[token], tail))
            kv = host[0][1:].reshape(-1, 512)[selected].float()
            rope = host[1][1:].reshape(-1, 64)[selected].float()
            scores = (query_cpu[token] @ kv.T + rope_cpu[token] @ rope.T) * scale
            expected = torch.softmax(scores, dim=-1) @ kv
            torch.testing.assert_close(output[token].cpu().float(), expected, rtol=2e-2, atol=2e-2)
