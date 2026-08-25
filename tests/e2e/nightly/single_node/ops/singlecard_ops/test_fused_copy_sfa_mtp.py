#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""E2E tests for the ``npu_fused_copy_sfa_mtp`` custom Ascend operator.

Migrated from ``nanovllm-DSA-offload/ut_ops/test_mtp_offload_chain.py``.

This test validates the MTP3 fused copy+attention chain: LIM outputs feed
directly into fused_copy_sfa_mtp which fuses DRAM->HBM scatter copy and
sparse+tail attention into one kernel launch. The reference is a CPU
golden implementation (attention_golden). The split chain (separate
scatter_copy + sparse_tail_attention_mtp) is not tested here because
those operators are not migrated.
"""

from __future__ import annotations

import gc
import math
from dataclasses import replace

import pytest
import torch
import torch_npu  # type: ignore

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = False
enable_custom_op()

import importlib
import sys

_lim_mod = importlib.import_module(
    "tests.e2e.nightly.single_node.ops.singlecard_ops.test_fused_li_manage_mtp"
)
sys.modules["test_fused_li_manage_mtp"] = _lim_mod

QUERY_COUNT = _lim_mod.QUERY_COUNT
BLOCK_SIZE = _lim_mod.BLOCK_SIZE
TOPK = _lim_mod.TOPK
UNION_CAPACITY = _lim_mod.UNION_CAPACITY
CKV_DIM = 512
KPE_DIM = 64


def _apply_scatter_reference(
    expected_kpe, expected_ckv, dram_kpe, dram_ckv,
    hbm_block_table, dram_block_table, source_ids,
    destination_slots, copy_counts,
):
    for request, count_value in enumerate(copy_counts.tolist()):
        count = int(count_value)
        if count == 0:
            continue
        sources = source_ids[request, :count].to(torch.int64)
        destinations = destination_slots[request, :count].to(torch.int64)
        src_blocks = dram_block_table[request, sources // BLOCK_SIZE].to(torch.int64)
        src_offsets = sources % BLOCK_SIZE
        dst_blocks = hbm_block_table[request, destinations // BLOCK_SIZE].to(torch.int64)
        dst_offsets = destinations % BLOCK_SIZE
        expected_kpe[dst_blocks, dst_offsets] = dram_kpe[src_blocks, src_offsets]
        expected_ckv[dst_blocks, dst_offsets] = dram_ckv[src_blocks, src_offsets]


def _host_from_cpu(cpu):
    """Offloaded DRAM KV stays on host for fused_copy_sfa_mtp."""
    return cpu.contiguous()


def _random_block_table(batch_size, blocks_per_request, generator):
    total_blocks = batch_size * blocks_per_request
    table = torch.randperm(total_blocks, generator=generator).to(torch.int32)
    return table.view(batch_size, blocks_per_request).contiguous(), total_blocks


def logical_rows(cache, block_table, request, logical_slots):
    slots = logical_slots.to(torch.int64)
    blocks = block_table[request, slots // BLOCK_SIZE].to(torch.int64)
    return cache[blocks, slots % BLOCK_SIZE]


def make_miss_fractions(batch_size):
    if batch_size == 1:
        return (1.0,)
    return tuple(r / (batch_size - 1) for r in range(batch_size))


def initialize_hbm(*, case, cache_tokens, final_kv_len, dram_kpe, dram_ckv,
                   dram_table, hbm_table, hbm_blocks, generator):
    initial_kpe = torch.zeros(hbm_blocks, BLOCK_SIZE, KPE_DIM, dtype=torch.bfloat16)
    initial_ckv = torch.zeros(hbm_blocks, BLOCK_SIZE, CKV_DIM, dtype=torch.bfloat16)
    source_ids = torch.empty(case.batch_size, cache_tokens, dtype=torch.int32)
    destination_slots = torch.empty_like(source_ids)
    for request in range(case.batch_size):
        pool_row = int(case.req_pool_entries_cpu[request])
        state = case.initial_cache_cpu[pool_row, : case.source_capacity]
        sources = torch.nonzero(state >= 0).flatten()
        source_ids[request] = sources.to(torch.int32)
        destination_slots[request] = state[sources]
    _apply_scatter_reference(
        initial_kpe, initial_ckv, dram_kpe, dram_ckv, hbm_table, dram_table,
        source_ids, destination_slots,
        torch.full((case.batch_size,), cache_tokens, dtype=torch.int32))
    dense_count = final_kv_len - cache_tokens
    dense_kpe = torch.randn(case.batch_size, dense_count, KPE_DIM,
                            generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    dense_ckv = torch.randn(case.batch_size, dense_count, CKV_DIM,
                            generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    dense_slots = torch.arange(cache_tokens, final_kv_len, dtype=torch.int64)
    for request in range(case.batch_size):
        blocks = hbm_table[request, dense_slots // BLOCK_SIZE].to(torch.int64)
        offsets = dense_slots % BLOCK_SIZE
        initial_kpe[blocks, offsets] = dense_kpe[request]
        initial_ckv[blocks, offsets] = dense_ckv[request]
    return initial_kpe.contiguous(), initial_ckv.contiguous()


def expected_after_scatter(*, initial_kpe, initial_ckv, dram_kpe, dram_ckv,
                           hbm_table, dram_table, lim_outputs):
    expected_kpe = initial_kpe.clone()
    expected_ckv = initial_ckv.clone()
    counts = lim_outputs[4].cpu()
    _apply_scatter_reference(
        expected_kpe, expected_ckv, dram_kpe, dram_ckv, hbm_table, dram_table,
        lim_outputs[2].cpu(), lim_outputs[3].cpu(), counts)
    return expected_kpe, expected_ckv, [int(v) for v in counts.tolist()]


def attention_golden(*, query, query_rope, kpe, ckv, hbm_table,
                     sparse_slots, cache_tokens, tail_tokens, scale):
    rows = []
    sparse = sparse_slots.reshape(-1, TOPK).to(torch.int64)
    for request in range(hbm_table.shape[0]):
        for query_idx in range(QUERY_COUNT):
            row = request * QUERY_COUNT + query_idx
            dense_end = cache_tokens + tail_tokens + query_idx + 1
            logical_slots = torch.cat((sparse[row], torch.arange(
                cache_tokens, dense_end, dtype=torch.int64)))
            selected_ckv = logical_rows(ckv, hbm_table, request, logical_slots).float()
            selected_kpe = logical_rows(kpe, hbm_table, request, logical_slots).float()
            scores = (query[row].float() @ selected_ckv.T +
                      query_rope[row].float() @ selected_kpe.T) * scale
            rows.append(torch.softmax(scores, dim=-1) @ selected_ckv)
    return torch.stack(rows)


def launch_fused(*, case, device, query, query_rope, actual_q, actual_kv,
                 scale, hbm_table, dram_table, dram_kpe, dram_ckv,
                 cache_slots, hbm_kpe, hbm_ckv, lim_buffers, attention_output):
    lim_outputs = _lim_mod.call_mtp_with_buffers(case, cache_slots, *lim_buffers)
    torch.ops._C_ascend.npu_fused_copy_sfa_mtp(
        query_rope, query, actual_q, actual_kv, case.cache_tokens,
        lim_outputs[0], lim_outputs[1], lim_outputs[2], lim_outputs[3],
        lim_outputs[4], hbm_table, dram_table,
        hbm_kpe.view(-1, BLOCK_SIZE, 1, KPE_DIM),
        hbm_ckv.view(-1, BLOCK_SIZE, 1, CKV_DIM),
        dram_kpe, dram_ckv, scale, attention_output)
    return lim_outputs, attention_output


def validate_chain(*, case, device, label, before_cache, cache_slots,
                   hbm_kpe, hbm_ckv, lim_outputs, attention,
                   initial_kpe_cpu, initial_ckv_cpu, dram_kpe_cpu, dram_ckv_cpu,
                   hbm_table_cpu, dram_table_cpu, cache_tokens, tail_tokens,
                   query_cpu, query_rope_cpu, scale):
    counts = _lim_mod.validate_result(case, before_cache, cache_slots,
                                      lim_outputs, label=label)
    expected_kpe, expected_ckv, payload_counts = expected_after_scatter(
        initial_kpe=initial_kpe_cpu, initial_ckv=initial_ckv_cpu,
        dram_kpe=dram_kpe_cpu, dram_ckv=dram_ckv_cpu,
        hbm_table=hbm_table_cpu, dram_table=dram_table_cpu,
        lim_outputs=lim_outputs)
    assert counts == payload_counts, f"{label}: LIM and SCATTER counts differ"
    assert torch.equal(hbm_kpe.cpu(), expected_kpe), f"{label}: KPE payload mismatch"
    assert torch.equal(hbm_ckv.cpu(), expected_ckv), f"{label}: CKV payload mismatch"
    golden = attention_golden(query=query_cpu, query_rope=query_rope_cpu,
                              kpe=expected_kpe, ckv=expected_ckv,
                              hbm_table=hbm_table_cpu,
                              sparse_slots=lim_outputs[0].cpu(),
                              cache_tokens=cache_tokens, tail_tokens=tail_tokens,
                              scale=scale)
    actual = attention.float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    return counts, float((actual - golden).abs().max())


@pytest.fixture(scope="module")
def device():
    dev = torch.device("npu:0")
    torch.npu.set_device(dev)
    return dev


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    gc.collect()
    torch.npu.empty_cache()
    yield
    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("batch_size,heads,source_len,cache_tokens,tail_tokens", [
    (4, 2, 20992, 8192, 64),
])
def test_fused_copy_sfa_mtp_chain(device, batch_size, heads, source_len,
                                  cache_tokens, tail_tokens):
    final_kv_len = cache_tokens + tail_tokens + QUERY_COUNT
    hbm_capacity = math.ceil(final_kv_len / BLOCK_SIZE) * BLOCK_SIZE
    case = _lim_mod.make_case(
        name="mtp_offload_chain", device=device, dtype=torch.bfloat16,
        candidate_lens=(source_len,) * batch_size,
        cache_tokens=(cache_tokens,) * batch_size,
        miss_fractions=make_miss_fractions(batch_size),
        source_capacity=source_len, seed=5007)
    generator = torch.Generator().manual_seed(5017)
    dram_table_cpu, dram_blocks = _random_block_table(
        batch_size, source_len // BLOCK_SIZE, generator)
    hbm_table_cpu, hbm_blocks = _random_block_table(
        batch_size, hbm_capacity // BLOCK_SIZE, generator)
    dram_kpe_cpu = torch.randn(dram_blocks, BLOCK_SIZE, KPE_DIM,
                               generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    dram_ckv_cpu = torch.randn(dram_blocks, BLOCK_SIZE, CKV_DIM,
                               generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    initial_kpe_cpu, initial_ckv_cpu = initialize_hbm(
        case=case, cache_tokens=cache_tokens, final_kv_len=final_kv_len,
        dram_kpe=dram_kpe_cpu, dram_ckv=dram_ckv_cpu, dram_table=dram_table_cpu,
        hbm_table=hbm_table_cpu, hbm_blocks=hbm_blocks, generator=generator)
    query_cpu = torch.randn(batch_size * QUERY_COUNT, heads, CKV_DIM,
                            generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    query_rope_cpu = torch.randn(batch_size * QUERY_COUNT, heads, KPE_DIM,
                                 generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    query = query_cpu.to(device)
    query_rope = query_rope_cpu.to(device)
    dram_kpe = _host_from_cpu(dram_kpe_cpu)
    dram_ckv = _host_from_cpu(dram_ckv_cpu)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    actual_q = torch.arange(QUERY_COUNT, batch_size * QUERY_COUNT + 1,
                            QUERY_COUNT, dtype=torch.int32, device=device)
    actual_kv = torch.full((batch_size,), final_kv_len, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)

    fused_cache = case.initial_cache_cpu.to(device)
    fused_kpe = initial_kpe_cpu.to(device)
    fused_ckv = initial_ckv_cpu.to(device)
    fused_attention_buffer = torch.empty(
        batch_size * QUERY_COUNT, heads, CKV_DIM, dtype=torch.bfloat16, device=device)
    fused_outputs, fused_attention = launch_fused(
        case=case, device=device, query=query, query_rope=query_rope,
        actual_q=actual_q, actual_kv=actual_kv, scale=scale,
        hbm_table=hbm_table, dram_table=dram_table, dram_kpe=dram_kpe,
        dram_ckv=dram_ckv, cache_slots=fused_cache, hbm_kpe=fused_kpe,
        hbm_ckv=fused_ckv, lim_buffers=_lim_mod.make_outputs(case),
        attention_output=fused_attention_buffer)
    torch.npu.synchronize()
    fused_counts, fused_max_abs = validate_chain(
        case=case, device=device, label="mtp_offload_chain/fused_eager",
        before_cache=case.initial_cache_cpu, cache_slots=fused_cache,
        hbm_kpe=fused_kpe, hbm_ckv=fused_ckv, lim_outputs=fused_outputs,
        attention=fused_attention, initial_kpe_cpu=initial_kpe_cpu,
        initial_ckv_cpu=initial_ckv_cpu, dram_kpe_cpu=dram_kpe_cpu,
        dram_ckv_cpu=dram_ckv_cpu, hbm_table_cpu=hbm_table_cpu,
        dram_table_cpu=dram_table_cpu, cache_tokens=cache_tokens,
        tail_tokens=tail_tokens, query_cpu=query_cpu, query_rope_cpu=query_rope_cpu,
        scale=scale)
    assert max(fused_counts) > TOPK and max(fused_counts) <= UNION_CAPACITY, \
        "chain coverage must include a union miss_count in (2048,8192]"
    print(f"FUSED_COPY_SFA_MTP_CHECK batch={batch_size} misses={fused_counts} "
          f"attention_max_abs={fused_max_abs:.6f} ok=1", flush=True)

    del case, fused_cache, fused_kpe, fused_ckv
    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("batch_size,heads,source_len,cache_tokens,tail_tokens", [
    (4, 2, 20992, 8192, 64),
])
def test_fused_copy_sfa_mtp_graph(device, batch_size, heads, source_len,
                                  cache_tokens, tail_tokens):
    final_kv_len = cache_tokens + tail_tokens + QUERY_COUNT
    hbm_capacity = math.ceil(final_kv_len / BLOCK_SIZE) * BLOCK_SIZE
    case = _lim_mod.make_case(
        name="mtp_offload_chain_graph", device=device, dtype=torch.bfloat16,
        candidate_lens=(source_len,) * batch_size,
        cache_tokens=(cache_tokens,) * batch_size,
        miss_fractions=make_miss_fractions(batch_size),
        source_capacity=source_len, seed=6007)
    generator = torch.Generator().manual_seed(6017)
    dram_table_cpu, dram_blocks = _random_block_table(
        batch_size, source_len // BLOCK_SIZE, generator)
    hbm_table_cpu, hbm_blocks = _random_block_table(
        batch_size, hbm_capacity // BLOCK_SIZE, generator)
    dram_kpe_cpu = torch.randn(dram_blocks, BLOCK_SIZE, KPE_DIM,
                               generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    dram_ckv_cpu = torch.randn(dram_blocks, BLOCK_SIZE, CKV_DIM,
                               generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    initial_kpe_cpu, initial_ckv_cpu = initialize_hbm(
        case=case, cache_tokens=cache_tokens, final_kv_len=final_kv_len,
        dram_kpe=dram_kpe_cpu, dram_ckv=dram_ckv_cpu, dram_table=dram_table_cpu,
        hbm_table=hbm_table_cpu, hbm_blocks=hbm_blocks, generator=generator)
    query_cpu = torch.randn(batch_size * QUERY_COUNT, heads, CKV_DIM,
                            generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    query_rope_cpu = torch.randn(batch_size * QUERY_COUNT, heads, KPE_DIM,
                                 generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    query = query_cpu.to(device)
    query_rope = query_rope_cpu.to(device)
    dram_kpe = _host_from_cpu(dram_kpe_cpu)
    dram_ckv = _host_from_cpu(dram_ckv_cpu)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    actual_q = torch.arange(QUERY_COUNT, batch_size * QUERY_COUNT + 1,
                            QUERY_COUNT, dtype=torch.int32, device=device)
    actual_kv = torch.full((batch_size,), final_kv_len, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)

    warm_cache = case.initial_cache_cpu.to(device)
    warm_kpe = initial_kpe_cpu.to(device)
    warm_ckv = initial_ckv_cpu.to(device)
    warm_attention = torch.empty(
        batch_size * QUERY_COUNT, heads, CKV_DIM, dtype=torch.bfloat16, device=device)
    launch_fused(case=case, device=device, query=query, query_rope=query_rope,
                 actual_q=actual_q, actual_kv=actual_kv, scale=scale,
                 hbm_table=hbm_table, dram_table=dram_table, dram_kpe=dram_kpe,
                 dram_ckv=dram_ckv, cache_slots=warm_cache, hbm_kpe=warm_kpe,
                 hbm_ckv=warm_ckv, lim_buffers=_lim_mod.make_outputs(case),
                 attention_output=warm_attention)
    torch.npu.synchronize()

    graph_cache = case.initial_cache_cpu.to(device)
    graph_kpe = initial_kpe_cpu.to(device)
    graph_ckv = initial_ckv_cpu.to(device)
    graph_buffers = _lim_mod.make_outputs(case)
    graph_attention_buffer = torch.empty_like(warm_attention)
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        graph_outputs, graph_attention = launch_fused(
            case=case, device=device, query=query, query_rope=query_rope,
            actual_q=actual_q, actual_kv=actual_kv, scale=scale,
            hbm_table=hbm_table, dram_table=dram_table, dram_kpe=dram_kpe,
            dram_ckv=dram_ckv, cache_slots=graph_cache, hbm_kpe=graph_kpe,
            hbm_ckv=graph_ckv, lim_buffers=graph_buffers,
            attention_output=graph_attention_buffer)
    torch.npu.synchronize()

    graph_cache.copy_(case.initial_cache_cpu.to(device))
    graph_kpe.copy_(initial_kpe_cpu.to(device))
    graph_ckv.copy_(initial_ckv_cpu.to(device))
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    graph_counts, graph_max_abs = validate_chain(
        case=case, device=device, label="mtp_offload_chain/graph",
        before_cache=case.initial_cache_cpu, cache_slots=graph_cache,
        hbm_kpe=graph_kpe, hbm_ckv=graph_ckv, lim_outputs=graph_outputs,
        attention=graph_attention, initial_kpe_cpu=initial_kpe_cpu,
        initial_ckv_cpu=initial_ckv_cpu, dram_kpe_cpu=dram_kpe_cpu,
        dram_ckv_cpu=dram_ckv_cpu, hbm_table_cpu=hbm_table_cpu,
        dram_table_cpu=dram_table_cpu, cache_tokens=cache_tokens,
        tail_tokens=tail_tokens, query_cpu=query_cpu, query_rope_cpu=query_rope_cpu,
        scale=scale)
    print(f"FUSED_COPY_SFA_MTP_GRAPH_CHECK graph_max_abs={graph_max_abs:.6f} ok=1",
          flush=True)

    repeat_kpe_before = graph_kpe.cpu()
    repeat_ckv_before = graph_ckv.cpu()
    repeat_attention_before = graph_attention.cpu()
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(graph_kpe.cpu(), repeat_kpe_before), "zero-miss replay modified KPE"
    assert torch.equal(graph_ckv.cpu(), repeat_ckv_before), "zero-miss replay modified CKV"
    assert torch.equal(graph_attention.cpu(), repeat_attention_before), \
        "zero-miss replay changed attention output"
    print("FUSED_COPY_SFA_MTP_GRAPH_REPEAT_CHECK zero_miss=1 ok=1", flush=True)

    del case, graph_cache, graph_kpe, graph_ckv, warm_cache, warm_kpe, warm_ckv
    gc.collect()
    torch.npu.empty_cache()
