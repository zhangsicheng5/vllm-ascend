"""E2E tests for the ``npu_sparse_tail_attention`` custom Ascend operator.

Migrated from ``nanovllm-DSA-offload/ut_ops/test_sparse_tail_attention.py``.

Validates query_len=1 sparse+tail attention against a CPU golden.
Performance comparison against dense MLA is optional (printed only).
"""

from __future__ import annotations

import gc
import math

import pytest
import torch
import torch_npu  # type: ignore

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = False
enable_custom_op()

BLOCK_SIZE = 128
CKV_DIM = 512
KPE_DIM = 64
TOPK = 2048


def _random_block_table(batch_size, blocks_per_request, generator):
    total_blocks = batch_size * blocks_per_request
    table = torch.randperm(total_blocks, generator=generator).to(torch.int32)
    return table.view(batch_size, blocks_per_request).contiguous(), total_blocks


def _logical_rows(cache, block_table, row, logical_slots):
    logical_slots = logical_slots.to(torch.int64)
    physical_blocks = block_table[row, logical_slots // BLOCK_SIZE].to(torch.int64)
    return cache[physical_blocks, logical_slots % BLOCK_SIZE, 0]


def _sparse_attention(query, key, sparse_slots, cache_tokens, block_table,
                      actual_q, actual_kv, query_rope, key_rope, scale):
    attention_out = torch.empty_like(query)
    torch.ops._C_ascend.npu_sparse_tail_attention(
        query_rope, query, actual_q, actual_kv, cache_tokens,
        sparse_slots, block_table, key_rope, key, scale, attention_out)
    return attention_out


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


@pytest.mark.parametrize("heads", [4, 128])
def test_sparse_tail_attention_semantic(device, heads):
    seed = 7
    cases = ((0, 1), (0, 129), (0, 2049), (2048, 0), (6144, 257), (5120, 1025))
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(cases)
    actual_lens = [c + t for c, t in cases]
    max_blocks = math.ceil(max(actual_lens) / BLOCK_SIZE)
    block_table_cpu, total_blocks = _random_block_table(batch_size, max_blocks, generator)
    query_cpu = torch.randn(batch_size, heads, CKV_DIM, generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    query_rope_cpu = torch.randn(batch_size, heads, KPE_DIM, generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    key_cpu = torch.randn(total_blocks, BLOCK_SIZE, 1, CKV_DIM, generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    key_rope_cpu = torch.randn(total_blocks, BLOCK_SIZE, 1, KPE_DIM, generator=generator, dtype=torch.float32).mul_(0.25).to(torch.bfloat16)
    sparse_slots_cpu = torch.zeros(batch_size, 1, TOPK, dtype=torch.int32)
    selected_slots = []
    for row, (cache_tokens, _) in enumerate(cases):
        if cache_tokens == 0:
            selected = torch.empty(0, dtype=torch.int64)
        else:
            selected = torch.randperm(cache_tokens, generator=generator)[:TOPK]
            sparse_slots_cpu[row, 0] = selected.to(torch.int32)
        selected_slots.append(selected)
    cache_tokens_cpu = torch.tensor([c for c, _ in cases], dtype=torch.int32)
    actual_kv_cpu = torch.tensor(actual_lens, dtype=torch.int32)
    actual_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)
    output = _sparse_attention(
        query_cpu.to(device), key_cpu.to(device), sparse_slots_cpu.to(device),
        cache_tokens_cpu.to(device), block_table_cpu.to(device), actual_q,
        actual_kv_cpu.to(device), query_rope_cpu.to(device), key_rope_cpu.to(device), scale)
    torch.npu.synchronize()

    golden_rows = []
    for row, (cache_tokens, tail_tokens) in enumerate(cases):
        logical_slots = selected_slots[row]
        if cache_tokens == 0:
            logical_slots = torch.arange(tail_tokens, dtype=torch.int64)
        elif tail_tokens:
            logical_slots = torch.cat((logical_slots, torch.arange(cache_tokens, cache_tokens + tail_tokens, dtype=torch.int64)))
        key = _logical_rows(key_cpu, block_table_cpu, row, logical_slots).float()
        key_rope = _logical_rows(key_rope_cpu, block_table_cpu, row, logical_slots).float()
        query = query_cpu[row].float()
        query_rope = query_rope_cpu[row].float()
        scores = (query @ key.T + query_rope @ key_rope.T) * scale
        golden_rows.append(torch.softmax(scores, dim=-1) @ key)
    golden = torch.stack(golden_rows)
    actual = output.float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    print(f"SPARSE_TAIL_ATTENTION_CHECK heads={heads} max_abs={float((actual - golden).abs().max()):.6f} ok=1", flush=True)


def _build_case(device, heads, seed, dtype=torch.bfloat16):
    cases = ((0, 1), (0, 129), (0, 2049), (2048, 0), (6144, 257), (5120, 1025))
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(cases)
    actual_lens = [c + t for c, t in cases]
    max_blocks = math.ceil(max(actual_lens) / BLOCK_SIZE)
    block_table_cpu, total_blocks = _random_block_table(batch_size, max_blocks, generator)
    query_cpu = torch.randn(batch_size, heads, CKV_DIM, generator=generator, dtype=torch.float32).mul_(0.25).to(dtype)
    query_rope_cpu = torch.randn(batch_size, heads, KPE_DIM, generator=generator, dtype=torch.float32).mul_(0.25).to(dtype)
    key_cpu = torch.randn(total_blocks, BLOCK_SIZE, 1, CKV_DIM, generator=generator, dtype=torch.float32).mul_(0.25).to(dtype)
    key_rope_cpu = torch.randn(total_blocks, BLOCK_SIZE, 1, KPE_DIM, generator=generator, dtype=torch.float32).mul_(0.25).to(dtype)
    sparse_slots_cpu = torch.zeros(batch_size, 1, TOPK, dtype=torch.int32)
    selected_slots = []
    for row, (cache_tokens, _) in enumerate(cases):
        if cache_tokens == 0:
            selected = torch.empty(0, dtype=torch.int64)
        else:
            selected = torch.randperm(cache_tokens, generator=generator)[:TOPK]
            sparse_slots_cpu[row, 0] = selected.to(torch.int32)
        selected_slots.append(selected)
    cache_tokens_cpu = torch.tensor([c for c, _ in cases], dtype=torch.int32)
    actual_kv_cpu = torch.tensor(actual_lens, dtype=torch.int32)
    return {
        "device": device, "heads": heads, "cases": cases, "dtype": dtype,
        "block_table_cpu": block_table_cpu, "query_cpu": query_cpu,
        "query_rope_cpu": query_rope_cpu, "key_cpu": key_cpu,
        "key_rope_cpu": key_rope_cpu, "sparse_slots_cpu": sparse_slots_cpu,
        "selected_slots": selected_slots, "cache_tokens_cpu": cache_tokens_cpu,
        "actual_kv_cpu": actual_kv_cpu, "scale": 1.0 / math.sqrt(CKV_DIM + KPE_DIM),
    }


def _compute_golden(case):
    cases = case["cases"]
    golden_rows = []
    for row, (cache_tokens, tail_tokens) in enumerate(cases):
        logical_slots = case["selected_slots"][row]
        if cache_tokens == 0:
            logical_slots = torch.arange(tail_tokens, dtype=torch.int64)
        elif tail_tokens:
            logical_slots = torch.cat((logical_slots, torch.arange(cache_tokens, cache_tokens + tail_tokens, dtype=torch.int64)))
        key = _logical_rows(case["key_cpu"], case["block_table_cpu"], row, logical_slots).float()
        key_rope = _logical_rows(case["key_rope_cpu"], case["block_table_cpu"], row, logical_slots).float()
        query = case["query_cpu"][row].float()
        query_rope = case["query_rope_cpu"][row].float()
        scores = (query @ key.T + query_rope @ key_rope.T) * case["scale"]
        golden_rows.append(torch.softmax(scores, dim=-1) @ key)
    return torch.stack(golden_rows)


@pytest.mark.parametrize("heads", [8])
def test_sparse_tail_attention_fp16(device, heads):
    case = _build_case(device, heads, seed=7, dtype=torch.float16)
    actual_q = torch.arange(1, 7, dtype=torch.int32, device=device)
    output = _sparse_attention(
        case["query_cpu"].to(device), case["key_cpu"].to(device),
        case["sparse_slots_cpu"].to(device), case["cache_tokens_cpu"].to(device),
        case["block_table_cpu"].to(device), actual_q, case["actual_kv_cpu"].to(device),
        case["query_rope_cpu"].to(device), case["key_rope_cpu"].to(device), case["scale"])
    torch.npu.synchronize()
    golden = _compute_golden(case)
    actual = output.float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    print(f"SPARSE_TAIL_ATTENTION_FP16_CHECK heads={heads} ok=1", flush=True)


@pytest.mark.parametrize("heads", [4])
def test_sparse_tail_attention_graph(device, heads):
    case = _build_case(device, heads, seed=7)
    actual_q = torch.arange(1, 7, dtype=torch.int32, device=device)
    query = case["query_cpu"].to(device)
    key = case["key_cpu"].to(device)
    sparse_slots = case["sparse_slots_cpu"].to(device)
    cache_tokens = case["cache_tokens_cpu"].to(device)
    block_table = case["block_table_cpu"].to(device)
    actual_kv = case["actual_kv_cpu"].to(device)
    query_rope = case["query_rope_cpu"].to(device)
    key_rope = case["key_rope_cpu"].to(device)
    scale = case["scale"]

    warm_out = _sparse_attention(query, key, sparse_slots, cache_tokens,
                                 block_table, actual_q, actual_kv, query_rope, key_rope, scale)
    torch.npu.synchronize()

    graph_out = torch.empty_like(warm_out)
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        torch.ops._C_ascend.npu_sparse_tail_attention(
            query_rope, query, actual_q, actual_kv, cache_tokens,
            sparse_slots, block_table, key_rope, key, scale, graph_out)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()

    golden = _compute_golden(case)
    actual = graph_out.float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    print(f"SPARSE_TAIL_ATTENTION_GRAPH_CHECK heads={heads} ok=1", flush=True)
