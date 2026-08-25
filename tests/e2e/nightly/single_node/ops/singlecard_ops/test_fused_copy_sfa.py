"""E2E tests for the ``npu_fused_copy_sfa`` custom Ascend operator.

Migrated from ``nanovllm-DSA-offload/ut_ops/test_fused_copy_sfa.py``.

Validates the MTP0 fused scatter_copy + sparse_tail_attention chain.
The reference is a CPU golden attention implementation. The split chain
(separate scatter_copy + sparse_tail_attention) is not tested because
those operators are not migrated.
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
SPARSE_COUNT = 2048


def _host_from_cpu(cpu):
    """Offloaded DRAM KV stays on host for fused_copy_sfa."""
    return cpu.contiguous()


def _logical_rows(cache, block_table, request, logical_slots):
    slots = logical_slots.to(torch.int64)
    blocks = block_table[request, slots // BLOCK_SIZE].to(torch.int64)
    return cache[blocks, slots % BLOCK_SIZE]


def _make_case(device, batch, heads, source_len, cache_tokens, tail_tokens, seed, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    final_kv_len = cache_tokens + tail_tokens
    cache_blocks = math.ceil(final_kv_len / BLOCK_SIZE)
    source_blocks = source_len // BLOCK_SIZE
    hbm_table_cpu = torch.empty((batch, cache_blocks), dtype=torch.int32)
    for row in range(batch):
        physical = torch.arange(row * cache_blocks, (row + 1) * cache_blocks, dtype=torch.int32)
        hbm_table_cpu[row] = physical[torch.randperm(cache_blocks)]
    dram_table_cpu = torch.stack([torch.randperm(source_blocks, dtype=torch.int64).to(torch.int32) for _ in range(batch)])
    counts_cpu = torch.randint(10, SPARSE_COUNT + 1, (batch,), dtype=torch.int32)
    counts_cpu[0] = 10
    slots_cpu = torch.empty((batch, SPARSE_COUNT), dtype=torch.int32)
    source_ids_cpu = torch.full((batch, SPARSE_COUNT), -1, dtype=torch.int32)
    for row in range(batch):
        slots_cpu[row] = torch.randperm(cache_tokens, dtype=torch.int64)[:SPARSE_COUNT].to(torch.int32)
        count = int(counts_cpu[row])
        if count:
            source_ids_cpu[row, :count] = torch.randperm(source_len, dtype=torch.int64)[:count].to(torch.int32)
    sparse_slots = slots_cpu.view(batch, 1, SPARSE_COUNT).to(device)
    source_token_ids = source_ids_cpu.to(device)
    copy_counts = counts_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    dram_table = dram_table_cpu.to(device)
    dram_kpe_cpu = torch.randn(batch * source_blocks, BLOCK_SIZE, KPE_DIM, dtype=torch.float32).mul_(0.25).to(dtype)
    dram_ckv_cpu = torch.randn(batch * source_blocks, BLOCK_SIZE, CKV_DIM, dtype=torch.float32).mul_(0.25).to(dtype)
    dram_kpe = _host_from_cpu(dram_kpe_cpu)
    dram_ckv = _host_from_cpu(dram_ckv_cpu)
    total_hbm_blocks = batch * cache_blocks
    initial_kpe = torch.zeros(total_hbm_blocks, BLOCK_SIZE, 1, KPE_DIM, dtype=dtype, device=device)
    initial_ckv = torch.zeros(total_hbm_blocks, BLOCK_SIZE, 1, CKV_DIM, dtype=dtype, device=device)
    query = torch.randn(batch, heads, CKV_DIM, dtype=dtype, device=device).mul_(0.25)
    query_rope = torch.randn(batch, heads, KPE_DIM, dtype=dtype, device=device).mul_(0.25)
    actual_q = torch.tensor([1] * batch, dtype=torch.int32, device=device)
    actual_kv = torch.full((batch,), final_kv_len, dtype=torch.int32, device=device)
    cache_tokens_t = torch.full((batch,), cache_tokens, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)
    fused_kpe = initial_kpe.clone()
    fused_ckv = initial_ckv.clone()
    fused_out = torch.empty(batch, heads, CKV_DIM, dtype=dtype, device=device)
    return {
        "device": device, "batch": batch, "heads": heads, "source_len": source_len,
        "cache_tokens": cache_tokens, "tail_tokens": tail_tokens, "final_kv_len": final_kv_len,
        "hbm_table": hbm_table, "hbm_table_cpu": hbm_table_cpu,
        "dram_table": dram_table, "dram_table_cpu": dram_table_cpu,
        "dram_kpe": dram_kpe, "dram_kpe_cpu": dram_kpe_cpu,
        "dram_ckv": dram_ckv, "dram_ckv_cpu": dram_ckv_cpu,
        "sparse_slots": sparse_slots, "source_token_ids": source_token_ids,
        "copy_counts": copy_counts, "query": query, "query_rope": query_rope,
        "actual_q": actual_q, "actual_kv": actual_kv,
        "cache_tokens_t": cache_tokens_t, "scale": scale,
        "fused_kpe": fused_kpe, "fused_ckv": fused_ckv, "fused_out": fused_out,
        "initial_kpe": initial_kpe, "initial_ckv": initial_ckv,
    }


def _launch_fused(case):
    torch.ops._C_ascend.npu_fused_copy_sfa(
        case["query_rope"], case["query"], case["actual_q"], case["actual_kv"],
        case["cache_tokens_t"], case["sparse_slots"], case["source_token_ids"],
        case["copy_counts"], case["hbm_table"], case["dram_table"],
        case["fused_kpe"], case["fused_ckv"], case["dram_kpe"], case["dram_ckv"],
        case["scale"], case["fused_out"])
    return case["fused_out"]


def _cpu_golden(case):
    batch = case["batch"]; heads = case["heads"]
    cache_tokens = case["cache_tokens"]; tail_tokens = case["tail_tokens"]
    hbm_table = case["hbm_table_cpu"]; dram_table = case["dram_table_cpu"]
    dram_kpe = case["dram_kpe_cpu"]; dram_ckv = case["dram_ckv_cpu"]
    fused_kpe = case["fused_kpe"].cpu(); fused_ckv = case["fused_ckv"].cpu()
    sparse_slots = case["sparse_slots"].cpu().reshape(batch, SPARSE_COUNT).to(torch.int64)
    source_ids = case["source_token_ids"].cpu().to(torch.int64)
    copy_counts = case["copy_counts"].cpu()
    kpe_3d = fused_kpe.view(-1, KPE_DIM)
    ckv_3d = fused_ckv.view(-1, CKV_DIM)
    dram_kpe_3d = dram_kpe.view(-1, KPE_DIM)
    dram_ckv_3d = dram_ckv.view(-1, CKV_DIM)
    cursor = 0
    for row, count_val in enumerate(copy_counts.tolist()):
        count = int(count_val)
        if count:
            sources = source_ids[row, :count]
            dests = sparse_slots[row, :count]
            src_blocks = dram_table[row, sources // BLOCK_SIZE].to(torch.int64)
            src_offsets = sources % BLOCK_SIZE
            dst_blocks = hbm_table[row, dests // BLOCK_SIZE].to(torch.int64)
            dst_offsets = dests % BLOCK_SIZE
            kpe_3d[dst_blocks * BLOCK_SIZE + dst_offsets] = dram_kpe_3d[src_blocks * BLOCK_SIZE + src_offsets]
            ckv_3d[dst_blocks * BLOCK_SIZE + dst_offsets] = dram_ckv_3d[src_blocks * BLOCK_SIZE + src_offsets]
        cursor += count
    golden_rows = []
    for row in range(batch):
        dense_end = cache_tokens + tail_tokens
        logical_slots = torch.cat((sparse_slots[row], torch.arange(cache_tokens, dense_end, dtype=torch.int64)))
        selected_ckv = _logical_rows(fused_ckv.view(-1, BLOCK_SIZE, 1, CKV_DIM), hbm_table, row, logical_slots).float().squeeze(1)
        selected_kpe = _logical_rows(fused_kpe.view(-1, BLOCK_SIZE, 1, KPE_DIM), hbm_table, row, logical_slots).float().squeeze(1)
        scores = (case["query"][row].cpu().float() @ selected_ckv.T + case["query_rope"][row].cpu().float() @ selected_kpe.T) * case["scale"]
        golden_rows.append(torch.softmax(scores, dim=-1) @ selected_ckv)
    return torch.stack(golden_rows)


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


@pytest.mark.parametrize("batch,heads,source_len,cache_tokens,tail_tokens", [
    (4, 2, 20992, 8192, 64),
])
def test_fused_copy_sfa_chain(device, batch, heads, source_len, cache_tokens, tail_tokens):
    case = _make_case(device, batch, heads, source_len, cache_tokens, tail_tokens, seed=7)
    _launch_fused(case)
    torch.npu.synchronize()
    golden = _cpu_golden(case)
    actual = case["fused_out"].float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    print(f"FUSED_COPY_SFA_CHECK batch={batch} heads={heads} ok=1", flush=True)
    gc.collect(); torch.npu.empty_cache()


@pytest.mark.parametrize("batch,heads,source_len,cache_tokens,tail_tokens", [
    (4, 2, 20992, 8192, 64),
])
def test_fused_copy_sfa_graph(device, batch, heads, source_len, cache_tokens, tail_tokens):
    case = _make_case(device, batch, heads, source_len, cache_tokens, tail_tokens, seed=7)
    warm_case = _make_case(device, batch, heads, source_len, cache_tokens, tail_tokens, seed=7)
    _launch_fused(warm_case)
    torch.npu.synchronize()
    del warm_case

    graph_kpe = case["fused_kpe"].clone()
    graph_ckv = case["fused_ckv"].clone()
    graph_out = torch.empty_like(case["fused_out"])
    case["fused_kpe"] = graph_kpe
    case["fused_ckv"] = graph_ckv
    case["fused_out"] = graph_out
    # Graph capture cannot perform host-to-device copy inside capture,
    # so stage DRAM KV to NPU before capture.
    case["dram_kpe"] = case["dram_kpe"].to(device)
    case["dram_ckv"] = case["dram_ckv"].to(device)

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        _launch_fused(case)
    torch.npu.synchronize()

    graph_kpe.copy_(case["initial_kpe"])
    graph_ckv.copy_(case["initial_ckv"])
    graph_out.zero_()
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()

    golden = _cpu_golden(case)
    actual = case["fused_out"].float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    print(f"FUSED_COPY_SFA_GRAPH_CHECK batch={batch} heads={heads} ok=1", flush=True)
    gc.collect(); torch.npu.empty_cache()


@pytest.mark.parametrize("batch,heads,source_len,cache_tokens,tail_tokens", [
    (1, 2, 20992, 8192, 64),
    (4, 4, 20992, 8192, 64),
    (4, 2, 20992, 8192, 0),
    (4, 2, 20992, 8192, 129),
])
def test_fused_copy_sfa_parametrize(device, batch, heads, source_len, cache_tokens, tail_tokens):
    case = _make_case(device, batch, heads, source_len, cache_tokens, tail_tokens, seed=7)
    _launch_fused(case)
    torch.npu.synchronize()
    golden = _cpu_golden(case)
    actual = case["fused_out"].float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    print(f"FUSED_COPY_SFA_PARAM batch={batch} heads={heads} tail={tail_tokens} ok=1", flush=True)
    gc.collect(); torch.npu.empty_cache()


@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_copy_sfa_fp16(device, dtype):
    case = _make_case(device, 4, 2, 20992, 8192, 64, seed=7, dtype=dtype)
    _launch_fused(case)
    torch.npu.synchronize()
    golden = _cpu_golden(case)
    actual = case["fused_out"].float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    print(f"FUSED_COPY_SFA_FP16_CHECK ok=1", flush=True)
    gc.collect(); torch.npu.empty_cache()
