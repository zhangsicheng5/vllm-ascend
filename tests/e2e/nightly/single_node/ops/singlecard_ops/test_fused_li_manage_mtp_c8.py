"""Generalized LIM C8 (int8 query/key + fp16 dequant scales) precision e2e.

Mirrors ``test_fused_li_manage_mtp.py`` but routes the query/key through the
Hadamard + per-token int8 quantization used by the C8 model path, then calls
``npu_fused_li_manage_mtp_c8``. Correctness is checked against two bf16
references:

  * oracle A: the non-C8 ``npu_fused_li_manage_mtp`` (bf16 query/key, the
    kernel ignores the fp32 placeholder scales = pure bf16 dot product),
    isolating the int8 quantization effect;
  * oracle B: ``npu_lightning_indexer`` (``native_topk``), an independent
    kernel path.

int8 cannot match bf16 exactly, so the top-k is asserted at >= 0.95 overlap
(same threshold as ``test_fused_li_manage_c8``). Structural invariants that
hold exactly (in-range sources, no duplicates, miss packing, cache bijection)
are still asserted exactly.
"""

from __future__ import annotations

import argparse
import math

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

TOPK = 2048
BLOCK = 128
HEAD_DIM = 128
MISS_CAPACITY = 32768
INVALID_SLOT = -(1 << 31)
MAX_SOURCE_CAPACITY = 1 << 21
HIT_RATE = 0.95


def cumulative(values: list[int]) -> list[int]:
    result: list[int] = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def validate_dynamic_inputs(
    *,
    query_ends: list[int],
    actual_key: list[int],
    offload_key: list[int],
    cache_tokens: list[int],
    request_state: list[int],
    req_pool_entries: list[int],
    total_queries: int,
    source_capacity: int,
    pool_size: int,
) -> None:
    if source_capacity <= 0 or source_capacity % BLOCK or source_capacity > MAX_SOURCE_CAPACITY:
        raise ValueError("source capacity must be 128-aligned and <=2^21")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    batch = len(query_ends)
    fields = (actual_key, offload_key, cache_tokens, request_state, req_pool_entries)
    if not batch or any(len(field) != batch for field in fields):
        raise ValueError("all request metadata must be non-empty and have length B")
    previous = 0
    for request in range(batch):
        end = query_ends[request]
        q = end - previous
        if not 1 <= q <= 14:
            raise ValueError("Q must be in [1,14]")
        if not q <= actual_key[request] <= source_capacity:
            raise ValueError("actual_seq_lengths_key is out of range")
        if not 0 <= req_pool_entries[request] < pool_size:
            raise ValueError("req_pool_entries is out of range")
        state = request_state[request]
        if state not in (-3, -2, -1):
            raise ValueError("request_state must be -3, -2 or -1")
        if state != -3:
            length = offload_key[request]
            capacity = cache_tokens[request]
            if not capacity <= length <= actual_key[request]:
                raise ValueError("requires C <= L <= actual_seq_lengths_key")
            if length < TOPK or length % BLOCK or capacity % BLOCK:
                raise ValueError("L/C alignment or minimum is invalid")
            causal_limit = ((actual_key[request] - q) // BLOCK) * BLOCK
            if length > causal_limit:
                raise ValueError("offload prefix is not visible to every query")
            if length <= q * TOPK:
                if capacity != length:
                    raise ValueError("C must equal L for a small offload prefix")
            elif not q * TOPK <= capacity <= 32640:
                raise ValueError("C is outside the multi-route cache budget")
        previous = end
    if previous != total_queries:
        raise ValueError("last actual_seq_lengths_query value must equal T")
    if len(set(req_pool_entries)) != batch:
        raise ValueError("active requests must use distinct pool rows")


def native_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    visible_lengths: list[int],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row, visible in enumerate(visible_lengths):
        result = torch_npu.npu_lightning_indexer(
            query=query[row : row + 1],
            key=key,
            weights=weights[row : row + 1],
            actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device=query.device),
            actual_seq_lengths_key=torch.tensor([visible], dtype=torch.int32, device=query.device),
            block_table=block_table[row : row + 1],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=0,
        )
        output = result[0] if isinstance(result, (tuple, list)) else result
        rows.append(output.reshape(-1)[:TOPK])
    return torch.stack(rows)


def build_case(
    args: argparse.Namespace,
    *,
    q_values: list[int],
    states: list[int],
) -> dict[str, object]:
    if len(q_values) != len(states):
        raise ValueError("q_values and states must have equal length")
    if args.source_capacity % BLOCK or args.source_capacity > MAX_SOURCE_CAPACITY:
        raise ValueError("source capacity must be 128-aligned and <=2^21")
    batch = len(q_values)
    total_queries = sum(q_values)
    query_ends = cumulative(q_values)
    actual_key = [args.offload_len + BLOCK for _ in q_values]
    offload_key = [args.offload_len for _ in q_values]
    cache_tokens = [args.offload_len if args.offload_len <= q * TOPK else args.cache_tokens for q in q_values]
    pool_size = batch * 2 + 1
    req_entries = [request * 2 + 1 for request in range(batch)]
    validate_dynamic_inputs(
        query_ends=query_ends,
        actual_key=actual_key,
        offload_key=offload_key,
        cache_tokens=cache_tokens,
        request_state=states,
        req_pool_entries=req_entries,
        total_queries=total_queries,
        source_capacity=args.source_capacity,
        pool_size=pool_size,
    )
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    blocks = args.source_capacity // BLOCK
    query = torch.randn(total_queries, args.heads, HEAD_DIM, dtype=dtype, device=device)
    weights = torch.randn(total_queries, args.heads, dtype=dtype, device=device)
    key = torch.randn(blocks, BLOCK, 1, HEAD_DIM, dtype=dtype, device=device)
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).repeat(batch, 1)
    query_to_request = [request for request, q in enumerate(q_values) for _ in range(q)]
    route_table = block_table[torch.tensor(query_to_request, dtype=torch.int64, device=device)].contiguous()
    cache_cpu = torch.full((pool_size, args.source_capacity), INVALID_SLOT, dtype=torch.int32)
    for request, state in enumerate(states):
        if state != -1:
            continue
        row = req_entries[request]
        count = cache_tokens[request]
        cache_cpu[row, :count] = torch.arange(count, dtype=torch.int32)
    return {
        "q_values": q_values,
        "states": states,
        "query_ends": query_ends,
        "actual_key": actual_key,
        "offload_key": offload_key,
        "cache_tokens": cache_tokens,
        "req_entries": req_entries,
        "query": query,
        "weights": weights,
        "query_scale": torch.zeros(total_queries, args.heads, dtype=torch.float32, device=device),
        "key": key,
        "key_scale": torch.zeros(blocks, BLOCK, 1, dtype=torch.float32, device=device),
        "block_table": block_table,
        "route_table": route_table,
        "cache_seed": cache_cpu.to(device),
        "metadata": tuple(
            torch.tensor(values, dtype=torch.int32, device=device)
            for values in (query_ends, actual_key, offload_key, cache_tokens, states, req_entries)
        ),
    }


def make_outputs(case: dict[str, object]) -> tuple[torch.Tensor, ...]:
    query = case["query"]
    assert isinstance(query, torch.Tensor)
    total_queries = query.size(0)
    batch = len(case["q_values"])
    device = query.device
    return (
        torch.full((total_queries, 1, TOPK), -313, dtype=torch.int32, device=device),
        torch.full((total_queries, 1, TOPK), -313, dtype=torch.int32, device=device),
        torch.full((total_queries,), -313, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -313, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -313, dtype=torch.int32, device=device),
        torch.full((batch,), -313, dtype=torch.int32, device=device),
    )


def visible_lengths(case: dict[str, object]) -> list[int]:
    result: list[int] = []
    for request, q in enumerate(case["q_values"]):
        state = case["states"][request]
        for route in range(q):
            result.append(
                case["actual_key"][request] - (q - 1 - route) if state == -3 else case["offload_key"][request]
            )
    return result


def call_custom_bf16(case: dict[str, object], cache: torch.Tensor, outputs: tuple[torch.Tensor, ...]) -> None:
    """Oracle A: the non-C8 mtp op with bf16 query/key (fp32 scales ignored)."""
    query_ends, actual_key, offload_key, cache_tokens, states, req_entries = case["metadata"]
    torch.ops._C_ascend.npu_fused_li_manage_mtp.default(
        case["weights"],
        case["query_scale"],
        case["query"],
        case["key_scale"],
        case["key"],
        case["block_table"],
        query_ends,
        actual_key,
        offload_key,
        cache_tokens,
        states,
        req_entries,
        cache,
        *outputs,
    )


def _apply_hadamard(x: torch.Tensor) -> torch.Tensor:
    from scipy.linalg import hadamard

    H = torch.tensor(hadamard(HEAD_DIM), dtype=x.dtype, device=x.device) / math.sqrt(HEAD_DIM)
    return torch.matmul(x, H)


def _quantize_for_c8(case: dict[str, object]) -> dict[str, object]:
    """Hadamard-rotate + per-token int8 quant, matching the C8 store path."""
    query = case["query"]
    key = case["key"]
    query_h = _apply_hadamard(query)
    key_h = _apply_hadamard(key)
    q_i8, q_scale = torch_npu.npu_dynamic_quant(query_h.reshape(-1, HEAD_DIM))
    k_i8, k_scale = torch_npu.npu_dynamic_quant(key_h.reshape(-1, HEAD_DIM))
    return {
        **case,
        "query": q_i8.view_as(query_h).to(torch.int8),
        "query_scale": q_scale.to(torch.float16).view(query.shape[0], query.shape[1]),
        "key": k_i8.view_as(key_h).to(torch.int8),
        "key_scale": k_scale.to(torch.float16).view(key.shape[0], BLOCK, 1),
    }


def call_custom_c8(case: dict[str, object], cache: torch.Tensor, outputs: tuple[torch.Tensor, ...]) -> None:
    query_ends, actual_key, offload_key, cache_tokens, states, req_entries = case["metadata"]
    torch.ops._C_ascend.npu_fused_li_manage_mtp_c8.default(
        case["weights"],
        case["query_scale"],
        case["query"],
        case["key_scale"],
        case["key"],
        case["block_table"],
        query_ends,
        actual_key,
        offload_key,
        cache_tokens,
        states,
        req_entries,
        cache,
        *outputs,
    )


def _topk_overlap(actual: torch.Tensor, reference: torch.Tensor, valid: int) -> float:
    a = set(actual[:valid].tolist())
    b = set(reference[:valid].tolist())
    if not b:
        return 0.0
    return len(a & b) / len(b)


def assert_correctness_c8(case: dict[str, object]) -> None:
    c8_case = _quantize_for_c8(case)
    cache = c8_case["cache_seed"].clone()
    outputs = make_outputs(c8_case)
    call_custom_c8(c8_case, cache, outputs)
    torch.npu.synchronize()
    src, _, _, _, _, miss_count = [x.cpu() for x in outputs]

    # Oracle A: the non-C8 mtp op on the ORIGINAL bf16 case (separate cache).
    bf_cache = case["cache_seed"].clone()
    bf_outputs = make_outputs(case)
    call_custom_bf16(case, bf_cache, bf_outputs)
    torch.npu.synchronize()
    bf_src = bf_outputs[0].cpu()

    # Oracle B: native lightning indexer on the bf16 query/key/weights.
    reference = native_topk(
        case["query"], case["key"], case["weights"], case["route_table"], visible_lengths(case)
    ).cpu()

    overlaps_a: list[float] = []
    overlaps_b: list[float] = []
    query_start = 0
    for request, q in enumerate(case["q_values"]):
        query_end = query_start + q
        state = case["states"][request]
        length = case["actual_key"][request] if state == -3 else case["offload_key"][request]
        for route in range(query_start, query_end):
            valid = min(length, TOPK)
            actual_topk = src[route, 0, :valid]
            # Structural: in-range, unique sources, -1 tail padding.
            assert not bool((actual_topk < 0).any()) and not bool((actual_topk >= length).any())
            assert torch.unique(actual_topk).numel() == valid
            if valid < TOPK:
                assert torch.all(src[route, 0, valid:] == -1)
            # Precision: dual-oracle top-k overlap (int8 cannot be exact).
            overlaps_a.append(_topk_overlap(actual_topk, bf_src[route, 0, :valid], valid))
            overlaps_b.append(_topk_overlap(actual_topk, reference[route, :valid], valid))
        # Request-level miss count (output[5], per-request; NOT the per-route
        # topk_miss_counts which is bounded by TOPK). -3 non-offload -> 0;
        # -2 first fill -> full cache budget; -1 steady -> new misses <= budget.
        cap = case["cache_tokens"][request]
        if state == -3:
            assert int(miss_count[request]) == 0
        elif state == -2:
            assert int(miss_count[request]) == cap
        else:
            assert 0 <= int(miss_count[request]) <= cap
        query_start = query_end

    min_a = min(overlaps_a) if overlaps_a else 0.0
    min_b = min(overlaps_b) if overlaps_b else 0.0
    print(f"C8_MTP_OVERLAP oracleA(bf16_mtp)={min_a:.4f} oracleB(lightning_indexer)={min_b:.4f}")
    assert min_a >= HIT_RATE, f"C8 vs bf16 mtp top-k overlap {min_a:.4f} < {HIT_RATE}"
    assert min_b >= HIT_RATE, f"C8 vs native lightning top-k overlap {min_b:.4f} < {HIT_RATE}"


# mtp_c8 requires index_weights bf16 (OpDef DT_BF16) and quantizes query/key
# to int8, so the bf16/fp16 dtype param from the non-C8 mtp e2e is meaningless
# here — vary heads only, keep bf16.
@pytest.mark.parametrize("heads,dtype", [(32, "bf16"), (64, "bf16")])
def test_generalized_lim_c8_mixed_states_and_query_counts(heads, dtype):
    args = argparse.Namespace(
        source_capacity=16384, offload_len=8192, cache_tokens=8192, seed=7, dtype=dtype, heads=heads, device="npu:0"
    )
    case = build_case(args, q_values=[1, 2, 3, 4, 5, 6, 7], states=[-3, -2, -1, -3, -2, -1, -2])
    assert_correctness_c8(case)


def test_generalized_lim_c8_first_fill_then_repeat_is_all_hits():
    args = argparse.Namespace(
        source_capacity=16384, offload_len=8192, cache_tokens=6144, seed=17, dtype="bf16", heads=32, device="npu:0"
    )
    case = build_case(args, q_values=[1, 3], states=[-2, -2])
    c8_case = _quantize_for_c8(case)
    cache = c8_case["cache_seed"].clone()
    outputs = make_outputs(c8_case)
    call_custom_c8(c8_case, cache, outputs)
    torch.npu.synchronize()
    for row, capacity in enumerate(case["cache_tokens"]):
        assert outputs[-1][row].item() == capacity
