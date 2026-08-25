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
"""E2E tests for the ``npu_fused_li_manage_mtp`` custom Ascend operator.

Migrated from ``nanovllm-DSA-offload/ut_ops/test_fused_li_manage_mtp.py``.

Behavior contract (from FUSED_LI_MANAGE_BEHAVIOR.md):
  ``torch_npu.npu_lightning_indexer`` generates the Top-2048 reference; the
  fused ``npu_fused_li_manage_mtp`` operator performs sparse selection, hit
  detection, eviction and request-pool state update in one kernel.

  Interface constraints:
    - Caller pre-allocates ``topk_dst_slots``, ``topk_src_ids``,
      ``miss_src_ids``, ``miss_dst_slots`` and ``miss_counts``.
    - ``cache_slots_pool`` and the five output buffers are written in-place;
      the op returns ``None`` (no allocating / ``_out`` entry / alias output).
    - ``req_pool_entries`` maps active requests to persistent pool rows;
      unmapped rows must be left unchanged.

  Single-request behavior:
    - ``topk_src_ids`` contains the same 2048 unique token IDs as the
      LightningIndexer reference.
    - ``miss_counts`` equals the miss count recomputed from the pre-update
      cache state.
    - ``topk_dst_slots`` are all in ``[0, C)`` and unique.
    - Existing hit tokens keep their slot; after update
      ``new_cache_slots[topk_src_ids[i]] == topk_dst_slots[i]``.
    - The valid slots after update still uniquely cover ``0..C-1``.
    - A second update with the same query yields ``miss_counts == 0``.
    - Order inside the miss/hit segments is unconstrained.

  Boundary & timing:
    - Performance data is only printed; latency thresholds do NOT decide
      test pass/fail.
    - Cache state is restored before timing, and restore+sync are excluded
      from the measured op latency.
"""

from __future__ import annotations

import gc
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch_npu  # type: ignore

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = False
enable_custom_op()

QUERY_COUNT = 4
HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 2048
UNION_CAPACITY = QUERY_COUNT * TOPK
MAX_SOURCE_CAPACITY = 1 << 18


@dataclass
class MtpCase:
    name: str
    device: torch.device
    dtype: torch.dtype
    batch_size: int
    source_capacity: int
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    req_pool_entries: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor
    req_pool_entries_cpu: torch.Tensor
    cache_tokens_cpu: torch.Tensor
    candidate_lens_cpu: torch.Tensor
    block_table_cpu: torch.Tensor
    initial_cache_cpu: torch.Tensor
    topk_cpu: list[torch.Tensor]
    union_cpu: list[torch.Tensor]


def _random_block_table(
    batch_size: int,
    blocks_per_request: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    total_blocks = batch_size * blocks_per_request
    table = torch.randperm(total_blocks, generator=generator).to(torch.int32)
    return table.view(batch_size, blocks_per_request).contiguous(), total_blocks


def _ordered_union(rows: list[torch.Tensor]) -> torch.Tensor:
    ordered: list[int] = []
    seen: set[int] = set()
    for row in rows:
        for value in row.tolist():
            token = int(value)
            if token not in seen:
                seen.add(token)
                ordered.append(token)
    return torch.tensor(ordered, dtype=torch.int64)


def _call_native_li(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    query_ends: torch.Tensor,
) -> torch.Tensor:
    result = torch.ops._C_ascend.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=query_ends,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
        return_value=False,
    )
    topk = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(topk, torch.Tensor):
        raise TypeError("native LightningIndexer did not return a Tensor")
    return topk


def _native_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    cache_tokens_cpu: torch.Tensor,
) -> list[torch.Tensor]:
    """Use native LightningIndexer as the four independent-query golden."""

    batch_size = int(candidate_lens.numel())
    active_query_rows: list[int] = []
    active_request_rows: list[int] = []
    candidate_cpu = candidate_lens.cpu()
    for request in range(batch_size):
        if int(cache_tokens_cpu[request]) == 0:
            continue
        if int(candidate_cpu[request]) < TOPK:
            raise AssertionError("active MTP LIM rows must have >=2048 candidates")
        for query_idx in range(QUERY_COUNT):
            active_query_rows.append(request * QUERY_COUNT + query_idx)
            active_request_rows.append(request)

    result_rows = [torch.empty(0, dtype=torch.int64) for _ in range(batch_size * QUERY_COUNT)]
    if not active_query_rows:
        return result_rows

    query_index = torch.tensor(active_query_rows, dtype=torch.int64, device=query.device)
    request_index = torch.tensor(active_request_rows, dtype=torch.int64, device=query.device)
    active_query = query.index_select(0, query_index).contiguous()
    active_weights = weights.index_select(0, query_index).contiguous()
    active_table = block_table.index_select(0, request_index).contiguous()
    active_lens = candidate_lens.index_select(0, request_index).contiguous()
    query_ends = torch.arange(
        1,
        len(active_query_rows) + 1,
        dtype=torch.int32,
        device=query.device,
    )
    topk = _call_native_li(
        active_query,
        key,
        active_weights,
        active_table,
        active_lens,
        query_ends,
    )
    expected_shape = (len(active_query_rows), 1, TOPK)
    if tuple(topk.shape) != expected_shape:
        raise AssertionError(
            f"native LightningIndexer shape={tuple(topk.shape)}, "
            f"expected={expected_shape}"
        )
    topk_cpu = topk.reshape(len(active_query_rows), TOPK).cpu().to(torch.int64)
    for local_row, query_row in enumerate(active_query_rows):
        result_rows[query_row] = topk_cpu[local_row].contiguous()
    return result_rows


def _make_cache_state(
    *,
    topk_rows: list[torch.Tensor],
    candidate_lens: tuple[int, ...],
    cache_tokens: tuple[int, ...],
    req_pool_entries: torch.Tensor,
    source_capacity: int,
    miss_fractions: tuple[float, ...],
    generator: torch.Generator,
    pool_size: int,
    exact_miss_counts: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if exact_miss_counts is not None and len(exact_miss_counts) != len(
        candidate_lens
    ):
        raise ValueError("exact miss counts must match the batch size")
    state = torch.full((pool_size, source_capacity), -777, dtype=torch.int32)
    unions: list[torch.Tensor] = []
    for request, (candidate_len, budget, miss_fraction) in enumerate(
        zip(candidate_lens, cache_tokens, miss_fractions)
    ):
        pool_row = int(req_pool_entries[request])
        state[pool_row].fill_(-1)
        if budget == 0:
            if exact_miss_counts is not None and exact_miss_counts[request] != 0:
                raise ValueError("C=0 rows require exact miss_count=0")
            unions.append(torch.empty(0, dtype=torch.int64))
            continue

        request_topk = topk_rows[
            request * QUERY_COUNT : (request + 1) * QUERY_COUNT
        ]
        union = _ordered_union(request_topk)
        unions.append(union)
        if union.numel() > budget:
            raise AssertionError(
                f"request={request}: union={union.numel()} exceeds C={budget}"
            )

        if budget == candidate_len:
            if exact_miss_counts is not None and exact_miss_counts[request] != 0:
                raise ValueError("fully cached rows require exact miss_count=0")
            cached = torch.arange(candidate_len, dtype=torch.int64)
        else:
            if exact_miss_counts is None:
                miss_count = min(
                    int(round(float(union.numel()) * miss_fraction)),
                    int(union.numel()),
                )
                hits = union[miss_count:]
            else:
                miss_count = int(exact_miss_counts[request])
                if miss_count < 0 or miss_count > int(union.numel()):
                    raise ValueError(
                        f"request={request}: exact miss_count={miss_count} "
                        f"must be in [0,{union.numel()}]"
                    )
                occurrences = torch.bincount(
                    torch.cat(request_topk), minlength=candidate_len
                )
                repeated = union[occurrences[union] > 1]
                single = union[occurrences[union] == 1]
                misses = torch.cat((repeated, single))[:miss_count]
                miss_mask = torch.zeros(candidate_len, dtype=torch.bool)
                miss_mask[misses] = True
                hits = union[~miss_mask[union]]
            union_mask = torch.zeros(candidate_len, dtype=torch.bool)
            union_mask[union] = True
            fillers = torch.arange(candidate_len, dtype=torch.int64)[~union_mask]
            needed = budget - int(hits.numel())
            if needed < 0 or fillers.numel() < needed:
                raise AssertionError(
                    f"cannot construct request={request} C={budget} state"
                )
            cached = torch.cat((hits, fillers[:needed]))

        if cached.numel() != budget or torch.unique(cached).numel() != budget:
            raise AssertionError("initial cache tokens must be unique and exactly C")
        slot_permutation = torch.randperm(budget, generator=generator).to(torch.int32)
        state[pool_row, cached] = slot_permutation
    return state.contiguous(), unions


def _make_balanced_mtp_cache_state(
    *,
    topk_rows: list[torch.Tensor],
    candidate_lens: tuple[int, ...],
    cache_tokens: tuple[int, ...],
    req_pool_entries: torch.Tensor,
    source_capacity: int,
    per_query_miss_count: int,
    generator: torch.Generator,
    pool_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build the MTP3 performance state without redefining miss as union miss."""

    state = torch.full((pool_size, source_capacity), -777, dtype=torch.int32)
    unions: list[torch.Tensor] = []
    for request, (candidate_len, budget) in enumerate(
        zip(candidate_lens, cache_tokens)
    ):
        pool_row = int(req_pool_entries[request])
        state[pool_row].fill_(-1)
        rows = topk_rows[request * QUERY_COUNT : (request + 1) * QUERY_COUNT]
        union = _ordered_union(rows)
        unions.append(union)
        if budget == 0:
            if per_query_miss_count:
                raise ValueError("C=0 rows cannot have performance misses")
            continue
        if union.numel() > budget:
            raise AssertionError(
                f"request={request}: TopK union={union.numel()} exceeds C={budget}"
            )
        if budget == candidate_len and per_query_miss_count:
            raise ValueError("fully cached rows cannot have performance misses")

        membership = torch.zeros(candidate_len, dtype=torch.uint8)
        for query_idx, row in enumerate(rows):
            membership[row] |= 1 << query_idx
        available_by_mask = {
            mask: torch.nonzero(membership == mask).flatten().to(torch.int64)
            for mask in range(1, 1 << QUERY_COUNT)
        }
        selected_parts: list[torch.Tensor] = []
        common = available_by_mask[0b1111]
        common_count = min(per_query_miss_count // 2, int(common.numel()))
        selected_parts.append(common[:common_count])
        pair_degree = per_query_miss_count - common_count
        opposite_pair_masks = (
            (0b0011, 0b1100),
            (0b0101, 0b1010),
            (0b1001, 0b0110),
        )
        capacities = [
            min(
                int(available_by_mask[left].numel()),
                int(available_by_mask[right].numel()),
            )
            for left, right in opposite_pair_masks
        ]
        if sum(capacities) < pair_degree:
            raise AssertionError(
                f"request={request}: pair-overlap capacity={capacities} cannot "
                f"supply degree={pair_degree}; adjust perf_query_noise"
            )
        pair_counts = [min(pair_degree // 3, capacity) for capacity in capacities]
        remaining = pair_degree - sum(pair_counts)
        while remaining:
            progressed = False
            for idx, capacity in enumerate(capacities):
                if pair_counts[idx] < capacity:
                    pair_counts[idx] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                raise AssertionError("failed to distribute pair-overlap misses")
        wanted_by_mask: dict[int, int] = {}
        for (left, right), count in zip(opposite_pair_masks, pair_counts):
            wanted_by_mask[left] = count
            wanted_by_mask[right] = count
        for mask, wanted in wanted_by_mask.items():
            available = available_by_mask[mask]
            selected_parts.append(available[:wanted])
        selected_misses = torch.cat(selected_parts)
        if torch.unique(selected_misses).numel() != selected_misses.numel():
            raise AssertionError("balanced MTP miss construction produced duplicates")

        per_query_counts = [
            int(torch.isin(selected_misses, row).sum()) for row in rows
        ]
        if per_query_counts != [per_query_miss_count] * QUERY_COUNT:
            raise AssertionError(
                f"request={request}: constructed per-query misses="
                f"{per_query_counts}, expected={per_query_miss_count}"
            )

        missing_mask = torch.zeros(candidate_len, dtype=torch.bool)
        missing_mask[selected_misses] = True
        hits = union[~missing_mask[union]]
        union_mask = torch.zeros(candidate_len, dtype=torch.bool)
        union_mask[union] = True
        fillers = torch.arange(candidate_len, dtype=torch.int64)[~union_mask]
        needed = budget - int(hits.numel())
        if needed < 0 or fillers.numel() < needed:
            raise AssertionError(
                f"request={request}: cannot build C={budget} cache with "
                f"{selected_misses.numel()} selected misses"
            )
        cached = torch.cat((hits, fillers[:needed]))
        if cached.numel() != budget or torch.unique(cached).numel() != budget:
            raise AssertionError("balanced initial cache must contain exactly C tokens")
        state[pool_row, cached] = torch.randperm(
            budget, generator=generator
        ).to(torch.int32)

    return state.contiguous(), unions


def make_case(
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    candidate_lens: tuple[int, ...],
    cache_tokens: tuple[int, ...],
    miss_fractions: tuple[float, ...],
    seed: int,
    source_capacity: int | None = None,
    exact_miss_counts: tuple[int, ...] | None = None,
    correlated_query_noise: float | None = None,
    balanced_query_miss_count: int | None = None,
) -> MtpCase:
    batch_size = len(candidate_lens)
    if not (
        len(cache_tokens) == batch_size == len(miss_fractions)
        and batch_size > 0
    ):
        raise ValueError("candidate/cache/miss tuples must have equal nonzero length")
    capacity = source_capacity or max(candidate_lens)
    if capacity % BLOCK_SIZE or capacity > MAX_SOURCE_CAPACITY:
        raise ValueError("source capacity must be block aligned and <=2^18")
    if any(length <= 0 or length > capacity or length % BLOCK_SIZE for length in candidate_lens):
        raise ValueError("candidate lengths must be positive, aligned, and <= capacity")
    for length, budget in zip(candidate_lens, cache_tokens):
        if budget == 0:
            continue
        if budget < min(length, UNION_CAPACITY) or budget > length:
            raise ValueError(
                f"active row requires min(candidate,8192)<=C<=candidate, got {length=}, {budget=}"
            )

    generator = torch.Generator().manual_seed(seed)
    block_table_cpu, physical_blocks = _random_block_table(
        batch_size, capacity // BLOCK_SIZE, generator
    )
    req_entries_cpu = torch.randperm(batch_size + 3, generator=generator)[
        :batch_size
    ].to(torch.int32)
    candidate_cpu = torch.tensor(candidate_lens, dtype=torch.int32)
    cache_tokens_cpu = torch.tensor(cache_tokens, dtype=torch.int32)

    if correlated_query_noise is None:
        query_cpu = torch.randn(
            batch_size * QUERY_COUNT,
            HEADS,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        weights_cpu = torch.rand(
            batch_size * QUERY_COUNT,
            HEADS,
            generator=generator,
            dtype=torch.float32,
        )
    else:
        if correlated_query_noise <= 0:
            raise ValueError("correlated query noise must be positive")
        base_query = torch.randn(
            batch_size, 1, HEADS, HEAD_DIM, generator=generator
        )
        query_noise = torch.randn(
            batch_size, QUERY_COUNT, HEADS, HEAD_DIM, generator=generator
        )
        query_cpu = (
            base_query + correlated_query_noise * query_noise
        ).reshape(batch_size * QUERY_COUNT, HEADS, HEAD_DIM)
        base_weights = torch.rand(
            batch_size, 1, HEADS, generator=generator, dtype=torch.float32
        )
        weights_cpu = base_weights.expand(-1, QUERY_COUNT, -1).reshape(
            batch_size * QUERY_COUNT, HEADS
        ).contiguous()
    query_cpu = query_cpu.to(dtype)
    weights_cpu = weights_cpu.to(dtype)
    torch.manual_seed(seed + 991)
    key = torch.randn(
        physical_blocks,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    query = query_cpu.to(device)
    weights = weights_cpu.to(device)
    block_table = block_table_cpu.to(device)
    candidate = candidate_cpu.to(device)
    topk_rows = _native_topk(
        query,
        key,
        weights,
        block_table,
        candidate,
        cache_tokens_cpu,
    )
    if balanced_query_miss_count is None:
        cache_cpu, union_rows = _make_cache_state(
            topk_rows=topk_rows,
            candidate_lens=candidate_lens,
            cache_tokens=cache_tokens,
            req_pool_entries=req_entries_cpu,
            source_capacity=capacity,
            miss_fractions=miss_fractions,
            generator=generator,
            pool_size=batch_size + 3,
            exact_miss_counts=exact_miss_counts,
        )
    else:
        if exact_miss_counts is not None:
            raise ValueError(
                "balanced per-query misses and exact union misses are exclusive"
            )
        cache_cpu, union_rows = _make_balanced_mtp_cache_state(
            topk_rows=topk_rows,
            candidate_lens=candidate_lens,
            cache_tokens=cache_tokens,
            req_pool_entries=req_entries_cpu,
            source_capacity=capacity,
            per_query_miss_count=balanced_query_miss_count,
            generator=generator,
            pool_size=batch_size + 3,
        )
    return MtpCase(
        name=name,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        source_capacity=capacity,
        query=query,
        key=key,
        weights=weights,
        req_pool_entries=req_entries_cpu.to(device),
        cache_tokens=cache_tokens_cpu.to(device),
        candidate_lens=candidate,
        block_table=block_table,
        req_pool_entries_cpu=req_entries_cpu,
        cache_tokens_cpu=cache_tokens_cpu,
        candidate_lens_cpu=candidate_cpu,
        block_table_cpu=block_table_cpu,
        initial_cache_cpu=cache_cpu,
        topk_cpu=topk_rows,
        union_cpu=union_rows,
    )


def call_mtp(case: MtpCase, cache_slots: torch.Tensor):
    outputs = make_outputs(case)
    call_mtp_with_buffers(case, cache_slots, *outputs)
    return outputs


def call_mtp_with_buffers(
    case: MtpCase,
    cache_slots: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_source_ids: torch.Tensor,
    miss_source_ids: torch.Tensor,
    miss_destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    torch.ops._C_ascend.npu_fused_li_manage_mtp(
        case.query,
        case.weights,
        case.key,
        case.block_table,
        case.candidate_lens,
        case.cache_tokens,
        case.req_pool_entries,
        cache_slots,
        topk_source_ids,
        topk_slots,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
    )
    return (
        topk_slots,
        topk_source_ids,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
    )


def make_outputs(case: MtpCase) -> tuple[torch.Tensor, ...]:
    topk_slots = torch.full(
        (case.batch_size * QUERY_COUNT, 1, TOPK),
        -313,
        dtype=torch.int32,
        device=case.device,
    )
    topk_source_ids = torch.full_like(topk_slots, -313)
    miss_sources = torch.full(
        (case.batch_size, UNION_CAPACITY),
        -313,
        dtype=torch.int32,
        device=case.device,
    )
    miss_destinations = torch.full_like(miss_sources, -313)
    miss_counts = torch.full(
        (case.batch_size,), -313, dtype=torch.int32, device=case.device
    )
    return (
        topk_slots,
        topk_source_ids,
        miss_sources,
        miss_destinations,
        miss_counts,
    )


def validate_result(
    case: MtpCase,
    before_cpu: torch.Tensor,
    cache_slots: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> list[int]:
    (
        topk_slots,
        topk_source_ids,
        miss_sources,
        miss_destinations,
        miss_counts,
    ) = outputs
    after_cpu = cache_slots.cpu()
    topk_slots_cpu = topk_slots.reshape(-1, TOPK).cpu().to(torch.int64)
    topk_source_ids_cpu = (
        topk_source_ids.reshape(-1, TOPK).cpu().to(torch.int64)
    )
    sources_cpu = miss_sources.cpu().to(torch.int64)
    destinations_cpu = miss_destinations.cpu().to(torch.int64)
    counts_cpu = miss_counts.cpu().to(torch.int64)
    active_pool_rows = set(int(value) for value in case.req_pool_entries_cpu.tolist())

    for pool_row in range(before_cpu.shape[0]):
        if pool_row not in active_pool_rows and not torch.equal(
            before_cpu[pool_row], after_cpu[pool_row]
        ):
            raise AssertionError(f"{label}: unused pool row {pool_row} changed")

    expected_counts: list[int] = []
    for request in range(case.batch_size):
        pool_row = int(case.req_pool_entries_cpu[request])
        budget = int(case.cache_tokens_cpu[request])
        candidate_len = int(case.candidate_lens_cpu[request])
        before = before_cpu[pool_row]
        after = after_cpu[pool_row]
        if budget == 0:
            expected_counts.append(0)
            assert int(counts_cpu[request]) == 0, f"{label}: C=0 row has nonzero miss count"
            assert torch.equal(before, after), f"{label}: C=0 pool row changed"
            continue

        valid_slots = after[:candidate_len]
        valid_slots = valid_slots[valid_slots >= 0].to(torch.int64)
        assert (
            valid_slots.numel() == budget
            and torch.unique(valid_slots).numel() == budget
            and int(valid_slots.min()) == 0
            and int(valid_slots.max()) == budget - 1
        ), f"{label}: request={request} state is not a permutation of [0,C)"

        cached_tokens = torch.nonzero(after[:candidate_len] >= 0).flatten()
        slot_to_token = torch.full((budget,), -1, dtype=torch.int64)
        slot_to_token[after[cached_tokens].to(torch.int64)] = cached_tokens
        actual_topk_rows: list[torch.Tensor] = []
        for query_idx in range(QUERY_COUNT):
            query_row = request * QUERY_COUNT + query_idx
            actual_slots = topk_slots_cpu[query_row]
            assert (
                not bool((actual_slots < 0).any())
                and not bool((actual_slots >= budget).any())
                and torch.unique(actual_slots).numel() == TOPK
            ), (
                f"{label}: request={request} query={query_idx} slots invalid "
                f"(negative={int((actual_slots < 0).sum())}, "
                f"out_of_range={int((actual_slots >= budget).sum())}, "
                f"unique={int(torch.unique(actual_slots).numel())}/{TOPK}, "
                f"min={int(actual_slots.min())}, max={int(actual_slots.max())})"
            )
            actual_tokens = slot_to_token[actual_slots]
            expected_sources = torch.where(
                before[actual_tokens] < 0,
                actual_tokens,
                torch.full_like(actual_tokens, -1),
            )
            assert torch.equal(
                topk_source_ids_cpu[query_row], expected_sources
            ), f"{label}: request={request} query={query_idx} aligned source IDs differ"
            golden_tokens = case.topk_cpu[query_row]
            assert torch.equal(
                torch.sort(actual_tokens).values,
                torch.sort(golden_tokens).values,
            ), (
                f"{label}: request={request} query={query_idx} topk token "
                f"set differs (missing={int(torch.isin(golden_tokens, actual_tokens, invert=True).sum())}, "
                f"extra={int(torch.isin(actual_tokens, golden_tokens, invert=True).sum())})"
            )
            actual_topk_rows.append(actual_tokens)

        union = _ordered_union(actual_topk_rows)
        golden_union = case.union_cpu[request]
        assert torch.equal(
            torch.sort(union).values,
            torch.sort(golden_union).values,
        ), f"{label}: request={request} topk union set differs"
        assert not bool((after[union] < 0).any()), f"{label}: request={request} union is not cached"

        expected_misses = union[before[union] < 0]
        expected_count = int(expected_misses.numel())
        expected_counts.append(expected_count)
        actual_count = int(counts_cpu[request])
        assert actual_count == expected_count, (
            f"{label}: request={request} miss_count={actual_count}, "
            f"expected={expected_count}"
        )
        assert torch.equal(
            sources_cpu[request, :actual_count], expected_misses
        ), f"{label}: request={request} ordered union misses differ"
        active_destinations = destinations_cpu[request, :actual_count]
        if actual_count:
            assert (
                not bool((active_destinations < 0).any())
                and not bool((active_destinations >= budget).any())
                and torch.unique(active_destinations).numel() == actual_count
            ), f"{label}: request={request} miss destination slots invalid"

        old_hits = union[before[union] >= 0]
        if old_hits.numel():
            assert torch.equal(after[old_hits], before[old_hits]), (
                f"{label}: request={request} changed an existing hit slot"
            )
        if actual_count:
            assert torch.equal(
                after[expected_misses].to(torch.int64), active_destinations
            ), f"{label}: request={request} miss-to-slot mapping is wrong"

        old_tokens = torch.nonzero(before[:candidate_len] >= 0).flatten()
        evicted = old_tokens[after[old_tokens] < 0]
        if evicted.numel():
            union_mask = torch.zeros(candidate_len, dtype=torch.bool)
            union_mask[union] = True
            assert not bool(union_mask[evicted].any()), (
                f"{label}: request={request} evicted a union token"
            )

    return expected_counts


def _compare_valid_outputs(
    case: MtpCase,
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> None:
    left_topk = left[0].reshape(case.batch_size, QUERY_COUNT, TOPK).cpu()
    right_topk = right[0].reshape(case.batch_size, QUERY_COUNT, TOPK).cpu()
    left_topk_sources = left[1].reshape(
        case.batch_size, QUERY_COUNT, TOPK
    ).cpu()
    right_topk_sources = right[1].reshape(
        case.batch_size, QUERY_COUNT, TOPK
    ).cpu()
    for request in range(case.batch_size):
        if int(case.cache_tokens_cpu[request]) == 0:
            continue
        assert torch.equal(left_topk[request], right_topk[request]), (
            f"{label}: request={request} topk_slots differ"
        )
        assert torch.equal(
            left_topk_sources[request], right_topk_sources[request]
        ), f"{label}: request={request} topk_source_ids differ"
    left_counts = left[4].cpu()
    right_counts = right[4].cpu()
    assert torch.equal(left_counts, right_counts), f"{label}: miss_counts differ"
    for request, count_value in enumerate(left_counts.tolist()):
        count = int(count_value)
        assert torch.equal(
            left[2][request, :count].cpu(), right[2][request, :count].cpu()
        ), f"{label}: request={request} miss IDs differ"
        assert torch.equal(
            left[3][request, :count].cpu(), right[3][request, :count].cpu()
        ), f"{label}: request={request} miss slots differ"


def _event_us(
    runner: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    reset: Callable[[], None] | None = None,
) -> float:
    """Measure only NPU work; mutable request-state reset is not timed."""

    for _ in range(warmup):
        if reset is not None:
            reset()
        runner()
    torch.npu.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))
    return statistics.mean(samples_ms) * 1000.0


def _assert_topk_sets(
    actual: torch.Tensor,
    expected_rows: list[torch.Tensor],
    *,
    label: str,
) -> None:
    actual_cpu = actual.reshape(-1, TOPK).cpu().to(torch.int64)
    expected = torch.stack(expected_rows)
    assert torch.equal(
        torch.sort(actual_cpu, dim=1).values,
        torch.sort(expected, dim=1).values,
    ), f"{label}: TopK sets differ from native golden"


def _query_miss_counts(case: MtpCase) -> tuple[list[int], list[int]]:
    per_query_totals = [0] * QUERY_COUNT
    union_counts: list[int] = []
    for request in range(case.batch_size):
        pool_row = int(case.req_pool_entries_cpu[request])
        before = case.initial_cache_cpu[pool_row]
        rows = case.topk_cpu[
            request * QUERY_COUNT : (request + 1) * QUERY_COUNT
        ]
        for query_idx, row in enumerate(rows):
            per_query_totals[query_idx] += int((before[row] < 0).sum())
        union = case.union_cpu[request]
        union_counts.append(int((before[union] < 0).sum()))
    return per_query_totals, union_counts


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


# ---------------------------------------------------------------------------
# Meta check (in-place op returns None; buffers keep caller shapes/dtypes)
# ---------------------------------------------------------------------------


def test_meta_check():
    batch_size = 2
    source_capacity = 8192
    meta = torch.device("meta")
    query = torch.empty(
        batch_size * QUERY_COUNT, HEADS, HEAD_DIM, device=meta, dtype=torch.bfloat16
    )
    key = torch.empty(
        batch_size * source_capacity // BLOCK_SIZE,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        device=meta,
        dtype=torch.bfloat16,
    )
    weights = torch.empty(
        batch_size * QUERY_COUNT, HEADS, device=meta, dtype=torch.bfloat16
    )
    req_entries = torch.empty(batch_size, device=meta, dtype=torch.int32)
    cache_slots = torch.empty(
        batch_size + 1, source_capacity, device=meta, dtype=torch.int32
    )
    cache_tokens = torch.empty(batch_size, device=meta, dtype=torch.int32)
    candidate_lens = torch.empty(batch_size, device=meta, dtype=torch.int32)
    block_table = torch.empty(
        batch_size,
        source_capacity // BLOCK_SIZE,
        device=meta,
        dtype=torch.int32,
    )
    expected_shapes = (
        (batch_size * QUERY_COUNT, 1, TOPK),
        (batch_size * QUERY_COUNT, 1, TOPK),
        (batch_size, UNION_CAPACITY),
        (batch_size, UNION_CAPACITY),
        (batch_size,),
    )
    buffers = (
        torch.empty(expected_shapes[0], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[1], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[2], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[3], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[4], device=meta, dtype=torch.int32),
    )
    result = torch.ops._C_ascend.npu_fused_li_manage_mtp(
        query,
        weights,
        key,
        block_table,
        candidate_lens,
        cache_tokens,
        req_entries,
        cache_slots,
        buffers[1],
        buffers[0],
        buffers[2],
        buffers[3],
        buffers[4],
    )
    assert result is None, "Fused LIM Manage MTP must return None"
    assert (
        tuple(tuple(output.shape) for output in buffers) == expected_shapes
    ), "Fused LIM Manage MTP Meta buffers changed shape"
    assert all(
        output.dtype == torch.int32 for output in buffers
    ), "Fused LIM Manage MTP Meta buffers changed dtype"


# ---------------------------------------------------------------------------
# Semantic checks: fresh / persistent / repeat in-place contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_semantic_mixed(device, dtype):
    case = make_case(
        name="mixed_b6",
        device=device,
        dtype=dtype,
        candidate_lens=(1024, 4096, 8192, 20992, 32768, 65536),
        cache_tokens=(0, 4096, 8192, 8192, 12288, 12288),
        miss_fractions=(0.0, 0.0, 0.0, 0.02, 0.5, 1.0),
        seed=7,
    )
    _run_semantic_case(case)
    del case
    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_semantic_fp16_full_source(device, dtype):
    case = make_case(
        name="full_source_b1",
        device=device,
        dtype=dtype,
        candidate_lens=(8192,),
        cache_tokens=(8192,),
        miss_fractions=(0.0,),
        seed=1007,
    )
    _run_semantic_case(case)
    del case
    gc.collect()
    torch.npu.empty_cache()


def _run_semantic_case(case: MtpCase) -> None:
    fresh_cache = case.initial_cache_cpu.to(case.device)
    before = case.initial_cache_cpu.clone()
    fresh_outputs = call_mtp(case, fresh_cache)
    torch.npu.synchronize()
    counts = validate_result(
        case, before, fresh_cache, fresh_outputs, label=f"{case.name}/fresh"
    )

    persistent_cache = case.initial_cache_cpu.to(case.device)
    persistent_buffers = make_outputs(case)
    persistent_outputs = call_mtp_with_buffers(
        case, persistent_cache, *persistent_buffers
    )
    torch.npu.synchronize()
    validate_result(
        case,
        before,
        persistent_cache,
        persistent_outputs,
        label=f"{case.name}/persistent",
    )
    for returned, supplied in zip(persistent_outputs, persistent_buffers):
        assert returned.data_ptr() == supplied.data_ptr(), (
            f"{case.name}: caller-owned output buffer was replaced"
        )
    _compare_valid_outputs(
        case, fresh_outputs, persistent_outputs, label=case.name
    )
    assert torch.equal(fresh_cache.cpu(), persistent_cache.cpu()), (
        f"{case.name}: cache states differ"
    )

    repeat_before = fresh_cache.cpu()
    repeat_outputs = call_mtp(case, fresh_cache)
    torch.npu.synchronize()
    repeat_counts = validate_result(
        case,
        repeat_before,
        fresh_cache,
        repeat_outputs,
        label=f"{case.name}/repeat",
    )
    assert not any(repeat_counts), f"{case.name}: identical repeat must be zero miss"


# ---------------------------------------------------------------------------
# Graph capture + replay with dynamic inputs and evolving state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("replays", [3])
def test_graph_capture_replay(device, replays):
    batch_size = 6
    case = make_case(
        name="graph",
        device=device,
        dtype=torch.bfloat16,
        candidate_lens=(20992,) * batch_size,
        cache_tokens=(8192,) * batch_size,
        miss_fractions=(0.75,) * batch_size,
        source_capacity=32768,
        seed=3007,
    )
    graph_cache = case.initial_cache_cpu.to(device)
    eager_cache = case.initial_cache_cpu.to(device)
    graph_buffers = make_outputs(case)

    warm_cache = case.initial_cache_cpu.to(device)
    call_mtp_with_buffers(case, warm_cache, *make_outputs(case))
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        graph_outputs = call_mtp_with_buffers(
            case, graph_cache, *graph_buffers
        )
    torch.npu.synchronize()
    for returned, supplied in zip(graph_outputs, graph_buffers):
        assert returned.data_ptr() == supplied.data_ptr(), (
            "graph capture lost caller-owned LIM buffers"
        )

    graph_cache.copy_(case.initial_cache_cpu.to(device))
    eager_cache.copy_(case.initial_cache_cpu.to(device))
    torch.npu.synchronize()
    generator = torch.Generator().manual_seed(4017)
    max_blocks = case.source_capacity // BLOCK_SIZE
    base_table = case.block_table_cpu.clone()
    for replay in range(replays):
        query_cpu = torch.randn(
            case.query.shape, generator=generator, dtype=torch.float32
        ).to(case.dtype)
        weights_cpu = torch.rand(
            case.weights.shape, generator=generator, dtype=torch.float32
        ).to(case.dtype)
        request_order = torch.roll(
            torch.arange(batch_size, dtype=torch.int64), shifts=replay + 1
        )
        pool_entries_cpu = torch.roll(
            case.req_pool_entries_cpu, shifts=replay + 1
        )
        candidate_len = min(20992 + replay * 1024, case.source_capacity)
        candidate_len -= candidate_len % BLOCK_SIZE
        candidate_cpu = torch.full(
            (batch_size,), candidate_len, dtype=torch.int32
        )
        table_cpu = base_table.index_select(0, request_order).contiguous()
        assert table_cpu.shape == (batch_size, max_blocks), (
            "graph block-table refresh changed shape"
        )

        case.query.copy_(query_cpu.to(device))
        case.weights.copy_(weights_cpu.to(device))
        case.req_pool_entries.copy_(pool_entries_cpu.to(device))
        case.candidate_lens.copy_(candidate_cpu.to(device))
        case.block_table.copy_(table_cpu.to(device))
        torch.npu.current_stream().synchronize()
        graph.replay()
        torch.npu.synchronize()

        eager_outputs = call_mtp(case, eager_cache)
        torch.npu.synchronize()
        _compare_valid_outputs(
            case, graph_outputs, eager_outputs, label=f"graph/replay={replay}"
        )
        assert torch.equal(graph_cache.cpu(), eager_cache.cpu()), (
            f"graph replay={replay} cache state differs"
        )

    del case, graph_cache, eager_cache, warm_cache
    gc.collect()
    torch.npu.empty_cache()


# ---------------------------------------------------------------------------
# Performance: prints latency only, no threshold assertion (per BEHAVIOR.md)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,source_len,cache_tokens", [(24, 20992, 8192)])
@pytest.mark.skipif(
    not torch_npu.npu.is_available(),
    reason="performance case needs a real NPU",
)
def test_performance(device, batch_size, source_len, cache_tokens):
    perf_query_miss_count = 200
    perf_query_noise = 0.25
    warmup = 10
    iters = 100
    case = make_case(
        name="performance",
        device=device,
        dtype=torch.bfloat16,
        candidate_lens=(source_len,) * batch_size,
        cache_tokens=(cache_tokens,) * batch_size,
        miss_fractions=(0.0,) * batch_size,
        seed=6007,
        correlated_query_noise=perf_query_noise,
        balanced_query_miss_count=perf_query_miss_count,
    )

    union_sizes = [int(row.numel()) for row in case.union_cpu]
    union_mean = statistics.mean(union_sizes)
    assert 3000.0 <= union_mean <= 4000.0, (
        f"MTP3 TopK union mean={union_mean:.2f} outside [3000,4000]; "
        "adjust perf_query_noise and rerun."
    )

    per_query_totals, expected_union_counts = _query_miss_counts(case)
    expected_query_total = batch_size * perf_query_miss_count
    assert per_query_totals == [expected_query_total] * QUERY_COUNT, (
        f"performance workload per-query misses={per_query_totals}, "
        f"expected={expected_query_total} for every query"
    )
    minimum_unique_per_request = (
        2 * perf_query_miss_count - perf_query_miss_count // 2
    )
    maximum_unique_per_request = 2 * perf_query_miss_count
    assert all(
        minimum_unique_per_request <= count <= maximum_unique_per_request
        for count in expected_union_counts
    ), (
        f"performance workload union misses={expected_union_counts}, "
        f"expected range=[{minimum_unique_per_request},"
        f"{maximum_unique_per_request}] per request"
    )

    correctness_cache = case.initial_cache_cpu.to(device)
    correctness_outputs = call_mtp(case, correctness_cache)
    torch.npu.synchronize()
    correctness_counts = validate_result(
        case,
        case.initial_cache_cpu,
        correctness_cache,
        correctness_outputs,
        label="performance/correctness",
    )
    assert correctness_counts == expected_union_counts, (
        f"MTP LIM miss_counts={correctness_counts}, "
        f"expected={expected_union_counts}"
    )

    query_view = case.query.view(batch_size, QUERY_COUNT, HEADS, HEAD_DIM)
    weights_view = case.weights.view(batch_size, QUERY_COUNT, HEADS)
    single_query = query_view[:, 0].contiguous()
    single_weights = weights_view[:, 0].contiguous()
    single_query_ends = torch.arange(
        1, batch_size + 1, dtype=torch.int32, device=device
    )
    mtp_query_ends = torch.arange(
        QUERY_COUNT,
        batch_size * QUERY_COUNT + 1,
        QUERY_COUNT,
        dtype=torch.int32,
        device=device,
    )

    native_single = _call_native_li(
        single_query,
        case.key,
        single_weights,
        case.block_table,
        case.candidate_lens,
        single_query_ends,
    )
    native_mtp = _call_native_li(
        case.query,
        case.key,
        case.weights,
        case.block_table,
        case.candidate_lens,
        mtp_query_ends,
    )
    torch.npu.synchronize()
    _assert_topk_sets(
        native_single,
        [case.topk_cpu[request * QUERY_COUNT] for request in range(batch_size)],
        label="official_li_single",
    )
    expected_native_mtp_shape = (batch_size * QUERY_COUNT, 1, TOPK)
    assert tuple(native_mtp.shape) == expected_native_mtp_shape, (
        f"official LI MTP3 shape={tuple(native_mtp.shape)}, "
        f"expected={expected_native_mtp_shape}"
    )

    single_initial = case.initial_cache_cpu.to(device)
    single_cache = torch.empty_like(single_initial)
    single_buffers = (
        torch.empty((batch_size, 1, TOPK), dtype=torch.int32, device=device),
        torch.empty((batch_size, 1, TOPK), dtype=torch.int32, device=device),
        torch.empty((batch_size,), dtype=torch.int32, device=device),
    )
    mtp_initial = case.initial_cache_cpu.to(device)
    mtp_cache = torch.empty_like(mtp_initial)
    mtp_buffers = make_outputs(case)

    def native_single_step() -> object:
        return _call_native_li(
            single_query,
            case.key,
            single_weights,
            case.block_table,
            case.candidate_lens,
            single_query_ends,
        )

    def fused_mtp_step() -> object:
        return call_mtp_with_buffers(case, mtp_cache, *mtp_buffers)

    official_li_single_us = _event_us(
        native_single_step, warmup=warmup, iters=iters
    )
    official_li_mtp3_us = _event_us(
        native_single_step, warmup=warmup, iters=iters
    )
    fused_lim_mtp3_us = _event_us(
        fused_mtp_step,
        warmup=warmup,
        iters=iters,
        reset=lambda: mtp_cache.copy_(mtp_initial),
    )
    management_mtp3_us = fused_lim_mtp3_us - official_li_mtp3_us
    management_ratio = (
        management_mtp3_us / (fused_lim_mtp3_us - official_li_single_us)
        if (fused_lim_mtp3_us - official_li_single_us) > 0
        else float("nan")
    )
    print(
        "FUSED_LI_MANAGE_MTP_PERF "
        f"batch={batch_size} candidate_len={source_len} "
        f"cache_tokens={cache_tokens} "
        f"per_query_misses={perf_query_miss_count} "
        f"unique_union_misses_mean={statistics.mean(expected_union_counts):.2f} "
        f"topk_union_mean={union_mean:.2f} "
        f"official_li_single_us={official_li_single_us:.3f} "
        f"official_li_mtp3_us={official_li_mtp3_us:.3f} "
        f"fused_lim_mtp3_us={fused_lim_mtp3_us:.3f} "
        f"index_management_mtp3_us={management_mtp3_us:+.3f} "
        f"management_ratio={management_ratio:.4f} target_ratio=4.0000 "
        f"timer=npu_event warmup={warmup} iters={iters}",
        flush=True,
    )
    del case
    gc.collect()
    torch.npu.empty_cache()
