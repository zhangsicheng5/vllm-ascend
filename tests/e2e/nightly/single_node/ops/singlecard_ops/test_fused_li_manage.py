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
"""E2E tests for the ``npu_fused_li_manage`` custom Ascend operator.

Migrated from ``nanovllm-DSA-offload/ut_ops/test_fused_li_manage.py``.

Validates query_len=1 LightningIndexer + hit/evict + request-pool update.
The reference is ``torch.ops._C_ascend.npu_lightning_indexer`` (same golden
used by the source repo). Performance data is printed only; no threshold
assertion (per source repo convention).
"""

from __future__ import annotations

import gc
import statistics
from collections.abc import Callable

import pytest
import torch
import torch_npu  # type: ignore

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = False
enable_custom_op()

BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 2048
EXACT_PAYLOAD_MAX_SOURCE_TOKENS = 1 << 18
MAX_ACTUAL_SEQ_LEN = (1 << 21) - 1
MAX_CAPACITY = 1 << 21
MAX_CACHE_TOKENS = (1 << 14) - 1


def build_case(*, device, heads, batch_size, seq_len, cache_tokens_value,
               miss_count, seed):
    blocks_per_request = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    capacity = blocks_per_request * BLOCK_SIZE
    total_blocks = batch_size * blocks_per_request
    query = torch.zeros((batch_size, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    query[:, 0, 0] = 1; query[:, 0, 1] = 64; query[:, 0, 2] = 4096; query[:, 0, 3] = 262144
    weights = torch.zeros((batch_size, heads), dtype=torch.bfloat16, device=device)
    weights[:, 0] = 1
    key = torch.zeros((total_blocks, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16, device=device)
    logical_ids = torch.arange(capacity, dtype=torch.int32, device=device).view(1, blocks_per_request, BLOCK_SIZE)
    key_rows = key.view(batch_size, blocks_per_request, BLOCK_SIZE, 1, HEAD_DIM)
    key_rows[:, :, :, 0, 0] = (logical_ids % 64).to(torch.bfloat16)
    key_rows[:, :, :, 0, 1] = ((logical_ids // 64) % 64).to(torch.bfloat16)
    key_rows[:, :, :, 0, 2] = ((logical_ids // 4096) % 64).to(torch.bfloat16)
    key_rows[:, :, :, 0, 3] = (logical_ids // 262144).to(torch.bfloat16)
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(batch_size, blocks_per_request)
    candidate_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    query_lens = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    cache_tokens = torch.full((batch_size,), cache_tokens_value, dtype=torch.int32, device=device)
    pool_size = batch_size + 1
    req_entries_cpu = torch.arange(batch_size, 0, -1, dtype=torch.int32)
    req_entries = req_entries_cpu.to(device)
    initial_cache_cpu = torch.full((pool_size, capacity), -1, dtype=torch.int32)
    topk = torch.arange(seq_len - TOPK, seq_len, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    hit_tokens = topk[miss_count:]
    other_count = cache_tokens_value - hit_tokens.numel()
    lower_count = seq_len - TOPK
    other_tokens = (torch.div(torch.arange(other_count, dtype=torch.int64) * lower_count, other_count, rounding_mode="floor") if other_count else torch.empty(0, dtype=torch.int64))
    cached_tokens = torch.cat((hit_tokens, other_tokens))
    for row in range(batch_size):
        slots = torch.randperm(cache_tokens_value, generator=generator, dtype=torch.int32)
        initial_cache_cpu[int(req_entries_cpu[row]), cached_tokens] = slots
    initial_cache = initial_cache_cpu.to(device)
    cache_slots = initial_cache.clone()
    source_ids = torch.full((batch_size, 1, TOPK), -1, dtype=torch.int32, device=device)
    destination_slots = torch.full_like(source_ids, -1)
    miss_counts = torch.full((batch_size,), -1, dtype=torch.int32, device=device)
    return {
        "capacity": capacity, "query": query, "key": key, "weights": weights,
        "req_entries": req_entries, "req_entries_cpu": req_entries_cpu,
        "initial_cache": initial_cache, "initial_cache_cpu": initial_cache_cpu,
        "cache_slots": cache_slots, "cache_tokens": cache_tokens,
        "query_lens": query_lens, "candidate_lens": candidate_lens,
        "block_table": block_table, "source_ids": source_ids,
        "destination_slots": destination_slots, "miss_counts": miss_counts,
    }


def call_fused_li_manage(case):
    torch.ops._C_ascend.npu_fused_li_manage(
        case["query"], case["weights"], case["key"], case["block_table"],
        case["candidate_lens"], case["cache_tokens"], case["req_entries"],
        case["cache_slots"], case["source_ids"], case["destination_slots"],
        case["miss_counts"])


def call_lightning_indexer(case):
    output = torch.ops._C_ascend.npu_lightning_indexer(
        query=case["query"], key=case["key"], weights=case["weights"],
        actual_seq_lengths_query=case["query_lens"],
        actual_seq_lengths_key=case["candidate_lens"],
        block_table=case["block_table"], layout_query="TND",
        layout_key="PA_BSND", sparse_count=TOPK, sparse_mode=3,
        return_value=False)
    return output[0] if isinstance(output, (tuple, list)) else output


def validate_outputs(case, *, seq_len, cache_tokens_value, expected_miss_count,
                     reference_topk, old_cache_pool):
    batch_size = int(case["candidate_lens"].numel())
    counts = case["miss_counts"].cpu()
    expected_counts = torch.full((batch_size,), expected_miss_count, dtype=torch.int32)
    assert torch.equal(counts, expected_counts), f"miss_counts={counts.tolist()}, expected={expected_counts.tolist()}"
    sources = case["source_ids"].view(batch_size, TOPK).cpu().to(torch.int64)
    slots = case["destination_slots"].view(batch_size, TOPK).cpu().to(torch.int64)
    state_pool = case["cache_slots"].cpu()
    old_pool = old_cache_pool.cpu()
    req_entries_cpu = case["req_entries_cpu"].to(torch.int64)
    reference = reference_topk.view(batch_size, TOPK).cpu().to(torch.int64)
    mapped_rows = set(int(v) for v in req_entries_cpu.tolist())
    for pool_row in range(state_pool.shape[0]):
        if pool_row not in mapped_rows:
            assert torch.equal(state_pool[pool_row], old_pool[pool_row]), f"Unmapped pool row {pool_row} modified"
    for row in range(batch_size):
        source_row = sources[row]; slot_row = slots[row]
        assert not bool((source_row < 0).any()) and not bool((source_row >= seq_len).any()), f"row={row} out-of-range token"
        assert torch.unique(source_row).numel() == TOPK, f"row={row} duplicates"
        assert torch.equal(torch.sort(source_row).values, torch.sort(reference[row]).values), f"row={row} topk differs from reference"
        assert not bool((slot_row < 0).any()) and not bool((slot_row >= cache_tokens_value).any()), f"row={row} invalid slot"
        assert torch.unique(slot_row).numel() == TOPK, f"row={row} slot duplicates"
        pool_row = int(req_entries_cpu[row])
        old_state = old_pool[pool_row]; new_state = state_pool[pool_row]
        old_topk_slots = old_state[source_row].to(torch.int64)
        actual_miss_mask = old_topk_slots == -1
        actual_miss_count = int(actual_miss_mask.sum())
        assert actual_miss_count == expected_miss_count, f"row={row} miss={actual_miss_count}, expected={expected_miss_count}"
        if expected_miss_count:
            assert bool(actual_miss_mask[:expected_miss_count].all()), f"row={row} miss prefix has hit"
        assert not bool(actual_miss_mask[expected_miss_count:].any()), f"row={row} hit suffix has miss"
        assert torch.equal(slot_row[expected_miss_count:], old_topk_slots[expected_miss_count:]), f"row={row} hit slot not preserved"
        assert torch.equal(new_state[source_row].to(torch.int64), slot_row), f"row={row} new_cache[topk]!=slots"
        valid_slots = new_state[new_state >= 0].to(torch.int64)
        assert valid_slots.numel() == cache_tokens_value and torch.unique(valid_slots).numel() == cache_tokens_value, f"row={row} cache permutation invalid"
    return state_pool


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


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("seq_len", [262272])
def test_fused_li_manage_semantic(device, heads, seq_len):
    cache_tokens = 6144
    miss_count = 300
    case = build_case(device=device, heads=heads, batch_size=1, seq_len=seq_len,
                      cache_tokens_value=cache_tokens, miss_count=miss_count, seed=7)
    lightning_output = call_lightning_indexer(case)
    torch.npu.synchronize()
    reference_topk = lightning_output.view(1, TOPK).cpu()
    case["cache_slots"].copy_(case["initial_cache"])
    call_fused_li_manage(case)
    torch.npu.synchronize()
    validate_outputs(case, seq_len=seq_len, cache_tokens_value=cache_tokens,
                     expected_miss_count=miss_count, reference_topk=reference_topk,
                     old_cache_pool=case["initial_cache_cpu"])
    call_fused_li_manage(case)
    torch.npu.synchronize()
    validate_outputs(case, seq_len=seq_len, cache_tokens_value=cache_tokens,
                     expected_miss_count=0, reference_topk=reference_topk,
                     old_cache_pool=case["cache_slots"].cpu().clone())
    gc.collect(); torch.npu.empty_cache()
