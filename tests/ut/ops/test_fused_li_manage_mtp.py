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
"""Unit tests for the ``npu_fused_li_manage_mtp`` operator meta contract.

These tests run on CPU (no NPU required) by registering a local
``torch.library`` schema + meta implementation that mirrors the C++ schema in
``csrc/torch_binding.cpp`` and the void meta in ``csrc/torch_binding_meta.cpp``.

The in-place operator returns ``None`` and writes six caller-supplied output
buffers. The meta implementation is a no-op (void return) because in-place
outputs already carry the caller's shapes/dtypes. These tests verify that:
  1. The schema is invokable on ``meta`` device tensors without error.
  2. The return value is ``None`` (no allocating/alias output).
  3. The caller-owned output buffers are not reallocated (shapes/dtypes intact).
  4. The ``_ordered_union`` reference helper produces stable ordered unions.

When the C++ extension (``vllm_ascend_C``) is available, its registered schema
takes precedence and the local registration here is skipped.
"""

from __future__ import annotations

import pytest
import torch
from torch.library import Library

try:
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    try:
        _has_cpp_ext = bool(torch.ops._C_ascend.npu_fused_li_manage_mtp)
    except (AttributeError, RuntimeError):
        _has_cpp_ext = False
except Exception:
    _has_cpp_ext = False

QUERY_COUNT = 4
HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 2048
UNION_CAPACITY = QUERY_COUNT * TOPK

# If the C++ extension registered the op, use it; otherwise register a local
# mirror schema+meta so the shape/dtype contract can still be validated on CPU.
_LIB = Library("_ascend_lim_mtp_ut", "DEF")


def _register_local_schema() -> None:
    _LIB.define(
        "npu_fused_li_manage_mtp("
        "Tensor query, Tensor index_weights, Tensor index_key_cache, "
        "Tensor index_block_table, Tensor num_candidate_tokens, "
        "Tensor num_cache_tokens, Tensor req_pool_entries, "
        "Tensor(a!) cache_slots_pool, Tensor(b!) topk_src_ids, "
        "Tensor(c!) topk_dst_slots, Tensor(d!) miss_src_ids, "
        "Tensor(e!) miss_dst_slots, Tensor(f!) miss_counts) -> ()"
    )

    def _meta(
        query,
        index_weights,
        index_key_cache,
        index_block_table,
        num_candidate_tokens,
        num_cache_tokens,
        req_pool_entries,
        cache_slots_pool,
        topk_src_ids,
        topk_dst_slots,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
    ):
        return None

    _LIB.impl("npu_fused_li_manage_mtp", _meta, "Meta")


if not _has_cpp_ext:
    _register_local_schema()


def _op():
    if _has_cpp_ext:
        return torch.ops._C_ascend.npu_fused_li_manage_mtp
    return getattr(torch.ops, "_ascend_lim_mtp_ut").npu_fused_li_manage_mtp


@pytest.fixture
def meta_case():
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
    topk_slots = torch.empty(
        batch_size * QUERY_COUNT, 1, TOPK, device=meta, dtype=torch.int32
    )
    topk_source_ids = torch.empty_like(topk_slots)
    miss_sources = torch.empty(
        batch_size, UNION_CAPACITY, device=meta, dtype=torch.int32
    )
    miss_destinations = torch.empty_like(miss_sources)
    miss_counts = torch.empty(batch_size, device=meta, dtype=torch.int32)
    return {
        "query": query,
        "weights": weights,
        "key": key,
        "block_table": block_table,
        "candidate_lens": candidate_lens,
        "cache_tokens": cache_tokens,
        "req_entries": req_entries,
        "cache_slots": cache_slots,
        "topk_slots": topk_slots,
        "topk_source_ids": topk_source_ids,
        "miss_sources": miss_sources,
        "miss_destinations": miss_destinations,
        "miss_counts": miss_counts,
        "batch_size": batch_size,
    }


def test_meta_returns_none(meta_case):
    result = _op().default(
        meta_case["query"],
        meta_case["weights"],
        meta_case["key"],
        meta_case["block_table"],
        meta_case["candidate_lens"],
        meta_case["cache_tokens"],
        meta_case["req_entries"],
        meta_case["cache_slots"],
        meta_case["topk_source_ids"],
        meta_case["topk_slots"],
        meta_case["miss_sources"],
        meta_case["miss_destinations"],
        meta_case["miss_counts"],
    )
    assert result is None, "Fused LIM Manage MTP must return None"


def test_meta_preserves_buffer_shapes(meta_case):
    expected = {
        "topk_slots": (meta_case["batch_size"] * QUERY_COUNT, 1, TOPK),
        "topk_source_ids": (meta_case["batch_size"] * QUERY_COUNT, 1, TOPK),
        "miss_sources": (meta_case["batch_size"], UNION_CAPACITY),
        "miss_destinations": (meta_case["batch_size"], UNION_CAPACITY),
        "miss_counts": (meta_case["batch_size"],),
    }
    _op().default(
        meta_case["query"],
        meta_case["weights"],
        meta_case["key"],
        meta_case["block_table"],
        meta_case["candidate_lens"],
        meta_case["cache_tokens"],
        meta_case["req_entries"],
        meta_case["cache_slots"],
        meta_case["topk_source_ids"],
        meta_case["topk_slots"],
        meta_case["miss_sources"],
        meta_case["miss_destinations"],
        meta_case["miss_counts"],
    )
    for name, shape in expected.items():
        assert tuple(meta_case[name].shape) == shape, (
            f"{name} shape changed: {tuple(meta_case[name].shape)} != {shape}"
        )


def test_meta_preserves_buffer_dtypes(meta_case):
    _op().default(
        meta_case["query"],
        meta_case["weights"],
        meta_case["key"],
        meta_case["block_table"],
        meta_case["candidate_lens"],
        meta_case["cache_tokens"],
        meta_case["req_entries"],
        meta_case["cache_slots"],
        meta_case["topk_source_ids"],
        meta_case["topk_slots"],
        meta_case["miss_sources"],
        meta_case["miss_destinations"],
        meta_case["miss_counts"],
    )
    for name in ("topk_slots", "topk_source_ids", "miss_sources", "miss_destinations", "miss_counts"):
        assert meta_case[name].dtype == torch.int32, (
            f"{name} dtype changed: {meta_case[name].dtype} != int32"
        )


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


def test_ordered_union_dedup_and_order():
    rows = [
        torch.tensor([3, 1, 4, 1], dtype=torch.int64),
        torch.tensor([4, 5, 9, 2], dtype=torch.int64),
        torch.tensor([6, 3, 5, 8], dtype=torch.int64),
    ]
    union = _ordered_union(rows)
    assert union.tolist() == [3, 1, 4, 5, 9, 2, 6, 8]
    assert union.dtype == torch.int64
    assert torch.unique(union).numel() == union.numel()
