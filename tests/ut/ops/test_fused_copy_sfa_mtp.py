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
"""Unit tests for the ``npu_fused_copy_sfa_mtp`` operator meta contract.

These tests run on CPU (no NPU required) by registering a local
``torch.library`` schema + meta implementation that mirrors the C++ schema in
``csrc/torch_binding.cpp`` and the void meta in ``csrc/torch_binding_meta.cpp``.

The in-place operator returns ``None`` and writes three caller-supplied output
buffers (hbm_k_rope, hbm_kv_cache, attention_out). The meta implementation is
a no-op (void return) because in-place outputs already carry the caller's
shapes/dtypes. These tests verify that:
  1. The schema is invokable on ``meta`` device tensors without error.
  2. The return value is ``None`` (no allocating/alias output).
  3. The caller-owned output buffers are not reallocated (shapes/dtypes intact).

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
        _has_cpp_ext = bool(torch.ops._C_ascend.npu_fused_copy_sfa_mtp)
    except (AttributeError, RuntimeError):
        _has_cpp_ext = False
except Exception:
    _has_cpp_ext = False

BATCH = 4
QUERY_COUNT = 4
HEADS = 2
CKV_DIM = 512
KPE_DIM = 64
BLOCK_SIZE = 128
TOPK = 2048
UNION_CAPACITY = QUERY_COUNT * TOPK
CACHE_TOKENS = 8192
HBM_BLOCKS = 64
DRAM_BLOCKS = 164

_LIB = Library("_ascend_fcsfa_mtp_ut", "DEF")


def _register_local_schema() -> None:
    _LIB.define(
        "npu_fused_copy_sfa_mtp("
        "Tensor query_rope, Tensor query, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
        "Tensor num_cache_tokens, Tensor topk_dst_slots, Tensor topk_src_ids, "
        "Tensor miss_src_ids, Tensor miss_dst_slots, Tensor miss_counts, "
        "Tensor hbm_block_table, Tensor dram_block_table, "
        "Tensor(a!) hbm_k_rope, Tensor(b!) hbm_kv_cache, "
        "Tensor dram_k_rope, Tensor dram_kv_cache, float scale_value, "
        "Tensor(c!) attention_out) -> ()"
    )

    def _meta(
        query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
        num_cache_tokens, topk_dst_slots, topk_src_ids,
        miss_src_ids, miss_dst_slots, miss_counts,
        hbm_block_table, dram_block_table,
        hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache,
        scale_value, attention_out,
    ):
        return None

    _LIB.impl("npu_fused_copy_sfa_mtp", _meta, "Meta")


if not _has_cpp_ext:
    _register_local_schema()


def _op():
    if _has_cpp_ext:
        return torch.ops._C_ascend.npu_fused_copy_sfa_mtp
    return getattr(torch.ops, "_ascend_fcsfa_mtp_ut").npu_fused_copy_sfa_mtp


@pytest.fixture
def meta_case():
    meta = torch.device("meta")
    bq = BATCH * QUERY_COUNT
    return {
        "query_rope": torch.empty(bq, HEADS, KPE_DIM, device=meta, dtype=torch.bfloat16),
        "query": torch.empty(bq, HEADS, CKV_DIM, device=meta, dtype=torch.bfloat16),
        "actual_seq_lengths_query": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "actual_seq_lengths_kv": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "num_cache_tokens": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "topk_dst_slots": torch.empty(bq, 1, TOPK, device=meta, dtype=torch.int32),
        "topk_src_ids": torch.empty(bq, 1, TOPK, device=meta, dtype=torch.int32),
        "miss_src_ids": torch.empty(BATCH, UNION_CAPACITY, device=meta, dtype=torch.int32),
        "miss_dst_slots": torch.empty(BATCH, UNION_CAPACITY, device=meta, dtype=torch.int32),
        "miss_counts": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "hbm_block_table": torch.empty(BATCH, HBM_BLOCKS, device=meta, dtype=torch.int32),
        "dram_block_table": torch.empty(BATCH, DRAM_BLOCKS, device=meta, dtype=torch.int32),
        "hbm_k_rope": torch.empty(HBM_BLOCKS, BLOCK_SIZE, 1, KPE_DIM, device=meta, dtype=torch.bfloat16),
        "hbm_kv_cache": torch.empty(HBM_BLOCKS, BLOCK_SIZE, 1, CKV_DIM, device=meta, dtype=torch.bfloat16),
        "dram_k_rope": torch.empty(DRAM_BLOCKS, BLOCK_SIZE, KPE_DIM, device=meta, dtype=torch.bfloat16),
        "dram_kv_cache": torch.empty(DRAM_BLOCKS, BLOCK_SIZE, CKV_DIM, device=meta, dtype=torch.bfloat16),
        "scale_value": 0.0442,
        "attention_out": torch.empty(bq, HEADS, CKV_DIM, device=meta, dtype=torch.bfloat16),
    }


def _call_op(mc):
    return _op().default(
        mc["query_rope"], mc["query"],
        mc["actual_seq_lengths_query"], mc["actual_seq_lengths_kv"],
        mc["num_cache_tokens"], mc["topk_dst_slots"], mc["topk_src_ids"],
        mc["miss_src_ids"], mc["miss_dst_slots"], mc["miss_counts"],
        mc["hbm_block_table"], mc["dram_block_table"],
        mc["hbm_k_rope"], mc["hbm_kv_cache"],
        mc["dram_k_rope"], mc["dram_kv_cache"],
        mc["scale_value"], mc["attention_out"],
    )


def test_meta_returns_none(meta_case):
    result = _call_op(meta_case)
    assert result is None, "FusedCopySfaMtp must return None"


def test_meta_preserves_buffer_shapes(meta_case):
    _call_op(meta_case)
    expected = {
        "hbm_k_rope": (HBM_BLOCKS, BLOCK_SIZE, 1, KPE_DIM),
        "hbm_kv_cache": (HBM_BLOCKS, BLOCK_SIZE, 1, CKV_DIM),
        "attention_out": (BATCH * QUERY_COUNT, HEADS, CKV_DIM),
    }
    for name, shape in expected.items():
        assert tuple(meta_case[name].shape) == shape, (
            f"{name} shape changed: {tuple(meta_case[name].shape)} != {shape}"
        )


def test_meta_preserves_buffer_dtypes(meta_case):
    _call_op(meta_case)
    for name in ("hbm_k_rope", "hbm_kv_cache", "attention_out"):
        assert meta_case[name].dtype == torch.bfloat16, (
            f"{name} dtype changed: {meta_case[name].dtype} != bfloat16"
        )
