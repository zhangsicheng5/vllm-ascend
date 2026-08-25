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
"""Unit tests for the ``npu_sparse_tail_attention`` operator meta contract.

CPU-runnable meta contract tests. The in-place operator returns ``None``
and writes one caller-supplied output buffer (attention_out).
"""

from __future__ import annotations

import pytest
import torch
from torch.library import Library

try:
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    try:
        _has_cpp_ext = bool(torch.ops._C_ascend.npu_sparse_tail_attention)
    except (AttributeError, RuntimeError):
        _has_cpp_ext = False
except Exception:
    _has_cpp_ext = False

BATCH = 4
HEADS = 2
CKV_DIM = 512
KPE_DIM = 64
BLOCK_SIZE = 128
TOPK = 2048
CACHE_TOKENS = 8192
HBM_BLOCKS = 64

_LIB = Library("_ascend_sta_ut", "DEF")


def _register_local_schema() -> None:
    _LIB.define(
        "npu_sparse_tail_attention("
        "Tensor query_rope, Tensor query, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
        "Tensor num_cache_tokens, Tensor topk_dst_slots, "
        "Tensor hbm_block_table, Tensor hbm_k_rope, Tensor hbm_kv_cache, "
        "float scale_value, Tensor(a!) attention_out) -> ()"
    )

    def _meta(query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
              num_cache_tokens, topk_dst_slots, hbm_block_table,
              hbm_k_rope, hbm_kv_cache, scale_value, attention_out):
        return None

    _LIB.impl("npu_sparse_tail_attention", _meta, "Meta")


if not _has_cpp_ext:
    _register_local_schema()


def _op():
    if _has_cpp_ext:
        return torch.ops._C_ascend.npu_sparse_tail_attention
    return getattr(torch.ops, "_ascend_sta_ut").npu_sparse_tail_attention


@pytest.fixture
def meta_case():
    meta = torch.device("meta")
    return {
        "query_rope": torch.empty(BATCH, HEADS, KPE_DIM, device=meta, dtype=torch.bfloat16),
        "query": torch.empty(BATCH, HEADS, CKV_DIM, device=meta, dtype=torch.bfloat16),
        "actual_seq_lengths_query": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "actual_seq_lengths_kv": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "num_cache_tokens": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "topk_dst_slots": torch.empty(BATCH, 1, TOPK, device=meta, dtype=torch.int32),
        "hbm_block_table": torch.empty(BATCH, HBM_BLOCKS, device=meta, dtype=torch.int32),
        "hbm_k_rope": torch.empty(HBM_BLOCKS, BLOCK_SIZE, 1, KPE_DIM, device=meta, dtype=torch.bfloat16),
        "hbm_kv_cache": torch.empty(HBM_BLOCKS, BLOCK_SIZE, 1, CKV_DIM, device=meta, dtype=torch.bfloat16),
        "scale_value": 0.0442,
        "attention_out": torch.empty(BATCH, HEADS, CKV_DIM, device=meta, dtype=torch.bfloat16),
    }


def _call_op(mc):
    return _op().default(
        mc["query_rope"], mc["query"],
        mc["actual_seq_lengths_query"], mc["actual_seq_lengths_kv"],
        mc["num_cache_tokens"], mc["topk_dst_slots"],
        mc["hbm_block_table"], mc["hbm_k_rope"], mc["hbm_kv_cache"],
        mc["scale_value"], mc["attention_out"],
    )


def test_meta_returns_none(meta_case):
    result = _call_op(meta_case)
    assert result is None, "SparseTailAttention must return None"


def test_meta_preserves_buffer_shapes(meta_case):
    _call_op(meta_case)
    assert tuple(meta_case["attention_out"].shape) == (BATCH, HEADS, CKV_DIM), \
        "attention_out shape changed"


def test_meta_preserves_buffer_dtypes(meta_case):
    _call_op(meta_case)
    assert meta_case["attention_out"].dtype == torch.bfloat16, \
        "attention_out dtype changed"
