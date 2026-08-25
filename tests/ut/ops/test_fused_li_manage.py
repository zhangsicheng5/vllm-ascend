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
"""Unit tests for the ``npu_fused_li_manage`` operator meta contract.

CPU-runnable meta contract tests. The in-place operator returns ``None``
and writes four caller-supplied output buffers. Mirrors the C++ schema in
``csrc/torch_binding.cpp`` and the void meta in ``csrc/torch_binding_meta.cpp``.
"""

from __future__ import annotations

import pytest
import torch
from torch.library import Library

try:
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    try:
        _has_cpp_ext = bool(torch.ops._C_ascend.npu_fused_li_manage)
    except (AttributeError, RuntimeError):
        _has_cpp_ext = False
except Exception:
    _has_cpp_ext = False

BATCH = 1
HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 2048
CACHE_TOKENS = 6144
SEQ_LEN = 262272
BLOCKS_PER_REQ = SEQ_LEN // BLOCK_SIZE

_LIB = Library("_ascend_flim_ut", "DEF")


def _register_local_schema() -> None:
    _LIB.define(
        "npu_fused_li_manage("
        "Tensor query, Tensor index_weights, Tensor index_key_cache, "
        "Tensor index_block_table, Tensor num_candidate_tokens, "
        "Tensor num_cache_tokens, Tensor req_pool_entries, "
        "Tensor(a!) cache_slots_pool, Tensor(b!) topk_src_ids, "
        "Tensor(c!) topk_dst_slots, Tensor(d!) miss_counts) -> ()"
    )

    def _meta(query, index_weights, index_key_cache, index_block_table,
              num_candidate_tokens, num_cache_tokens, req_pool_entries,
              cache_slots_pool, topk_src_ids, topk_dst_slots, miss_counts):
        return None

    _LIB.impl("npu_fused_li_manage", _meta, "Meta")


if not _has_cpp_ext:
    _register_local_schema()


def _op():
    if _has_cpp_ext:
        return torch.ops._C_ascend.npu_fused_li_manage
    return getattr(torch.ops, "_ascend_flim_ut").npu_fused_li_manage


@pytest.fixture
def meta_case():
    meta = torch.device("meta")
    pool_size = BATCH + 1
    return {
        "query": torch.empty(BATCH, HEADS, HEAD_DIM, device=meta, dtype=torch.bfloat16),
        "index_weights": torch.empty(BATCH, HEADS, device=meta, dtype=torch.bfloat16),
        "index_key_cache": torch.empty(BATCH * BLOCKS_PER_REQ, BLOCK_SIZE, 1, HEAD_DIM, device=meta, dtype=torch.bfloat16),
        "index_block_table": torch.empty(BATCH, BLOCKS_PER_REQ, device=meta, dtype=torch.int32),
        "num_candidate_tokens": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "num_cache_tokens": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "req_pool_entries": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "cache_slots_pool": torch.empty(pool_size, SEQ_LEN, device=meta, dtype=torch.int32),
        "topk_src_ids": torch.empty(BATCH, 1, TOPK, device=meta, dtype=torch.int32),
        "topk_dst_slots": torch.empty(BATCH, 1, TOPK, device=meta, dtype=torch.int32),
        "miss_counts": torch.empty(BATCH, device=meta, dtype=torch.int32),
    }


def _call_op(mc):
    return _op().default(
        mc["query"], mc["index_weights"], mc["index_key_cache"],
        mc["index_block_table"], mc["num_candidate_tokens"],
        mc["num_cache_tokens"], mc["req_pool_entries"],
        mc["cache_slots_pool"], mc["topk_src_ids"],
        mc["topk_dst_slots"], mc["miss_counts"],
    )


def test_meta_returns_none(meta_case):
    result = _call_op(meta_case)
    assert result is None, "FusedLiManage must return None"


def test_meta_preserves_buffer_shapes(meta_case):
    _call_op(meta_case)
    expected = {
        "topk_src_ids": (BATCH, 1, TOPK),
        "topk_dst_slots": (BATCH, 1, TOPK),
        "miss_counts": (BATCH,),
    }
    for name, shape in expected.items():
        assert tuple(meta_case[name].shape) == shape, (
            f"{name} shape changed: {tuple(meta_case[name].shape)} != {shape}"
        )


def test_meta_preserves_buffer_dtypes(meta_case):
    _call_op(meta_case)
    for name in ("topk_src_ids", "topk_dst_slots", "miss_counts"):
        assert meta_case[name].dtype == torch.int32, (
            f"{name} dtype changed: {meta_case[name].dtype} != int32"
        )
