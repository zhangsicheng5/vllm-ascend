# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("has_cmp,explicit_cmp_lengths", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("n_heads", [64, 128])
def test_a5_attention_metadata_uses_consumed_lengths(monkeypatch, has_cmp, explicit_cmp_lengths, n_heads):
    from vllm_ascend.models.deepseek_v41 import mixed_quant_attention as packed

    q = torch.zeros((3, n_heads, 512), dtype=torch.bfloat16)
    ori_lengths = torch.tensor([[1], [2], [3]], dtype=torch.int32)
    cmp_lengths = torch.tensor([[0], [1], [2]], dtype=torch.int32) if explicit_cmp_lengths else None
    indices = torch.zeros((3, 1, 128), dtype=torch.int32)
    metadata = SimpleNamespace(
        positions=torch.arange(3),
        swa=SimpleNamespace(
            ori_sparse_indices=indices,
            ori_topk_length=ori_lengths,
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([2, 3], dtype=torch.int32),
            block_table=torch.zeros((2, 1), dtype=torch.int32),
            smla_metadata=torch.full((1024,), 987654321, dtype=torch.int32),
        ),
        attention=SimpleNamespace(
            cache_seq_lens=torch.tensor([1, 2], dtype=torch.int32),
            block_table=torch.zeros((2, 1), dtype=torch.int32),
        ),
    )
    new_metadata = torch.zeros(1024, dtype=torch.int32)
    expected_cmp = (
        cmp_lengths if explicit_cmp_lengths else torch.full_like(ori_lengths, 512 if has_cmp else 0)
    )
    calls = []

    def build(actual_ori, actual_cmp, **kwargs):
        torch.testing.assert_close(actual_ori, ori_lengths)
        torch.testing.assert_close(actual_cmp, expected_cmp)
        assert kwargs["has_cmp_kv"] == has_cmp
        assert kwargs["num_heads_q"] == n_heads
        calls.append("metadata")
        return new_metadata

    def compute(actual_q, **kwargs):
        assert calls == ["metadata"]
        assert kwargs["metadata"] is new_metadata
        assert kwargs["cmp_topk_length"] is (cmp_lengths if has_cmp else None)
        calls.append("compute")
        return actual_q, torch.empty(0)

    monkeypatch.setattr(packed.ops, "mixed_quant_sparse_flash_mla_metadata", build)
    monkeypatch.setattr(packed.ops, "mixed_quant_sparse_flash_mla", compute)
    cache = torch.empty((1, 128, 1, 584), dtype=torch.uint8)
    result = packed.qsmla(
        q,
        cache,
        cache if has_cmp else None,
        metadata,
        torch.zeros((3, 512), dtype=torch.int32),
        window_size=128,
        sinks=torch.zeros(n_heads),
        softmax_scale=0.5,
        compressed_lengths=cmp_lengths,
    )
    assert result is q
    assert calls == ["metadata", "compute"]


@pytest.mark.parametrize("n_heads", [64, 128])
def test_a5_smla_metadata_helper_forwards_query_heads(monkeypatch, n_heads):
    from vllm_ascend.models.deepseek_v41 import mixed_quant_attention as packed

    lengths = torch.ones((2, 1), dtype=torch.int32)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    expected = torch.zeros(1024, dtype=torch.int32)

    def build(ori, cmp, **kwargs):
        assert ori is lengths and cmp is lengths
        assert kwargs["cu_seqlens_q"] is cu_q
        assert kwargs["num_heads_q"] == n_heads
        return expected

    monkeypatch.setattr(packed.ops, "mixed_quant_sparse_flash_mla_metadata", build)
    assert packed.build_smla_metadata(lengths, cu_q, num_heads_q=n_heads) is expected


def test_deepseek_v41_model_call_sites_use_compiled_operator_layouts():
    """Keep framework call sites aligned with the operator PR contracts."""
    source = ast.parse((REPO_ROOT / "vllm_ascend/attention/dsa_v41.py").read_text())
    calls = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("npu_sparse_flash_mla", "npu_sparse_flash_mla_metadata")
    ]
    assert len(calls) == 2
    for call in calls:
        layouts = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg in ("layout_q", "layout_kv")}
        assert layouts == {"layout_q": "TND", "layout_kv": "PA_BBND"}

    source = ast.parse((REPO_ROOT / "vllm_ascend/models/deepseek_v41/indexer.py").read_text())
    common = next(
        node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "common" for target in node.targets)
    )
    assert isinstance(common, ast.Call)
    layouts = {kw.arg: ast.literal_eval(kw.value) for kw in common.keywords if kw.arg in ("layout_q", "layout_k")}
    assert layouts == {"layout_q": "TND", "layout_k": "PA_BBND"}


def test_compressor_call_sites_use_compiled_modes():
    for filename, count in (("models/deepseek_v4/compressor.py", 1),):
        source = ast.parse((REPO_ROOT / "vllm_ascend" / filename).read_text())
        calls = [
            node
            for node in ast.walk(source)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.ops._C_ascend.compressor"
        ]
        assert len(calls) == count
        for call in calls:
            modes = {
                kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg in ("rotary_mode", "cache_mode")
            }
            assert modes == {"rotary_mode": 2, "cache_mode": 1}
