#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Op-level tests for the DISCRETE sparse-index layout of SparseFlashAttention.

COMPACTED layout (legacy) puts the (-1) padding only after the valid token ids,
e.g. ``[3, 2, -1, -1]``; the kernel stops at the first -1. DISCRETE layout
(``sparse_indices_discrete=True``) allows -1 holes anywhere, e.g.
``[-1, 3, -1, 2]``; the kernel must skip interior holes instead of stopping.

To avoid hand-rolling kernel inputs (easy to get subtly wrong and then segfault
on device), this reuses the proven tensor construction from
``test_sfa_v1_precision`` and only swaps the call site to invoke the op directly
with the new flag. Decode shapes only, so every selected token is causally valid
for sparse_mode=3.
"""

import os

# Offline NPU box: use the locally-cached HF config (the DeepSeek-V3.2 model is
# already present for inference) instead of hitting the network, which is
# unreachable and otherwise aborts _get_vllm_config. Must be set before any
# transformers/vllm import below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import math

import pytest
import torch
from vllm.forward_context import set_forward_context

# Importing this module also runs enable_custom_op() and the torch_npu mocks.
from tests.ut.attention.a2 import test_sfa_v1_precision as P

# Only the HF *config* (dims) is needed, no weights. Override with a local model
# dir on an offline box, e.g. SFA_TEST_MODEL=/path/to/DeepSeek-V3.2-Exp.
_MODEL = os.environ.get("SFA_TEST_MODEL", "deepseek-ai/DeepSeek-V3.2-Exp")


def _build_inputs(seq_lens, query_lens, dtype, device, vllm_config):
    """Mirror test_sfa_v1_precision._run_precision_check tensor construction."""
    spec = P.BatchSpec(seq_lens=list(seq_lens), query_lens=list(query_lens), name="discrete")
    cache_config = vllm_config.cache_config
    hf_text = vllm_config.model_config.hf_text_config
    block_size = cache_config.block_size
    qk_rope = hf_text.qk_rope_head_dim
    kv_lora = hf_text.kv_lora_rank
    num_heads = hf_text.num_attention_heads
    scale = 1.0 / math.sqrt(kv_lora + qk_rope)

    meta = P.create_common_attn_metadata(spec, block_size=block_size, device=device)

    k_nope_ctx = [torch.randn(s, kv_lora, dtype=dtype, device=device) for s in seq_lens]
    k_rope_ctx = [torch.randn(s, qk_rope, dtype=dtype, device=device) for s in seq_lens]
    k_nope_cache, k_rope_cache, block_table = P._build_paged_kv_cache_from_metadata(
        common_attn_metadata=meta,
        seq_lens=list(seq_lens),
        k_nope_contexts=k_nope_ctx,
        k_rope_contexts=k_rope_ctx,
        block_size=block_size,
        kv_lora_rank=kv_lora,
        qk_rope_head_dim=qk_rope,
        dtype=dtype,
        device=device,
    )

    num_tokens = sum(query_lens)
    ql_nope = torch.randn(num_tokens, num_heads, kv_lora, dtype=dtype, device=device)
    q_pe = torch.randn(num_tokens, num_heads, qk_rope, dtype=dtype, device=device)
    cum_query_lens = torch.tensor(
        [sum(query_lens[: i + 1]) for i in range(len(query_lens))], dtype=torch.int32, device=device)
    seq_lens_tensor = torch.tensor(list(seq_lens), dtype=torch.int32, device=device)

    return dict(
        ql_nope=ql_nope, q_pe=q_pe, k_nope_cache=k_nope_cache, k_rope_cache=k_rope_cache,
        block_table=block_table, cum_query_lens=cum_query_lens, seq_lens_tensor=seq_lens_tensor,
        scale=scale, num_heads=num_heads, k_nope_ctx=k_nope_ctx, k_rope_ctx=k_rope_ctx,
    )


def _run_op(t, topk_indices, *, discrete):
    """Direct kernel call; only ``sparse_indices_discrete`` differs across runs."""
    out, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=t["ql_nope"],
        key=t["k_nope_cache"],
        value=t["k_nope_cache"],
        sparse_indices=topk_indices,
        scale_value=t["scale"],
        sparse_block_size=1,
        block_table=t["block_table"],
        actual_seq_lengths_query=t["cum_query_lens"],
        actual_seq_lengths_kv=t["seq_lens_tensor"],
        query_rope=t["q_pe"],
        key_rope=t["k_rope_cache"],
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        sparse_indices_discrete=discrete,
    )
    return out


def _compacted_topk(selected, device):
    """Front-packed ids + trailing -1: ``[t0, t1, ..., -1, -1]``."""
    topk = torch.full((len(selected), 1, P.SPARSE_COUNT), -1, dtype=torch.int32, device=device)
    for t, sel in enumerate(selected):
        if sel:
            topk[t, 0, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=device)
    return topk


def _scatter_topk(selected, device):
    """Same ids spread across even slots (a -1 hole before/between each id).

    id ``sel[k]`` goes to slot ``2*k``; needs ``2*len(sel)-1 < seq_len`` so every
    valid id stays inside the kernel scan span ``min(seq_len, sparse_count)``.
    """
    topk = torch.full((len(selected), 1, P.SPARSE_COUNT), -1, dtype=torch.int32, device=device)
    for t, sel in enumerate(selected):
        for k, tok in enumerate(sel):
            topk[t, 0, 2 * k] = tok
    return topk


def _cpu_golden(t, selected, dtype):
    """fp32 gather-softmax over exactly the selected ids (per decode token)."""
    outputs = []
    for b, sel in enumerate(selected):
        k_nope = t["k_nope_ctx"][b][sel].float()             # (M, kv_lora)
        k_rope = t["k_rope_ctx"][b][sel].float()             # (M, qk_rope)
        K = torch.cat([k_nope, k_rope], dim=-1)              # (M, 576)
        V = k_nope                                           # (M, kv_lora)
        head_out = []
        for h in range(t["num_heads"]):
            Q = torch.cat([t["ql_nope"][b, h].float(), t["q_pe"][b, h].float()], dim=-1)
            scores = (Q @ K.transpose(0, 1)) * t["scale"]
            attn = torch.softmax(scores, dim=-1)
            head_out.append(attn @ V)
        outputs.append(torch.stack(head_out, dim=0))
    return torch.stack(outputs, dim=0).to(dtype)


def _assert_close(actual, expected, dtype, tag):
    atol = 5e-3 if dtype == torch.float16 else 1e-2
    diff = (actual.float() - expected.float()).abs()
    peak = expected.float().abs().amax().clamp_min(1e-6)
    assert actual.shape == expected.shape, f"[{tag}] shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=atol), (
        f"[{tag}] max|err|={diff.amax().item():.3e} peak|ref|={peak.item():.3e}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_lens", [[64], [64, 96]])
def test_discrete_matches_compacted(dtype, seq_lens):
    """DISCRETE scattered == COMPACTED packed for the same selected subset."""
    torch.manual_seed(2026)
    device = torch.device("npu")
    vllm_config = P._get_vllm_config(_MODEL, dtype, tensor_parallel_size=1)
    query_lens = [1] * len(seq_lens)
    t = _build_inputs(seq_lens, query_lens, dtype, device, vllm_config)

    # subset of <= seq_len/2 tokens so the scattered (2*k) slots stay in span
    selected = [sorted(range(0, s, 3)) for s in seq_lens]

    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        out_compacted = _run_op(t, _compacted_topk(selected, device), discrete=False)
        out_discrete = _run_op(t, _scatter_topk(selected, device), discrete=True)

    _assert_close(out_discrete, out_compacted, dtype, f"discrete-vs-compacted seq={seq_lens}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_lens", [[64], [64, 96]])
def test_discrete_vs_cpu_golden(dtype, seq_lens):
    """DISCRETE selection vs an independent fp32 gather-softmax golden."""
    torch.manual_seed(7)
    device = torch.device("npu")
    vllm_config = P._get_vllm_config(_MODEL, dtype, tensor_parallel_size=1)
    query_lens = [1] * len(seq_lens)
    t = _build_inputs(seq_lens, query_lens, dtype, device, vllm_config)

    selected = [sorted(range(1, s, 4)) for s in seq_lens]  # leading -1 hole + interleaved holes

    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        out = _run_op(t, _scatter_topk(selected, device), discrete=True)

    _assert_close(out, _cpu_golden(t, selected, dtype), dtype, f"discrete-vs-golden seq={seq_lens}")
