#!/usr/bin/env python3
"""Baseline probe: call SparseFlashAttention WITHOUT the new discrete attr.

Isolation tool. This omits ``sparse_indices_discrete`` entirely, so it runs
against BOTH the unmodified op (e.g. checked out at main) and this branch. If it
segfaults on main too, the crash is a pre-existing op/input issue, NOT the
discrete change.

Same production-style inputs as probe_discrete_sfa.py: TND query + PA_BSND paged
KV, N=8, sparse_block_size=1. Only COMPACTED (front-packed -1) indices.

It depends only on installed packages (torch/torch_npu/vllm_ascend), so you can
copy it out of the repo and run it after `git checkout main`:
    cp tests/op_test/sparse_flash_attention/diag/probe_baseline_sfa.py ~/probe_baseline.py
    git checkout main && <clean rebuild> && source set_env.bash
    python3 ~/probe_baseline.py
"""

import math
import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

NUM_HEADS = 8
D = 512
DROPE = 64
BLOCK_SIZE = 128
SEQ_LEN = 64
K = 8
DTYPE = torch.bfloat16


def log(*a):
    print("[baseline]", *a, flush=True)


def main():
    log("torch", torch.__version__, "| npu:", torch.npu.is_available())
    device = torch.device("npu")
    torch.manual_seed(0)

    k_nope_dense = torch.randn(SEQ_LEN, D, dtype=DTYPE, device=device)
    k_rope_dense = torch.randn(SEQ_LEN, DROPE, dtype=DTYPE, device=device)
    k_nope_cache = torch.zeros(2, BLOCK_SIZE, 1, D, dtype=DTYPE, device=device)
    k_rope_cache = torch.zeros(2, BLOCK_SIZE, 1, DROPE, dtype=DTYPE, device=device)
    k_nope_cache[1, :SEQ_LEN, 0, :] = k_nope_dense
    k_rope_cache[1, :SEQ_LEN, 0, :] = k_rope_dense
    block_table = torch.tensor([[1]], dtype=torch.int32, device=device)

    ql_nope = torch.randn(1, NUM_HEADS, D, dtype=DTYPE, device=device)
    q_pe = torch.randn(1, NUM_HEADS, DROPE, dtype=DTYPE, device=device)
    cum_q = torch.tensor([1], dtype=torch.int32, device=device)
    seq_kv = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(D + DROPE)

    sel = [3, 17, 40, 61]
    topk = torch.full((1, 1, K), -1, dtype=torch.int32, device=device)
    topk[0, 0, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=device)
    log("inputs built. q", tuple(ql_nope.shape), "kv_cache", tuple(k_nope_cache.shape), "sel", sel)

    log(">>> calling op (COMPACTED, no discrete attr) ...")
    out, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql_nope, key=k_nope_cache, value=k_nope_cache,
        sparse_indices=topk, scale_value=scale, sparse_block_size=1,
        block_table=block_table, actual_seq_lengths_query=cum_q,
        actual_seq_lengths_kv=seq_kv, query_rope=q_pe, key_rope=k_rope_cache,
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2,
    )
    torch.npu.synchronize()
    log("op ok. out", tuple(out.shape))

    Kmat = torch.cat([k_nope_dense[sel].float(), k_rope_dense[sel].float()], dim=-1)
    V = k_nope_dense[sel].float()
    outs = []
    for h in range(NUM_HEADS):
        Q = torch.cat([ql_nope[0, h].float(), q_pe[0, h].float()], dim=-1)
        attn = torch.softmax((Q @ Kmat.transpose(0, 1)) * scale, dim=-1)
        outs.append(attn @ V)
    gold = torch.stack(outs, dim=0)
    dg = (out.reshape(NUM_HEADS, D).float() - gold).abs().amax().item()
    log(f"max|op - golden| = {dg:.3e}  (peak|gold|={gold.abs().amax().item():.3e})")
    log("RESULT:", "PASS" if dg < 3e-2 else "FAIL")
    sys.exit(0 if dg < 3e-2 else 1)


if __name__ == "__main__":
    main()
