#!/usr/bin/env python3
"""Minimal standalone probe for SparseFlashAttention discrete sparse_indices.

No vllm / no HuggingFace / no pytest / no forward_context. Mirrors the PRODUCTION
op call (vllm_ascend/device/device_op.py): TND query + PA_BSND paged KV cache,
N>=8 heads, sparse_block_size=1 -- the only config actually exercised on 910b.

    query / query_rope : (T, N, D) / (T, N, Drope)         D=512, Drope=64
    key / key_rope     : (num_blocks, block_size, 1, D/Drope)   (paged, PA_BSND)
    block_table        : (batch, max_blocks) int32
    actual_seq_lengths_query : cumulative (TND), _kv : raw per-batch
    sparse_indices     : (T, 1, K) int32, token ids into [0, seq_len)

Run:  python3 probe_discrete_sfa.py
COMPACTED first (known-good); if that crashes the inputs/env are wrong, not the
discrete change.
"""

import math
import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

NUM_HEADS = 8          # query heads (MQA: 1 kv head); production uses >=8
D = 512                # nope dim
DROPE = 64             # rope dim
BLOCK_SIZE = 128
SEQ_LEN = 64           # single decode request
K = 8                  # topk width (sparse_indices last dim)
DTYPE = torch.bfloat16


def log(*a):
    print("[probe]", *a, flush=True)


def build(device):
    torch.manual_seed(0)
    # dense per-batch K (for golden) + paged cache (block 0 reserved, block 1 holds tokens)
    k_nope_dense = torch.randn(SEQ_LEN, D, dtype=DTYPE, device=device)
    k_rope_dense = torch.randn(SEQ_LEN, DROPE, dtype=DTYPE, device=device)
    total_blocks = 2
    k_nope_cache = torch.zeros(total_blocks, BLOCK_SIZE, 1, D, dtype=DTYPE, device=device)
    k_rope_cache = torch.zeros(total_blocks, BLOCK_SIZE, 1, DROPE, dtype=DTYPE, device=device)
    k_nope_cache[1, :SEQ_LEN, 0, :] = k_nope_dense
    k_rope_cache[1, :SEQ_LEN, 0, :] = k_rope_dense
    block_table = torch.tensor([[1]], dtype=torch.int32, device=device)

    ql_nope = torch.randn(1, NUM_HEADS, D, dtype=DTYPE, device=device)        # TND: (T=1, N, D)
    q_pe = torch.randn(1, NUM_HEADS, DROPE, dtype=DTYPE, device=device)
    cum_q = torch.tensor([1], dtype=torch.int32, device=device)               # TND prefix sum
    seq_kv = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)        # raw per-batch
    scale = 1.0 / math.sqrt(D + DROPE)
    return dict(ql_nope=ql_nope, q_pe=q_pe, k_nope_cache=k_nope_cache, k_rope_cache=k_rope_cache,
                block_table=block_table, cum_q=cum_q, seq_kv=seq_kv, scale=scale,
                k_nope_dense=k_nope_dense, k_rope_dense=k_rope_dense)


def run_op(t, topk, discrete):
    out, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=t["ql_nope"], key=t["k_nope_cache"], value=t["k_nope_cache"],
        sparse_indices=topk, scale_value=t["scale"], sparse_block_size=1,
        block_table=t["block_table"], actual_seq_lengths_query=t["cum_q"],
        actual_seq_lengths_kv=t["seq_kv"], query_rope=t["q_pe"], key_rope=t["k_rope_cache"],
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2,
        sparse_indices_discrete=discrete,
    )
    torch.npu.synchronize()
    return out


def compacted_topk(sel, device):
    topk = torch.full((1, 1, K), -1, dtype=torch.int32, device=device)
    topk[0, 0, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=device)
    return topk


def scatter_topk(sel, device):
    topk = torch.full((1, 1, K), -1, dtype=torch.int32, device=device)
    for k, tok in enumerate(sel):
        topk[0, 0, 2 * k] = tok
    return topk


def cpu_golden(t, sel):
    Kmat = torch.cat([t["k_nope_dense"][sel].float(), t["k_rope_dense"][sel].float()], dim=-1)  # (M,576)
    V = t["k_nope_dense"][sel].float()
    outs = []
    for h in range(NUM_HEADS):
        Q = torch.cat([t["ql_nope"][0, h].float(), t["q_pe"][0, h].float()], dim=-1)
        attn = torch.softmax((Q @ Kmat.transpose(0, 1)) * t["scale"], dim=-1)
        outs.append(attn @ V)
    return torch.stack(outs, dim=0)  # (N, 512)


def main():
    log("torch", torch.__version__, "| npu:", torch.npu.is_available())
    device = torch.device("npu")
    t = build(device)
    log("inputs built. q", tuple(t["ql_nope"].shape), "kv_cache", tuple(t["k_nope_cache"].shape))

    sel = [3, 17, 40, 61]  # 4 token ids in [0,64); scattered slots 0,2,4,6 < K
    log("selected ids:", sel)

    log(">>> COMPACTED (discrete=False) ...")
    out_c = run_op(t, compacted_topk(sel, device), discrete=False)
    log("COMPACTED ok. out", tuple(out_c.shape))

    log(">>> DISCRETE (discrete=True) ...")
    out_d = run_op(t, scatter_topk(sel, device), discrete=True)
    log("DISCRETE ok. out", tuple(out_d.shape))

    gold = cpu_golden(t, sel)                       # (N, 512)
    od = out_d.reshape(NUM_HEADS, D).float()
    oc = out_c.reshape(NUM_HEADS, D).float()
    dc = (od - oc).abs().amax().item()
    dg = (od - gold).abs().amax().item()
    peak = gold.abs().amax().item()
    log(f"max|discrete - compacted| = {dc:.3e}")
    log(f"max|discrete - golden|    = {dg:.3e}  (peak|gold|={peak:.3e})")
    ok = dc < 2e-2 and dg < 3e-2
    log("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
