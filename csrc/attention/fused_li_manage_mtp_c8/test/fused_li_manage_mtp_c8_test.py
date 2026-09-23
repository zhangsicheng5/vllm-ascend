#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fused_li_manage_mtp_c8 上板功能验证（nanovllm fused_li_manage_mtp 迁移一致性）。

golden 从源算子 kernel 数学推导（baseline/op_kernel/lightning_indexer_*.h）:
    S[t,n,i] = relu(sum_d q[t,n,d] * k[i,d])     # fixpipe reluPre=1, L0C fp32
    score[t,i] = sum_n w[t,n] * S[t,n,i]         # vector DoScale + DoReduce
打分范围: state=-1/-2 -> [0, offload_seq); state=-3 -> causal [0, act_k - routes + t + 1)

验证内容:
  1. 每 route top-2048 集合 == golden top-2048 集合（miss 前缀/hit 后缀分别对比）
  2. miss_counts / topk_miss_counts 与 golden 模拟一致（-1/-2 路径可精确模拟）
  3. 槽位守恒: pool 行有效槽位多重集合在置换前后不变, 且 miss_dst 互异、落点正确
  4. safe-failure case: topk 全 -1 且 miss=0（用哨兵区分"未写"）
"""
import json
import os
import sys

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401  注册 torch.ops._C_ascend

torch.npu.set_device(0)
DEV = "npu:0"
TOPK = 2048
BLOCK = 128
MISS_CAP = 32768
SENT = -1000003  # 哨兵: 区分 "写了 -1" 与 "未写"

CASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_li_manage_mtp_c8.json")


def load_cases():
    cases = []
    with open(CASE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def golden_score(q32, w32, k_tokens, valid_len):
    """score[t, i] = sum_n w[t,n] * relu(q[t,n,:] . k[i,:])  for i < valid_len[t] (broadcast mask).

    q32: [T, H, D] fp32; w32: [T, H] fp32; k_tokens: [L, D] fp32 (block-table 展开).
    返回 [T, L] fp32, 无效位置 -inf.
    """
    T, H, D = q32.shape
    S = torch.einsum("tnd,id->tni", q32, k_tokens)  # [T,H,L] fp32
    S = torch.relu(S)
    return torch.einsum("tn,tni->ti", w32, S)  # [T,L]


def golden_topk(score, valid_lens):
    """返回 list of (topk_src list, tie_free flag)。tie: golden 第 2047/2048 名严格不等。"""
    out = []
    for t in range(score.shape[0]):
        v = score[t, : valid_lens[t]].clone()
        k = min(TOPK, v.numel())
        topv, topi = torch.topk(v, k)
        # kernel 语义: 值降序, 等值按源 ID 升序(稳定排序, topk 返回的 topi 天然升序)
        order = torch.argsort(-topv, stable=True)
        src = topi[order].tolist()
        tie_free = k == TOPK and float(topv[k - 1]) > float(topv[k - 2]) if k >= 2 else False
        out.append((src, bool(tie_free)))
    return out


def build_case_inputs(c, gen):
    heads = c["heads"]
    B = c["batch"]
    routes = c["routes"]
    states = c["state"]
    cand = c["candidate"]
    cache = c["cache"]
    actk = c["actual_key"]
    blocks = c["blocks"]
    dt = torch.bfloat16 if c["dtype"] == "bfloat16" else torch.float16

    T = sum(routes)
    # 前缀和 actual_seq_lengths_query
    aslq = torch.tensor([sum(routes[:i + 1]) for i in range(B)], dtype=torch.int32)
    ask = torch.tensor(actk, dtype=torch.int32)
    offl = torch.tensor(cand, dtype=torch.int32)
    nct = torch.tensor(cache, dtype=torch.int32)
    rstate = torch.tensor(states, dtype=torch.int32)
    rpe = torch.arange(B, dtype=torch.int32)

    query = (torch.randn(T, heads, 128, generator=gen) * 0.5).to(dt)
    weights = (torch.randn(T, heads, generator=gen) * 0.3).to(dt)
    q_scale = torch.rand(T, heads, generator=gen)  # fp32 (源 def, kernel 不消费)
    k_scale = torch.rand(blocks, BLOCK, 1, generator=gen)
    key_cache = (torch.randn(blocks, BLOCK, 1, 128, generator=gen) * 0.5).to(dt)
    block_table = torch.arange(blocks, dtype=torch.int32).view(1, -1).repeat(B, 1)

    pool = torch.empty(B, blocks * BLOCK, dtype=torch.int32)
    for b in range(B):
        if states[b] == -1 and cache[b] > 0:
            # 标准稳态行: [0, C) 恒等有效 + [C, N) 无效(-1)。
            # kernel 的 -1 过渡探针 slots[C] < C 判定非恒等残留, 跳过清理 —— 正是稳态形态
            cap = blocks * BLOCK
            pool[b] = torch.arange(cap, dtype=torch.int32)
            pool[b][cache[b]:] = -1
        else:
            # -3 恒等填充 / -2 first-decode(全清重建)
            pool[b] = torch.arange(blocks * BLOCK, dtype=torch.int32)

    outs = {
        "topk_src": torch.full((T, 1, TOPK), SENT, dtype=torch.int32),
        "topk_dst": torch.full((T, 1, TOPK), SENT, dtype=torch.int32),
        "topk_miss": torch.full((T,), SENT, dtype=torch.int32),
        "miss_src": torch.full((B, MISS_CAP), SENT, dtype=torch.int32),
        "miss_dst": torch.full((B, MISS_CAP), SENT, dtype=torch.int32),
        "miss_cnt": torch.full((B,), SENT, dtype=torch.int32),
    }
    ins = {
        "query": query, "weights": weights, "q_scale": q_scale, "k_scale": k_scale,
        "key_cache": key_cache, "block_table": block_table, "aslq": aslq, "ask": ask,
        "offl": offl, "nct": nct, "rstate": rstate, "rpe": rpe, "pool": pool,
    }
    return ins, outs


def to_dev(ins, outs):
    d_ins = {k: v.to(DEV) for k, v in ins.items()}
    d_outs = {k: v.to(DEV) for k, v in outs.items()}
    return d_ins, d_outs


def call_op(di, do):
    torch.ops._C_ascend.npu_fused_li_manage_mtp_c8(
        di["weights"], di["q_scale"], di["query"], di["k_scale"],
        di["key_cache"], di["block_table"], di["aslq"], di["ask"],
        di["offl"], di["nct"], di["rstate"], di["rpe"], di["pool"],
        do["topk_src"], do["topk_dst"], do["topk_miss"],
        do["miss_src"], do["miss_dst"], do["miss_cnt"])
    torch.npu.synchronize()


def expand_keys(key_cache, block_table_row, length):
    """按 block_table 展开前 length 个 token 的 key: [length, 128] fp32."""
    nblk = (length + BLOCK - 1) // BLOCK
    kc = key_cache[block_table_row[:nblk].long()]  # [nblk,128,1,128]
    return kc.view(-1, 128)[:length].float()


def check_case(c, gen):
    name = c["case"]
    heads, B = c["heads"], c["batch"]
    routes, states = c["routes"], c["state"]
    cand, cache, actk = c["candidate"], c["cache"], c["actual_key"]
    ins, outs = build_case_inputs(c, gen)
    pool_before = ins["pool"].clone()
    di, do = to_dev(ins, outs)
    call_op(di, do)

    topk_src = do["topk_src"].cpu()
    topk_dst = do["topk_dst"].cpu()
    topk_miss = do["topk_miss"].cpu()
    miss_src = do["miss_src"].cpu()
    miss_dst = do["miss_dst"].cpu()
    miss_cnt = do["miss_cnt"].cpu()
    pool_after = di["pool"].cpu()  # in-place output

    errs = []
    q32 = ins["query"].float()
    w32 = ins["weights"].float()
    # safe-failure request 的 topk 会被 kernel 重写为 -1(源语义), 跳过 golden 打分对比
    skip_golden = set()
    if c["expect"] == "safe_failure":
        skip_golden = {b for b in range(B) if states[b] == -1}
    t_base = 0
    for b in range(B):
        if b in skip_golden:
            t_base += routes[b]
            continue
        q_b = q32[t_base:t_base + routes[b]]
        w_b = w32[t_base:t_base + routes[b]]
        if states[b] == -3:
            valid = [actk[b] - routes[b] + t + 1 for t in range(routes[b])]
            k_tok = expand_keys(ins["key_cache"], ins["block_table"][b], max(valid))
        else:
            valid = [cand[b]] * routes[b]
            k_tok = expand_keys(ins["key_cache"], ins["block_table"][b], cand[b])
        score = golden_score(q_b, w_b, k_tok, valid)
        gtop = golden_topk(score, valid)

        for t in range(routes[b]):
            gi = t_base + t
            ksrc = topk_src[gi, 0].tolist()
            gsrc, tie_free = gtop[t]
            # 集合对比 (golden 去重: topk 索引天然唯一)
            if set(ksrc[:len(gsrc)]) != set(gsrc):
                miss_set = set(ksrc) - set(gsrc)
                extra_set = set(gsrc) - set(ksrc)
                errs.append(f"[{name}] b{b} t{t} topk集合不一致: kernel多{sorted(miss_set)[:5]} golden多{sorted(extra_set)[:5]}")
            elif tie_free and ksrc[:len(gsrc)] != gsrc:
                # 顺序对比仅在 tie-free
                diff = [i for i in range(len(gsrc)) if ksrc[i] != gsrc[i]]
                errs.append(f"[{name}] b{b} t{t} topk顺序不一致(tie-free), 首个差异位 {diff[0]}: k={ksrc[diff[0]]} g={gsrc[diff[0]]}")

        if states[b] == -3:
            # miss=0, pool 恒等, topk_dst == 恒等 src
            if int(topk_miss[t_base:t_base + routes[b]].max()) != 0 or int(miss_cnt[b]) != 0:
                errs.append(f"[{name}] b{b} standard 路径 miss 应为 0, got topk_miss={topk_miss[t_base:t_base+routes[b]].tolist()} miss_cnt={miss_cnt[b].item()}")
            for t in range(routes[b]):
                dst = topk_dst[t_base + t, 0]
                src = topk_src[t_base + t, 0]
                if not torch.equal(dst, src):
                    errs.append(f"[{name}] b{b} t{t} standard topk_dst 应恒等于 src")
        t_base += routes[b]

    # safe failure / steady / first-decode 分路径断言
    for b in range(B):
        C = cache[b]
        if c["expect"] == "safe_failure" and states[b] == -1:
            tb, te = (sum(routes[:b]), sum(routes[:b + 1]))
            if int((topk_src[tb:te] == -1).all()) != 1 or int(miss_cnt[b]) != 0:
                errs.append(f"[{name}] b{b} safe-failure 契约: topk 应全 -1 且 miss=0, got miss={miss_cnt[b].item()}")
            continue
        if states[b] == -2:
            C = cache[b]
            if C == 0:
                continue
            mc = int(miss_cnt[b])
            if mc != C:
                errs.append(f"[{name}] b{b} first-decode miss_cnt 应为 C={C}, got {mc}")
            # golden: union of 4 routes' topk, 升序取 C 个, 不足补非 union 小 source
            tb = sum(routes[:b])
            union = set()
            for t in range(routes[b]):
                gsrc, _ = golden_topk(
                    golden_score(q32[tb + t:tb + t + 1], w32[tb + t:tb + t + 1],
                                 expand_keys(ins["key_cache"], ins["block_table"][b], cand[b]),
                                 [cand[b]]), [cand[b]])[0]
                union.update(gsrc)
            resident = sorted(union)
            if len(resident) < C:
                s = 0
                pool_ids = set(resident)
                while len(resident) < C:
                    if s not in pool_ids:
                        resident.append(s)
                    s += 1
            got = miss_src[b, :mc].tolist()
            if got != resident:
                errs.append(f"[{name}] b{b} first-decode miss_src 列表不一致: 前5 got={got[:5]} exp={resident[:5]}")
            if miss_dst[b, :mc].tolist() != list(range(C)):
                errs.append(f"[{name}] b{b} first-decode miss_dst 应为恒等 [0,C)")
            if int(topk_miss[tb:tb + routes[b]].min()) != TOPK:
                errs.append(f"[{name}] b{b} first-decode topk_miss 应全 2048")
        if states[b] == -1:
            C = cache[b]
            tb = sum(routes[:b])
            # golden miss: 各 route golden topk 中 source >= C 的并集 (初始池 [0,C) 恒等有效)
            union_miss = set()
            for t in range(routes[b]):
                gsrc, _ = golden_topk(
                    golden_score(q32[tb + t:tb + t + 1], w32[tb + t:tb + t + 1],
                                 expand_keys(ins["key_cache"], ins["block_table"][b], cand[b]),
                                 [cand[b]]), [cand[b]])[0]
                union_miss.update(s for s in gsrc if s >= C)
            mc = int(miss_cnt[b])
            if mc != len(union_miss):
                errs.append(f"[{name}] b{b} steady miss_cnt 应为 {len(union_miss)}, got {mc}")
            else:
                got = sorted(miss_src[b, :mc].tolist())
                if got != sorted(union_miss):
                    errs.append(f"[{name}] b{b} steady miss_src 并集不一致")
                dsts = miss_dst[b, :mc].tolist()
                if len(set(dsts)) != mc or any(d < 0 or d >= C for d in dsts):
                    errs.append(f"[{name}] b{b} steady miss_dst 应互异且在 [0,C)")
            # 槽位守恒: 置换前后有效槽位多重集合不变
            before = pool_before[b][pool_before[b] >= 0].tolist()
            after = pool_after[b][pool_after[b] >= 0].tolist()
            if sorted(before) != sorted(after):
                errs.append(f"[{name}] b{b} 槽位守恒破坏: before={sorted(before)[:5]}... after={sorted(after)[:5]}...")
            # miss 落点: pool[miss_src[k]] == miss_dst[k]
            for k in range(mc):
                if int(pool_after[b, miss_src[b, k].item()]) != int(miss_dst[b, k]):
                    errs.append(f"[{name}] b{b} miss 落点错误 @k={k}")
                    break

    return errs


def main():
    gen = torch.Generator().manual_seed(20260902)
    cases = load_cases()
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    total_err = 0
    for c in cases:
        if only and c["case"] not in only:
            continue
        print(f"== case {c['case']} ==", flush=True)
        try:
            errs = check_case(c, gen)
        except Exception as e:  # noqa: BLE001
            errs = [f"EXCEPTION: {type(e).__name__}: {e}"]
        if errs:
            total_err += len(errs)
            for e in errs:
                print(f"  FAIL {e}")
        else:
            print("  PASS")
    print(f"\n===== {'ALL PASS' if total_err == 0 else f'{total_err} FAILURES'} =====")
    return 0 if total_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
