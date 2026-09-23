#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fused_li_manage_mtp_c8 源 UT 场景矩阵移植验证。

参考: nanovllm-DSA-offload/ut_ops/test_fused_li_manage_mtp.py（源算子 UT，2089 行）
目标: 把源 UT 覆盖的全部精度场景在 _c8 ABI（13入/7出, pool 原地更新）上等价验证。

移植 mode（与源 UT 一一对应）:
  correctness        Q=[1..7] × 4 state 组合（-3*/-2*/-1*/mixed）
  replacement        -1 稳态 L>C 驱逐: (Q,L,C)=(1,8320,8192)/([1,2,3],8320,8192)/
                     (4,16256,12288)/(7,16256,14336), capacity=16384
  first_decode       -2 冷启动(L>C) + -3 恒等行复用(-3 调用后原 cache 张量转 -2)
  lifecycle          状态链 (-2,-1,-1)/(-3,-1)/(-3,-2,-1) × Q=[1,4,7](B=3),
                     末态 (-1,-1) 断言 topk_miss=0 & miss=0
  steady_strict      9 场景严格稳态语义: 精确 union 控制 + slot-0 强制 + 随机
                     block_table + 镜像重构逐元素比对 + 重复调用不动点
  long               21-bit source ID: L=2^18+2048, capacity=L+128, Q=1/4/7 ×
                     C=8192/12288/14336 × states -3/-2/-1 + -3→-2 + lifecycle + mixed
  key_tag            3-bit source_high 分数标签: 2^18 边界等值并列截断 + host ULP 枚举
  invalid            13 组元数据值域违例(kernel safe-failure) + fp16 scale(RuntimeError)
                     + 结构违例子进程探针

与源 UT 的方法论差异（均已在验证报告注明）:
  1. 参考实现用 CPU golden(fp32 作用于 bf16 输入)替代官方 npu_lightning_indexer
     （官方对拍已在 ext_test 覆盖）; -3 有序对比仅在 golden 2047/2048 名严格不等
     (tie_free)时进行，避免 fp32 golden 与 kernel bf16 打分在内部并列处的顺序误报。
  2. key_tag 用例用解析式 golden（band 内并列分数精确可计算）做全集+全序对比。
  3. 源 UT 的 host 侧 validate_dynamic_inputs 在 c8 ABI 不存在（无 total_queries
     输入）: 值域违例由 kernel ValidateOffloadRequest 拦截(safe-failure);
     aslq 与 T 不一致属结构违例, kernel -1 路径无 tSize 钳制, 只做隔离子进程探针。
"""
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_li_manage_mtp_c8_test import (  # noqa: E402
    BLOCK, DEV, MISS_CAP, SENT, TOPK, expand_keys, golden_score, golden_topk,
    to_dev,
)
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402  注册 torch.ops._C_ascend

torch.npu.set_device(0)

INVALID_SLOT = -(1 << 31)   # kernel 驱逐哨兵(INT32_MIN); -1 亦被符号位判别接受
PACKED = 1 << 17            # 21-bit source ID 的高 4 位边界 ([slot15|src17])
HEADS = 32
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


# ============================================================= case 构建
def build_src_case(q_values, states, *, offload_len, cache_tokens=8192,
                   source_capacity=16384, heads=32, dtype="bf16", seed=7,
                   correlated=False, noise=None, shared_w=False,
                   rpe=None, pool_rows=None):
    """源 UT build_case 的 c8 移植: 元数据/池形态/张量构造逐条对齐。

    - actual_key = offload_len + 128（最小合法 actk, 守卫 L ≤ ⌊(actK−Q)/128⌋·128 恰好取等）
    - cache_tokens: offload ≤ Q·2048 时强制 C=L（守卫分支一）, 否则用入参 C
    - pool_size = 2B+1, req_pool_entries = [1,3,5,...]（刻意非连续）
    - state=-1 行预置 [0,C) 恒等 + 其余 INVALID_SLOT；-3/-2 行全 INVALID_SLOT
    """
    assert len(q_values) == len(states)
    B, T = len(q_values), sum(q_values)
    dt = DTYPE[dtype]
    actk = [offload_len + BLOCK] * B
    offl = [offload_len] * B
    nct = [offload_len if offload_len <= q * TOPK else cache_tokens
           for q in q_values]
    query_ends = [sum(q_values[:i + 1]) for i in range(B)]
    rpe = rpe if rpe is not None else [2 * r + 1 for r in range(B)]
    pool_rows = pool_rows if pool_rows is not None else 2 * B + 1

    gen = torch.Generator().manual_seed(seed)
    if correlated:
        base = torch.randn(heads, 128, generator=gen)
        query = (base.unsqueeze(0) + noise * torch.randn(T, heads, 128, generator=gen)).to(dt)
    else:
        query = torch.randn(T, heads, 128, generator=gen).to(dt)
    if shared_w:
        w1 = (torch.randn(heads, generator=gen) * 0.3).to(dt)
        weights = w1.unsqueeze(0).expand(T, heads).contiguous()
    else:
        weights = (torch.randn(T, heads, generator=gen)).to(dt)
    blocks = source_capacity // BLOCK
    key = (torch.randn(blocks, BLOCK, 1, 128, generator=gen) * 0.5).to(dt)
    block_table = torch.arange(blocks, dtype=torch.int32).repeat(B, 1)
    q2r = [r for r, q in enumerate(q_values) for _ in range(q)]
    route_table = block_table[torch.tensor(q2r)].contiguous()

    pool = torch.full((pool_rows, source_capacity), INVALID_SLOT, dtype=torch.int32)
    for b, st in enumerate(states):
        if st == -1:
            pool[rpe[b], :nct[b]] = torch.arange(nct[b], dtype=torch.int32)

    case = {
        "q_values": q_values, "states": list(states), "heads": heads,
        "query_ends": query_ends, "actual_key": actk, "offload_key": offl,
        "cache_tokens": nct, "req_entries": list(rpe), "pool_rows": pool_rows,
        "source_capacity": source_capacity,
        "query": query, "weights": weights,
        "q_scale": torch.rand(T, heads, generator=gen),
        "k_scale": torch.rand(blocks, BLOCK, 1, generator=gen),
        "key": key, "block_table": block_table, "route_table": route_table,
        "pool": pool,
        "metadata": {
            "aslq": torch.tensor(query_ends, dtype=torch.int32),
            "ask": torch.tensor(actk, dtype=torch.int32),
            "offl": torch.tensor(offl, dtype=torch.int32),
            "nct": torch.tensor(nct, dtype=torch.int32),
            "rstate": torch.tensor(states, dtype=torch.int32),
            "rpe": torch.tensor(rpe, dtype=torch.int32),
        },
    }
    return case


def make_outputs(case):
    T = case["query"].size(0)
    B = len(case["q_values"])
    return {
        "topk_src": torch.full((T, 1, TOPK), SENT, dtype=torch.int32),
        "topk_dst": torch.full((T, 1, TOPK), SENT, dtype=torch.int32),
        "topk_miss": torch.full((T,), SENT, dtype=torch.int32),
        "miss_src": torch.full((B, MISS_CAP), SENT, dtype=torch.int32),
        "miss_dst": torch.full((B, MISS_CAP), SENT, dtype=torch.int32),
        "miss_cnt": torch.full((B,), SENT, dtype=torch.int32),
    }


def to_dev_case(case, outs):
    di = {k: case[k].to(DEV) for k in
          ("query", "weights", "q_scale", "k_scale", "key", "block_table", "pool")}
    di.update({k: v.to(DEV) for k, v in case["metadata"].items()})
    do = {k: v.to(DEV) for k, v in outs.items()}
    return di, do


def call_c8(di, do):
    torch.ops._C_ascend.npu_fused_li_manage_mtp_c8(
        di["weights"], di["q_scale"], di["query"], di["k_scale"],
        di["key"], di["block_table"], di["aslq"], di["ask"],
        di["offl"], di["nct"], di["rstate"], di["rpe"], di["pool"],
        do["topk_src"], do["topk_dst"], do["topk_miss"],
        do["miss_src"], do["miss_dst"], do["miss_cnt"])
    torch.npu.synchronize()


def rerun(di, do, states=None):
    """重置输出哨兵后调用; 可选变更 rstate（lifecycle 用）。返回 (res, old_pool_cpu)。"""
    if states is not None:
        di["rstate"].copy_(torch.tensor(states, dtype=torch.int32, device=DEV))
    old_pool = di["pool"].cpu().clone()
    for v in do.values():
        v.fill_(SENT)
    call_c8(di, do)
    res = {k: v.cpu() for k, v in do.items()}
    res["pool_after"] = di["pool"].cpu()
    return res, old_pool


# ============================================================= golden 参考
def visible_lengths(case):
    out = []
    for b, q in enumerate(case["q_values"]):
        st = case["states"][b]
        for t in range(q):
            out.append(case["actual_key"][b] - (q - 1 - t) if st == -3
                       else case["offload_key"][b])
    return out


def golden_reference(case, route_table=None):
    """CPU golden topk: [T, TOPK] int64(不足补 -1) + tie_free 标志列表。"""
    rt = route_table if route_table is not None else case["route_table"]
    vis = visible_lengths(case)
    q32, w32, kc = case["query"].float(), case["weights"].float(), case["key"]
    rows, tie_free = [], []
    for t in range(q32.shape[0]):
        k_tok = expand_keys(kc, rt[t], vis[t])
        score = golden_score(q32[t:t + 1], w32[t:t + 1], k_tok, [vis[t]])
        src, tf = golden_topk(score, [vis[t]])[0]
        row = torch.full((TOPK,), -1, dtype=torch.int64)
        row[:len(src)] = torch.tensor(src, dtype=torch.int64)
        rows.append(row)
        tie_free.append(tf)
    return torch.stack(rows), tie_free


def check_route_topk(label, route, ksrc, ref_row, tie_free, errs, ordered=False):
    """集合对比必做; 有序对比: -3 路径(源 UT 语义)在 tie_free 时做。"""
    k = ksrc[:TOPK] if isinstance(ksrc, list) else ksrc[:TOPK].tolist()
    r = ref_row.tolist()
    if sorted(k) != sorted(r):
        extra = sorted(set(k) - set(r))[:5]
        miss = sorted(set(r) - set(k))[:5]
        errs.append(f"[{label}] route{route} topk 集合不一致: kernel多{extra} golden多{miss}")
        return
    if ordered and tie_free and k != r:
        d = next(i for i in range(TOPK) if k[i] != r[i])
        errs.append(f"[{label}] route{route} topk 顺序不一致(tie-free) @pos{d}: "
                    f"k={k[d]} g={r[d]}")


# ============================================================= 通用正确性断言
def assert_correctness_c8(case, res, old_pool, errs, label, ref):
    """源 UT assert_correctness 移植: -1/-2/-3 全语义 + 池态断言。

    old_pool: 调用前池快照(CPU); res: rerun 返回的输出字典;
    ref: golden_reference(case) 返回的 (rows, tie_free)。
    """
    ref_rows, tie_free = ref
    src, dst, route_miss = res["topk_src"], res["topk_dst"], res["topk_miss"]
    miss_src, miss_dst, miss_cnt = res["miss_src"], res["miss_dst"], res["miss_cnt"]
    cache_cpu = res["pool_after"]
    tb = 0
    for b, q in enumerate(case["q_values"]):
        st = case["states"][b]
        row = case["req_entries"][b]
        length = case["actual_key"][b] if st == -3 else case["offload_key"][b]
        for t in range(q):
            route = tb + t
            valid = min(length, TOPK)
            check_route_topk(label, route, src[route, 0].tolist(),
                             ref_rows[route], tie_free[route], errs,
                             ordered=(st == -3))
            if valid < TOPK:
                if not bool((src[route, 0, valid:] == -1).all()):
                    errs.append(f"[{label}] route{route} 有效域外应填 -1")
            if st == -3:
                if int(route_miss[route]) != 0:
                    errs.append(f"[{label}] -3 route{route} topk_miss 应 0, "
                                f"got {int(route_miss[route])}")
                if not torch.equal(dst[route, 0], src[route, 0]):
                    errs.append(f"[{label}] -3 route{route} topk_dst 应恒等 src")
            else:
                for pos in range(TOPK):
                    s = int(src[route, 0, pos])
                    if s >= 0 and int(dst[route, 0, pos]) != int(cache_cpu[row, s]):
                        errs.append(f"[{label}] route{route} pos{pos} dst != 池终态映射")
                        break
        if st == -3:
            if int(miss_cnt[b]) != 0:
                errs.append(f"[{label}] -3 b{b} miss_cnt 应 0, got {int(miss_cnt[b])}")
            # PrepareNonOffloadRows: -3 行整行重写为恒等 [0, capacity)
            if not torch.equal(cache_cpu[row],
                               torch.arange(case["source_capacity"], dtype=torch.int32)):
                errs.append(f"[{label}] -3 b{b} 池行应整行恒等 arange")
        elif st == -2:
            C = case["cache_tokens"][b]
            if int(miss_cnt[b]) != C:
                errs.append(f"[{label}] -2 b{b} miss_cnt 应 C={C}, got {int(miss_cnt[b])}")
            else:
                if not torch.equal(miss_dst[b, :C], torch.arange(C, dtype=torch.int32)):
                    errs.append(f"[{label}] -2 b{b} miss_dst 应恒等 [0,C)")
                union = torch.unique(src[tb:tb + q].reshape(-1), sorted=True)
                union = union[union >= 0]
                selected = torch.zeros(length, dtype=torch.bool)
                selected[union.to(torch.int64)] = True
                remainder = torch.arange(length, dtype=torch.int32)[~selected]
                expected = torch.cat((union.to(torch.int32), remainder))[:C]
                if not torch.equal(miss_src[b, :C], expected):
                    errs.append(f"[{label}] -2 b{b} miss_src 应=排序union+补齐前C "
                                f"got[:5]={miss_src[b, :5].tolist()} exp[:5]={expected[:5].tolist()}")
                if not bool((route_miss[tb:tb + q] == TOPK).all()):
                    errs.append(f"[{label}] -2 b{b} topk_miss 应全 {TOPK}")
        else:
            # -1 旧池快照: 复刻 kernel PrepareNonOffloadRows 的恒等残留探针
            # (slots[C] >= C 判定 -3 恒等行, 清理 [C, capacity) 后再进入稳态)
            old_row = old_pool[row].clone()
            C = case["cache_tokens"][b]
            if int(old_row[C]) >= C:
                old_row[C:] = INVALID_SLOT
            union = torch.unique(src[tb:tb + q].reshape(-1), sorted=True)
            union = union[(union >= 0) & (union < length)]
            expected = union[old_row[union.to(torch.int64)] < 0]
            count = int(miss_cnt[b])
            if count != expected.numel():
                errs.append(f"[{label}] -1 b{b} miss_cnt 应 {expected.numel()}, got {count}")
            elif not torch.equal(miss_src[b, :count], expected.to(torch.int32)):
                errs.append(f"[{label}] -1 b{b} miss_src 应=排序并集(旧池非驻留)")
        if st in (-2, -1):
            C = case["cache_tokens"][b]
            rs = cache_cpu[row, :length]
            rs = rs[rs >= 0]
            if rs.numel() != C or not torch.equal(
                    torch.sort(rs).values, torch.arange(C, dtype=torch.int32)):
                errs.append(f"[{label}] b{b} state={st} 池驻留应恰为槽位双射 [0,{C}), "
                            f"got {rs.numel()} 驻留")
        tb += q
    # 未参与行不写
    for r in range(case["pool_rows"]):
        if r not in case["req_entries"]:
            if not torch.equal(res["pool_after"][r], old_pool[r]):
                errs.append(f"[{label}] 未参与行 {r} 被改写")
                break


# ============================================================= mode: correctness
REPLACEMENT = (([1], 8320, 8192), ([1, 2, 3], 8320, 8192),
               ([4], 16256, 12288), ([7], 16256, 14336))


def run_correctness(dtype="bf16", seed=7):
    q_values = [1, 2, 3, 4, 5, 6, 7]
    patterns = [
        [-3] * 7, [-2] * 7, [-1] * 7,
        [(-3, -2, -1)[i % 3] for i in range(7)],
    ]
    for states in patterns:
        case = build_src_case(q_values, states, offload_len=8192,
                              cache_tokens=8192, dtype=dtype, seed=seed)
        ref = golden_reference(case)
        di, do = to_dev_case(case, make_outputs(case))
        res, old = rerun(di, do)
        errs = []
        assert_correctness_c8(case, res, old, errs, f"corr-{states}", ref)
        report(f"correctness q={q_values} states={states} dtype={dtype}", errs)


# ============================================================= mode: replacement
def run_replacement(dtype="bf16"):
    for q_values, L, C in REPLACEMENT:
        case = build_src_case(q_values, [-1] * len(q_values), offload_len=L,
                              cache_tokens=C, dtype=dtype)
        ref = golden_reference(case)
        di, do = to_dev_case(case, make_outputs(case))
        res, old = rerun(di, do)
        errs = []
        assert_correctness_c8(case, res, old, errs, f"repl-{q_values}", ref)
        report(f"replacement regression q={q_values} state=-1 L={L} C={C}", errs)


# ============================================================= mode: first_decode
def run_first_decode(dtype="bf16"):
    for q_values, L, C in REPLACEMENT:
        # 1) 冷启动: 全 INVALID 行
        case = build_src_case(q_values, [-2] * len(q_values), offload_len=L,
                              cache_tokens=C, dtype=dtype)
        ref = golden_reference(case)
        di, do = to_dev_case(case, make_outputs(case))
        res, old = rerun(di, do)
        errs = []
        assert_correctness_c8(case, res, old, errs, f"fd-cold-{q_values}", ref)
        report(f"first-decode regression q={q_values} state=-2 L={L} C={C} seed=invalid", errs)

        # 2) -3 恒等行复用: 先 -3（整行恒等化）再用同一 cache 张量转 -2
        case3 = build_src_case(q_values, [-3] * len(q_values), offload_len=L,
                               cache_tokens=C, dtype=dtype)
        ref3 = golden_reference(case3)
        di, do = to_dev_case(case3, make_outputs(case3))
        res3, old3 = rerun(di, do)
        errs = []
        assert_correctness_c8(case3, res3, old3, errs, f"fd-std-{q_values}", ref3)
        report(f"first-decode pre-step q={q_values} state=-3 L={L} C={C}", errs)
        case2 = build_src_case(q_values, [-2] * len(q_values), offload_len=L,
                               cache_tokens=C, dtype=dtype)
        res2, old2 = rerun(di, do, states=[-2] * len(q_values))
        errs = []
        assert_correctness_c8(case2, res2, old2, errs, f"fd-trans-{q_values}", ref)
        report(f"first-decode regression q={q_values} state=-3->-2 L={L} C={C}", errs)


# ============================================================= mode: lifecycle
LIFECYCLE_SEQS = ((-2, -1, -1), (-3, -1), (-3, -2, -1))


def run_lifecycle_scenario(q_values, *, offload_len, cache_tokens,
                           source_capacity, correlated=False, noise=None,
                           shared_w=False, seed=7, dtype="bf16", tag=""):
    for sequence in LIFECYCLE_SEQS:
        case = build_src_case(q_values, [sequence[0]] * len(q_values),
                              offload_len=offload_len, cache_tokens=cache_tokens,
                              source_capacity=source_capacity, seed=seed,
                              correlated=correlated, noise=noise, shared_w=shared_w,
                              dtype=dtype)
        di, do = to_dev_case(case, make_outputs(case))
        last = None
        for st in sequence:
            states = [st] * len(q_values)
            case["states"] = list(states)
            # 参考按当步 state 重算: -3 打分域=causal actk, -1/-2=offload(源 UT 不做
            # 每步正确性断言, 本移植加强为逐步断言, 故参考必须随 state 演进)
            ref = golden_reference(case)
            res, old = rerun(di, do, states=states)
            errs = []
            assert_correctness_c8(case, res, old, errs, f"lc-{sequence}-{st}", ref)
            if errs:
                report(f"lifecycle q={q_values} sequence={sequence} step={st}{tag}", errs)
                return
            last = res
        if sequence[-2:] == (-1, -1):
            if not bool((last["topk_miss"] == 0).all()):
                report(f"lifecycle q={q_values} sequence={sequence}{tag}",
                       ["末态 (-1,-1) topk_miss 应全 0"])
                continue
            if not bool((last["miss_cnt"] == 0).all()):
                report(f"lifecycle q={q_values} sequence={sequence}{tag}",
                       ["末态 (-1,-1) miss_counts 应全 0"])
                continue
        report(f"lifecycle q={q_values} sequence={sequence}{tag}", [])


def run_lifecycle():
    run_lifecycle_scenario([1, 4, 7], offload_len=8192, cache_tokens=8192,
                           source_capacity=16384)


# ============================================================= mode: steady_strict
STRICT_SCENARIOS = (
    ("mtp0", [1], 8320, 8192, 64),
    ("mixed-mtp0-2", [1, 2, 3], 8320, 8192, 64),
    ("mtp3-zero", [4], 16256, 12288, 0),
    ("mtp3-normal", [4], 16256, 12288, 300),
    ("mtp3-heavy", [4], 16256, 12288, 750),
    ("mtp3-union-over-2048", [4], 16256, 12288, 2304),
    ("mtp4", [5], 16256, 12288, 400),
    ("mtp5", [6], 16256, 12288, 400),
    ("mtp6", [7], 16256, 14336, 400),
)


def _rand_prefix(values, count, gen):
    if count == 0:
        return values[:0]
    return values[torch.randperm(values.numel(), generator=gen)[:count]]


def _controlled_union_row(union, length, C, target, gen):
    """返回 (miss_ids, cached_ids): 精确 target 个 union 非驻留 + 填充到 C。"""
    actual = min(target, int(union.numel()), length - C)
    missing = _rand_prefix(union, actual, gen)
    mask = torch.zeros(length, dtype=torch.bool)
    mask[missing] = True
    hits = union[~mask[union]]
    in_union = torch.zeros(length, dtype=torch.bool)
    in_union[union] = True
    fillers = torch.arange(length, dtype=torch.int64)[~in_union]
    need = C - hits.numel()
    if need < 0 or need > fillers.numel():
        raise AssertionError("受控池预算不可满足")
    return missing, torch.cat((hits, _rand_prefix(fillers, need, gen))), actual


def _randomize_block_table(case, gen):
    blocks = case["source_capacity"] // BLOCK
    B = len(case["q_values"])
    table = torch.stack([torch.randperm(blocks, generator=gen)
                         for _ in range(B)]).to(torch.int32)
    q2r = [r for r, q in enumerate(case["q_values"]) for _ in range(q)]
    case["block_table"] = table
    case["route_table"] = table[torch.tensor(q2r)].contiguous()


def _force_topk_hit_slot_zero(case, ref, pool):
    """把每 request 首个 topk 命中源换到 slot 0(不改变驻留集合)。"""
    pool = pool.clone()
    tb = 0
    for b, q in enumerate(case["q_values"]):
        row = case["req_entries"][b]
        sources = torch.unique(ref[tb:tb + q].reshape(-1), sorted=True)
        sources = sources[(sources >= 0) & (sources < case["offload_key"][b])]
        hits = sources[pool[row, sources] >= 0]
        if hits.numel() == 0:
            raise AssertionError(f"strict b{b} 无 topk 命中")
        zero_sources = torch.nonzero(pool[row] == 0).flatten()
        if zero_sources.numel() != 1:
            raise AssertionError(f"strict b{b} slot0 属主数 {zero_sources.numel()} != 1")
        hit_source, zero_source = int(hits[0]), int(zero_sources[0])
        if hit_source != zero_source:
            hit_slot = int(pool[row, hit_source])
            pool[row, zero_source] = hit_slot
            pool[row, hit_source] = 0
        tb += q
    return pool


def _validate_strict(case, ref, di, do):
    """源 UT _validate_strict_steady_semantics 移植: 镜像重构 + 逐元素比对。

    返回 (updated_pool_cpu, res, observed_unions); 违例抛 AssertionError。
    """
    old_cache = di["pool"].cpu().clone()
    for v in do.values():
        v.fill_(SENT)
    call_c8(di, do)
    src, dst, route_miss = do["topk_src"].cpu(), do["topk_dst"].cpu(), do["topk_miss"].cpu()
    miss_src, miss_dst, miss_cnt = (do["miss_src"].cpu(), do["miss_dst"].cpu(),
                                    do["miss_cnt"].cpu())
    expected_cache = old_cache.clone()
    expected_source_rows, expected_union_rows = [], []
    tb = 0
    for b, q in enumerate(case["q_values"]):
        row, length = case["req_entries"][b], case["offload_key"][b]
        C = case["cache_tokens"][b]
        request_misses, request_topk = [], []
        for t in range(q):
            route = tb + t
            ids = ref[route]
            ids = ids[(ids >= 0) & (ids < length)]
            if ids.numel() != TOPK or torch.unique(ids).numel() != TOPK:
                raise AssertionError(f"strict route{route} 参考行不合法")
            old_slots = old_cache[row, ids]
            misses = torch.sort(ids[old_slots < 0]).values
            hits = torch.sort(ids[old_slots >= 0]).values
            expected_sources = torch.cat((misses, hits)).to(torch.int32)
            if not torch.equal(src[route, 0], expected_sources):
                d = torch.nonzero(src[route, 0] != expected_sources).flatten()
                raise AssertionError(
                    f"strict b{b} route{t} topk 应=miss前缀(升序)+hit后缀(升序), "
                    f"首位差 {int(d[0]) if d.numel() else -1}, 期望miss={misses.numel()}")
            if int(route_miss[route]) != misses.numel():
                raise AssertionError(
                    f"strict b{b} route{t} topk_miss 应 {misses.numel()}, "
                    f"got {int(route_miss[route])}")
            expected_source_rows.append(expected_sources)
            request_misses.append(misses)
            request_topk.append(ids)
        expected_union = torch.unique(torch.cat(request_misses), sorted=True).to(torch.int32)
        expected_union_rows.append(expected_union)
        count = int(expected_union.numel())
        if int(miss_cnt[b]) != count:
            raise AssertionError(f"strict b{b} miss_cnt 应 {count}, got {int(miss_cnt[b])}")
        if not torch.equal(miss_src[b, :count], expected_union):
            raise AssertionError(f"strict b{b} miss_src 应=排序去重并集")
        victim_slots = miss_dst[b, :count]
        if count:
            bad = (victim_slots < 0) | (victim_slots >= C)
            if bool(bad.any()):
                i = int(torch.nonzero(bad)[0])
                raise AssertionError(f"strict b{b} victim 槽越界: {int(victim_slots[i])}")
            if torch.unique(victim_slots).numel() != count:
                raise AssertionError(f"strict b{b} victim 槽不互异")
            old_row = old_cache[row]
            res_src = torch.nonzero((old_row >= 0) & (old_row < C)).flatten()
            res_slot = old_row[res_src].to(torch.int64)
            if res_src.numel() != C or not torch.equal(
                    torch.sort(res_slot).values, torch.arange(C)):
                raise AssertionError(f"strict b{b} 种子池非槽位双射")
            slot2src = torch.empty(C, dtype=torch.int64)
            slot2src[res_slot] = res_src
            victim_sources = slot2src[victim_slots.to(torch.int64)]
            topk_union = torch.unique(torch.cat(request_topk), sorted=True)
            protected = torch.isin(victim_sources, topk_union)
            if bool(protected.any()):
                i = int(torch.nonzero(protected)[0])
                raise AssertionError(
                    f"strict b{b} 驱逐了 topk 保护源 {int(victim_sources[i])}")
            for incoming, victim, slot in zip(expected_union.tolist(),
                                              victim_sources.tolist(),
                                              victim_slots.tolist()):
                expected_cache[row, victim] = INVALID_SLOT
                expected_cache[row, incoming] = slot
        tb += q
    actual_cache = di["pool"].cpu()
    if not torch.equal(actual_cache, expected_cache):
        m = torch.nonzero(actual_cache != expected_cache)
        f = m[0].tolist()
        raise AssertionError(
            f"strict 池镜像不一致: 首差 [{f[0]},{f[1]}] "
            f"actual={int(actual_cache[f[0], f[1]])} exp={int(expected_cache[f[0], f[1]])}")
    for route, expected_sources in enumerate(expected_source_rows):
        b = next(r for r, e in enumerate(case["query_ends"]) if route < e)
        row = case["req_entries"][b]
        if not torch.equal(dst[route, 0],
                           expected_cache[row, expected_sources.to(torch.int64)]):
            raise AssertionError(f"strict route{route} topk_dst 应=池终态映射")
    if not bool((dst == 0).any()):
        raise AssertionError("strict 输出丢失合法槽位 0")
    res = {"topk_src": src, "topk_dst": dst, "topk_miss": route_miss,
           "miss_src": miss_src, "miss_dst": miss_dst, "miss_cnt": miss_cnt}
    return actual_cache.clone(), res, [int(r.numel()) for r in expected_union_rows]


def run_steady_strict():
    for idx, (label, q_values, L, C, target) in enumerate(STRICT_SCENARIOS):
        case = build_src_case(q_values, [-1] * len(q_values), offload_len=L,
                              cache_tokens=C)
        gen = torch.Generator().manual_seed(7 + 20011 + idx * 97)
        _randomize_block_table(case, gen)
        ref = golden_reference(case)[0]
        pool = torch.full_like(case["pool"], INVALID_SLOT)
        actual_unions = []
        tb = 0
        for b, q in enumerate(q_values):
            union = torch.unique(ref[tb:tb + q].reshape(-1), sorted=True)
            union = union[(union >= 0) & (union < L)]
            tb += q
            missing, cached, actual = _controlled_union_row(union, L, C, target, gen)
            pool[case["req_entries"][b], cached] = torch.randperm(
                C, generator=gen, dtype=torch.int64).to(torch.int32)
            actual_unions.append(actual)
        if actual_unions != [target] * len(q_values):
            raise AssertionError(
                f"strict {label} 受控 union={actual_unions} != 目标 {[target]*len(q_values)}")
        pool = _force_topk_hit_slot_zero(case, ref, pool)
        case["pool"] = pool
        di, do = to_dev_case(case, make_outputs(case))
        updated, _, observed = _validate_strict(case, ref, di, do)
        if observed != actual_unions:
            raise AssertionError(f"strict {label} 观测 union={observed} != {actual_unions}")
        # 重复调用: 全命中不动点
        _, repeat_res, repeat_unions = _validate_strict(case, ref, di, do)
        if repeat_unions != [0] * len(q_values):
            raise AssertionError(f"strict {label} 重复调用 union={repeat_unions} 非 0")
        if not torch.equal(di["pool"].cpu(), updated):
            raise AssertionError(f"strict {label} 重复调用改写了池")
        if not bool((repeat_res["topk_miss"] == 0).all()) or \
                not bool((repeat_res["miss_cnt"] == 0).all()):
            raise AssertionError(f"strict {label} 重复调用计数非 0")
        report(f"steady semantic regression scenario={label} q={q_values} L={L} C={C} "
               f"union={target} random_mapping=1 random_block_table=1 repeat_stable=1", [])


# ============================================================= mode: long
LONG_L = PACKED + 2048          # 264192: 高位 band 2048 宽, 期望 topk 覆盖 ≥2^18 ≈16 个
LONG_CAP = LONG_L + BLOCK


def _long_union_ok(case, ref, cmax):
    # 只有 -1 稳态受 union ≤ C 约束(victim 供给); -2/-3 无约束
    for b, q in enumerate(case["q_values"]):
        if case["states"][b] != -1:
            continue
        tb = sum(case["q_values"][:b])
        union = torch.unique(ref[tb:tb + q].reshape(-1), sorted=True)
        union = union[(union >= 0) & (union < case["offload_key"][b])]
        if union.numel() > cmax:
            return False
    return True


def build_long_case(q_values, states, cache_tokens, seed):
    """长源 case: 共享权重+相关 query 控制每 request union ≤ 0.9·C（种子阶梯重试）。"""
    for attempt in range(8):
        case = build_src_case(q_values, states, offload_len=LONG_L,
                              cache_tokens=cache_tokens, source_capacity=LONG_CAP,
                              correlated=True, noise=0.5, shared_w=True,
                              seed=seed + attempt * 13)
        ref_rows, tf = golden_reference(case)
        if _long_union_ok(case, ref_rows, int(cache_tokens * 0.9)):
            return case, (ref_rows, tf)
    raise AssertionError("long: 8 个种子仍未满足 union 预算")


def run_long():
    for q_values, C in (([1], 8192), ([4], 12288), ([7], 14336)):
        for states in ([-3], [-2], [-1]):
            case, ref = build_long_case(q_values, states, C, seed=7)
            if not bool((ref[0] >= PACKED).any()):
                raise AssertionError("long 参考未覆盖 ≥2^18 源(21-bit 高位路径未验证)")
            di, do = to_dev_case(case, make_outputs(case))
            res, old = rerun(di, do)
            errs = []
            assert_correctness_c8(case, res, old, errs, f"long-{states}", ref)
            report(f"long correctness q={q_values} states={states} L={LONG_L} C={C}", errs)
        # -3 恒等行 → -2 复用
        case3, ref3 = build_long_case(q_values, [-3], C, seed=7)
        di, do = to_dev_case(case3, make_outputs(case3))
        rerun(di, do)
        case2, ref2 = build_long_case(q_values, [-2], C, seed=7)
        res2, old2 = rerun(di, do, states=[-2] * len(q_values))
        errs = []
        assert_correctness_c8(case2, res2, old2, errs, "long-trans", ref2)
        report(f"long first-decode regression q={q_values} state=-3->-2 "
               f"L={LONG_L} C={C}", errs)
        run_lifecycle_scenario(q_values, offload_len=LONG_L, cache_tokens=C,
                               source_capacity=LONG_CAP, correlated=True,
                               noise=0.5, shared_w=True, seed=7,
                               tag=f" L={LONG_L} C={C}")
    case, ref = build_long_case([1, 4, 7], [-3, -2, -1], 14336, seed=7)
    if not bool((ref[0] >= PACKED).any()):
        raise AssertionError("long mixed 参考未覆盖 ≥2^18 源")
    di, do = to_dev_case(case, make_outputs(case))
    res, old = rerun(di, do)
    errs = []
    assert_correctness_c8(case, res, old, errs, "long-mixed", ref)
    report(f"long mixed correctness q=[1, 4, 7] states=[-3, -2, -1] "
           f"L={LONG_L} C=14336", errs)


# ============================================================= mode: key_tag
def _run_key_tag_ulp():
    base = 0x3F800000
    reordered = 0
    for delta in range(9):
        score_a, score_b = base, base + delta
        source_a, source_b = 0, PACKED
        exact_a_first = score_a > score_b or (score_a == score_b and source_a < source_b)
        tagged_a = (score_a & ~0x7) | 7
        tagged_b = (score_b & ~0x7) | 0
        reordered += int(exact_a_first != (tagged_a > tagged_b))
    print(f"KEY_TAG_ULP_DIAG tag_bits=3 ulp_deltas=0..8 pairs=9 "
          f"reordered_pairs={reordered} max_key_delta_ulp=7", flush=True)


def run_key_tag():
    _run_key_tag_ulp()
    upper = TOPK // 2 + BLOCK
    capacity = PACKED + upper                 # 263296
    offload = capacity - BLOCK                # 263168
    case = build_src_case([1], [-3], offload_len=offload, cache_tokens=8192,
                          source_capacity=capacity)
    query, weights, key = case["query"], case["weights"], case["key"]
    query.fill_(1.0)
    weights.fill_(1.0)
    key_by_src = key.reshape(capacity, 1, 128)
    begin, end = PACKED - TOPK // 2, PACKED + upper
    selected = None
    for value in (1, -1):   # 探针: 保留官方/golden 完全落在并列 band 且跨 2^18 的符号
        key.zero_()
        key_by_src[begin:end].fill_(float(value))
        ref = golden_reference(case)
        row = ref[0]
        if bool(((row >= begin) & (row < end)).all()) and bool((row >= PACKED).any()):
            selected = value
            break
    if selected is None:
        raise AssertionError("key-tag 无法把截断压进跨 2^18 等值候选带")
    # 解析式 golden: band 内并列满分(=heads*128), 稳定平局按源 ID 升序取前 2048
    expected = torch.arange(begin, begin + TOPK, dtype=torch.int64)
    di, do = to_dev_case(case, make_outputs(case))
    res, old = rerun(di, do)
    errs = []
    got = res["topk_src"][0, 0].to(torch.int64)
    if not torch.equal(got, expected):
        if torch.equal(torch.sort(got).values, torch.sort(expected).values):
            errs.append("key-tag topk 集合一致但并列平局顺序与 ID 升序不同")
        else:
            extra = sorted(set(got.tolist()) - set(expected.tolist()))[:5]
            miss = sorted(set(expected.tolist()) - set(got.tolist()))[:5]
            errs.append(f"key-tag topk 集合不一致: kernel多{extra} 期望多{miss}")
    if not bool((res["topk_dst"][0, 0] == res["topk_src"][0, 0]).all()):
        errs.append("key-tag -3 dst 应恒等 src")
    if int(res["topk_miss"][0]) != 0 or int(res["miss_cnt"][0]) != 0:
        errs.append("key-tag -3 miss 应 0")
    if not torch.equal(res["pool_after"][case["req_entries"][0]],
                       torch.arange(capacity, dtype=torch.int32)):
        errs.append("key-tag -3 池行应整行恒等")
    report(f"key-tag regression capacity={capacity} boundary={PACKED} "
           f"candidates={2 * (TOPK // 2) + BLOCK} value={selected}", errs)


# ============================================================= mode: invalid
INVALID_BASE = dict(q_values=[4], states=[-1], offload_len=8192,
                    cache_tokens=8192, source_capacity=16384, rpe=[0],
                    pool_rows=2)
INVALID_CASES = (
    # (字段, 值, 期望)  期望: safe=safe-failure(topk=-1/miss=0/池不动)
    ("aslq", [0], "safe"), ("aslq", [8], "safe"),
    ("ask", [3], "safe"), ("ask", [20000], "loose"),
    ("offl", [2047], "safe"), ("offl", [8256], "safe"),
    ("nct", [2000], "safe"), ("nct", [8064], "safe"),
    ("rstate", [0], "safe"), ("rstate", [-4], "safe"),
    ("rpe", [-1], "safe"), ("rpe", [2], "safe"),
    ("aslq", [5], "probe"),
)


def run_invalid():
    case = build_src_case(**INVALID_BASE)
    passed = 0
    for field, value, expect in INVALID_CASES:
        if expect == "probe":
            continue  # 结构违例只在隔离子进程中观测(见 _probe_structural_mismatch)
        di, do = to_dev_case(case, make_outputs(case))
        di[field] = torch.tensor(value, dtype=torch.int32, device=DEV)
        pool_before = di["pool"].cpu().clone()
        for v in do.values():
            v.fill_(SENT)
        try:
            call_c8(di, do)
        except RuntimeError as e:
            report(f"invalid {field}={value}", [f"调用异常: {e}"])
            continue
        src = do["topk_src"].cpu()
        mc = do["miss_cnt"].cpu()
        pool_after = di["pool"].cpu()
        errs = []
        if expect == "safe":
            if not bool((src == -1).all()):
                errs.append(f"topk 应全 -1, got 唯一值 {src.unique().tolist()[:4]}")
            if int(mc[0]) != 0:
                errs.append(f"miss_cnt 应 0, got {int(mc[0])}")
            if not torch.equal(pool_after, pool_before):
                errs.append("safe-failure 不应改写池")
        else:  # loose: ask=20000 值域外, kernel 无 actk≤capacity 守卫 -> 只断言无害
            if int(mc[0]) != 0:
                errs.append(f"loose miss_cnt 应 0(L=C 全驻留), got {int(mc[0])}")
            if not torch.equal(pool_after, pool_before):
                errs.append("loose: 池行应保持不变(无 miss 即无驱逐)")
        if not errs:
            passed += 1
        report(f"invalid {field}={value} expect={expect}", errs)
    # fp16 query_dequant_scale -> host TORCH_CHECK RuntimeError
    di, do = to_dev_case(case, make_outputs(case))
    di["q_scale"] = di["q_scale"].to(torch.float16)
    try:
        call_c8(di, do)
        report("invalid fp16 q_scale", ["host 应拒绝 fp16 query_dequant_scale"])
    except RuntimeError:
        passed += 1
        report("invalid fp16 query_dequant_scale expect=RuntimeError", [])
    # 结构违例(aslq 与 T 不一致): c8 无 host 校验且 -1 路径无 tSize 钳制,
    # 隔离子进程观测, 不作 PASS/FAIL 门槛
    info = _probe_structural_mismatch()
    print(f"invalid structural probe: {info}", flush=True)
    print(f"invalid PASS cases={passed}/{len(INVALID_CASES) - 1 + 1}", flush=True)


def _probe_structural_mismatch():
    """aslq=[5] 但 T=4: 子进程内单次调用, 观测是否崩溃/污染, 超时即终止。"""
    script = (
        "import sys, torch, torch_npu\n"
        "sys.path.insert(0, %r)\n"
        "import fused_li_manage_mtp_c8_src_test as S\n"
        "import vllm_ascend.vllm_ascend_C\n"
        "case = S.build_src_case([4], [-1], offload_len=8192, cache_tokens=8192,\n"
        "                        source_capacity=16384, rpe=[0], pool_rows=2)\n"
        "di, do = S.to_dev_case(case, S.make_outputs(case))\n"
        "di['aslq'] = torch.tensor([5], dtype=torch.int32, device=S.DEV)\n"
        "S.call_c8(di, do)\n"
        "print('probe-done miss_cnt=', int(do['miss_cnt'].cpu()[0]))\n"
    ) % os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=90,
                           env={**os.environ, "ASCEND_RT_VISIBLE_DEVICES":
                                os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")})
        out = (r.stdout + r.stderr).strip().splitlines()
        tail = repr(out[-1]) if out else "''"
        return f"exit={r.returncode} tail={tail}"
    except subprocess.TimeoutExpired:
        return "timeout(90s) —— 结构违例导致设备侧挂起, 已终止子进程"


# ============================================================= 框架
def report(msg, errs):
    if errs:
        for e in errs:
            print(f"  FAIL {e}", flush=True)
        print(f"  >> {msg}: FAIL", flush=True)
        FAILURES.append(msg)
    else:
        print(f"  {msg}: PASS", flush=True)


FAILURES = []


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    modes = {
        "correctness": lambda: (run_correctness(), run_correctness(dtype="fp16")),
        "replacement": lambda: (run_replacement(), run_replacement(dtype="fp16")),
        "first_decode": run_first_decode,
        "lifecycle": run_lifecycle,
        "steady_strict": run_steady_strict,
        "long": run_long,
        "key_tag": run_key_tag,
        "invalid": run_invalid,
    }
    for name, fn in modes.items():
        if only not in ("all", name):
            continue
        print(f"\n===== mode {name} =====", flush=True)
        fn()
    print(f"\n===== {'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURES'} =====",
          flush=True)
    for m in FAILURES:
        print(f"  FAILED: {m}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
