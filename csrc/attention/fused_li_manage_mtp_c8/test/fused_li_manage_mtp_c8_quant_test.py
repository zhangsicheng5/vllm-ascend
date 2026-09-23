#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fused_li_manage_mtp_c8 真实 C8（int8+fp16 scale）精度验证: A/B/C 三口径。

背景: 本算子此前是 C8 接口占位（scale 被 (void)）；现已实现商用 quant_li 同构的
两段 mma 量化通路。本脚本按方案二验收口径验证:

  A = CPU golden(fp32 作用于原 bf16 输入)                        —— 无量化基准
  B = npu_fused_li_manage_mtp_c8(int8 q/k + fp16 双 scale)        —— 被测算子
  C = 同一 CPU golden 作用于反量化重建的 bf16 输入                —— 量化带内等价基准

  B 的核内数学闭式仿真(带 fp16 舍入点):
    score[t,i] = ( Σ_h fp16(relu(int8dot[t,h,i]) * 2^-10) * fp16_rint(w[t,h]*qs[t,h]) ) * ks[i]

断言:
  1) B vs C  topk 集合重合率 >= 99.5%   (kernel 数学等价, 预期 ~99.7%+)
  2) B vs A  topk 集合重合率 >= 98.5%   (总量化带, 对齐商用 quant li 实测 98.8%)
  3) tie-free 行上 B 与量化闭式 golden 全序一致
  4) 守卫回归: fp32 scale / bf16 query / fp16 weights -> RuntimeError
  5) 幂等: 同输入重复调用输出逐位一致

场景: mixed states(Q=1/3/7) / first_decode(-2) / steady(-1 L>C) /
      long(L=2^18+2048) / H=32 主形态 + H=64 抽查。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_li_manage_mtp_c8_test import (  # noqa: E402
    BLOCK, DEV, SENT, TOPK, expand_keys, golden_score, golden_topk,
)
from fused_li_manage_mtp_c8_src_test import (  # noqa: E402
    build_src_case, make_outputs,
)
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402  注册 torch.ops._C_ascend

torch.npu.set_device(0)

INT8_MAX = 127
DEQ_SCALE = 2.0 ** -10          # 商用 fixpipe DEQF16 指数
TH_BC, TH_BA = 99.5, 98.5       # B/C、B/A 集合重合率阈值 (%)


# ============================================================= 量化
def dyn_quant(tensor, dims):
    """逐 dims 粒度 int8 动态量化: scale=amax/127 (fp16), xq=round(x/scale) clamp ±127。"""
    amax = tensor.float().abs().amax(dim=dims, keepdim=True).clamp_min(1e-30)
    scale = (amax / INT8_MAX).half()                       # 存储精度即核内消费精度
    q = torch.clamp(torch.round(tensor.float() / scale.float()), -INT8_MAX, INT8_MAX)
    return q.to(torch.int8), scale


def quantize_case(case):
    """bf16 case -> (qi[.,.,128]i8, qs[T,H]f16, ki[blk,128,1,128]i8, ks[blk,128,1]f16,
    q_rec/k_rec 反量化重建 bf16)。"""
    qi, qs = dyn_quant(case["query"], dims=2)              # [T,H,128] 按 (t,h)
    ki, ks = dyn_quant(case["key"], dims=3)                # [blk,128,1,128] 按 token
    q_rec = (qi.float() * qs.float()).to(torch.bfloat16)
    k_rec = (ki.float() * ks.float()).to(torch.bfloat16)
    return qi, qs.squeeze(2), ki, ks.squeeze(3), q_rec, k_rec


# ============================================================= golden
def _gather_tokens(k_flat, bt_row, length):
    """按 block_table 展开 [L, D] (k_flat: [blk, 128, D])。"""
    blocks = (length + BLOCK - 1) // BLOCK
    idx = []
    for b in range(blocks):
        for r in range(min(BLOCK, length - b * BLOCK)):
            idx.append((int(bt_row[b]), r))
    return torch.stack([k_flat[p, r] for p, r in idx])


def golden_quant_row(qi, qs, w_bf16, ki, ks, bt_row, length):
    """B 路径核内数学的 fp32 闭式仿真(带 fp16 舍入点)。返回 [length] fp32。"""
    q32 = qi.float()                                        # [H,128]
    w32 = w_bf16.float()                                    # [H]
    wprime = (w32 * qs.float()).half().float()              # rint(w*qs) -> fp16
    tok = _gather_tokens(ki.float().squeeze(2), bt_row, length)      # [L,128]
    kscale = _gather_tokens(ks.float().squeeze(2), bt_row, length)  # [L]
    # int8 点积(整数, fp32 精确) -> relu -> *2^-10 -> fp16
    dot = torch.einsum("hd,id->hi", q32, tok)
    per_head = (torch.relu(dot) * DEQ_SCALE).half().float()  # [H,L]
    return (wprime @ per_head) * kscale


def golden_on(case, query_bf16, key_bf16):
    """A/C 共用: fp32 golden topk 于给定 (query, key) bf16 张量。"""
    rt = case["route_table"]
    vis = []
    for b, q in enumerate(case["q_values"]):
        st = case["states"][b]
        for t in range(q):
            vis.append(case["actual_key"][b] - (q - 1 - t) if st == -3
                       else case["offload_key"][b])
    q32, w32, kc = query_bf16.float(), case["weights"].float(), key_bf16
    rows, tie_free = [], []
    for t in range(q32.shape[0]):
        k_tok = expand_keys(kc, rt[t], vis[t])
        score = golden_score(q32[t:t + 1], w32[t:t + 1], k_tok, [vis[t]])
        src, tf = golden_topk(score, [vis[t]])[0]
        row = torch.full((TOPK,), -1, dtype=torch.int64)
        row[:len(src)] = torch.tensor(src, dtype=torch.int64)
        rows.append(row)
        tie_free.append(tf)
    return torch.stack(rows), tie_free, vis


# ============================================================= 调用
def to_dev_quant(case, outs):
    qi, qs, ki, ks, _, _ = quantize_case(case)
    di = {
        "query": qi.to(DEV), "q_scale": qs.to(DEV),
        "key": ki.to(DEV), "k_scale": ks.to(DEV),
        "weights": case["weights"].to(DEV),
        "block_table": case["block_table"].to(DEV),
        "pool": case["pool"].to(DEV),
    }
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


def run_op(case):
    di, do = to_dev_quant(case, make_outputs(case))
    for v in do.values():
        v.fill_(SENT)
    call_c8(di, do)
    return {k: v.cpu() for k, v in do.items()}


# ============================================================= 指标
def overlap(row_a, row_b):
    a = set(x for x in row_a.tolist() if x >= 0)
    b = set(x for x in row_b.tolist() if x >= 0)
    if not a or not b:
        return 100.0
    return 100.0 * len(a & b) / max(len(a), len(b))


def check_case(label, case, report):
    # 幂等: 同输入二次调用逐位一致
    res, res2 = run_op(case), run_op(case)
    for k in ("topk_src", "topk_dst", "topk_miss", "miss_cnt"):
        if not torch.equal(res[k], res2[k]):
            report(f"[{label}] 幂等失败: {k} 二次调用不一致")
            return

    qi, qs, ki, ks, q_rec, k_rec = quantize_case(case)
    ref_a, tie_a, vis = golden_on(case, case["query"], case["key"])
    ref_c, _, _ = golden_on(case, q_rec, k_rec)

    # 行号 -> (batch, t_in_batch) 映射, golden_quant 需要 (t,h) 粒度的 qi/qs
    t_idx = 0
    sum_bc, sum_ba, n = 0.0, 0.0, 0
    ordered_fail = 0
    for b, q in enumerate(case["q_values"]):
        for t in range(q):
            k_row = res["topk_src"][t_idx, 0]
            ov_c = overlap(k_row, ref_c[t_idx])
            ov_a = overlap(k_row, ref_a[t_idx])
            sum_bc += ov_c
            sum_ba += ov_a
            n += 1

            # 断言3: 量化闭式 golden, tie-free 时全序一致
            score = golden_quant_row(qi[t_idx].float(), qs[t_idx].float(),
                                     case["weights"][t_idx], ki, ks,
                                     case["route_table"][t_idx], vis[t_idx])
            src_q, tf_q = golden_topk(score.unsqueeze(0), [vis[t_idx]])[0]
            if tf_q:
                expect = torch.full((TOPK,), -1, dtype=torch.int64)
                expect[:len(src_q)] = torch.tensor(src_q, dtype=torch.int64)
                if not torch.equal(k_row[:len(src_q)].to(torch.int64), expect[:len(src_q)]):
                    d = next(i for i in range(len(src_q))
                             if int(k_row[i]) != int(expect[i]))
                    ordered_fail += 1
                    if ordered_fail <= 2:
                        report(f"[{label}] route{t_idx} 闭式golden顺序不一致 "
                               f"@pos{d}: kernel={int(k_row[d])} golden={int(expect[d])}")
            t_idx += 1

    mean_bc, mean_ba = sum_bc / max(n, 1), sum_ba / max(n, 1)
    status = "PASS" if mean_bc >= TH_BC and mean_ba >= TH_BA else "FAIL"
    print(f"  [{label}] B/C={mean_bc:.3f}%  B/A={mean_ba:.3f}%  ({status})")
    if mean_bc < TH_BC:
        report(f"[{label}] B/C 重合率 {mean_bc:.3f}% < {TH_BC}%")
    if mean_ba < TH_BA:
        report(f"[{label}] B/A 重合率 {mean_ba:.3f}% < {TH_BA}%")


# ============================================================= 场景
def scenarios():
    yield "mixed-states", dict(q_values=[1, 3, 7], states=[-3, -2, -1],
                               offload_len=8192)
    yield "two-batch", dict(q_values=[2, 2], states=[-1, -3], offload_len=8192)
    yield "first-decode", dict(q_values=[1, 4], states=[-2, -2],
                               offload_len=12288, cache_tokens=8192)
    yield "steady-L>C", dict(q_values=[4], states=[-1], offload_len=8320,
                             cache_tokens=8192)
    yield "long-2p18", dict(q_values=[1, 7], states=[-1, -3],
                            offload_len=2**18 + 2048,
                            source_capacity=2**18 + 2048 + 128)
    yield "h64-spot", dict(q_values=[3, 4], states=[-3, -1], offload_len=8192,
                           heads=64)


# ============================================================= 守卫回归
def run_guards(report):
    case = build_src_case([1], [-3], offload_len=4096)
    qi, qs, ki, ks, _, _ = quantize_case(case)
    outs = {k: v.to(DEV) for k, v in make_outputs(case).items()}
    meta = {k: v.to(DEV) for k, v in case["metadata"].items()}
    pool = case["pool"].to(DEV)
    bt = case["block_table"].to(DEV)
    w = case["weights"].to(DEV)

    def expect_reject(label, q_, qs_, k_, ks_, w_):
        try:
            torch.ops._C_ascend.npu_fused_li_manage_mtp_c8(
                w_, qs_, q_, ks_, k_, bt, meta["aslq"], meta["ask"],
                meta["offl"], meta["nct"], meta["rstate"], meta["rpe"], pool,
                outs["topk_src"].fill_(SENT), outs["topk_dst"].fill_(SENT),
                outs["topk_miss"].fill_(SENT), outs["miss_src"].fill_(SENT),
                outs["miss_dst"].fill_(SENT), outs["miss_cnt"].fill_(SENT))
            torch.npu.synchronize()
            report(f"[guard-{label}] 未按预期拒绝")
        except RuntimeError:
            print(f"  [guard-{label}] RuntimeError OK")

    expect_reject("fp32-scale", qi.to(DEV), qs.float().to(DEV), ki.to(DEV),
                  ks.float().to(DEV), w)
    expect_reject("bf16-query", case["query"].to(DEV), qs.to(DEV), ki.to(DEV),
                  ks.to(DEV), w)
    expect_reject("fp16-weights", qi.to(DEV), qs.to(DEV), ki.to(DEV),
                  ks.to(DEV), case["weights"].to(torch.float16).to(DEV))


# ============================================================= main
def main():
    errs = []

    def report(msg):
        errs.append(msg)
        print(f"  FAIL {msg}")

    print("== 场景精度 (A/B/C) ==")
    for name, kw in scenarios():
        case = build_src_case(seed=11, **kw)
        check_case(name, case, report)

    print("== 守卫回归 ==")
    run_guards(report)

    print("=" * 60)
    if errs:
        print(f"结果: FAIL ({len(errs)} 项)")
        for e in errs:
            print(" -", e)
        sys.exit(1)
    print("结果: PASS (全部口径达标)")


if __name__ == "__main__":
    main()
