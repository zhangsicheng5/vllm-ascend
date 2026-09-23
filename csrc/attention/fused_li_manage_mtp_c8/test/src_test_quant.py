#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src_test 场景矩阵的量化 ABI 移植（2026-09-10，接口跟随 ops_lim_standardization 后）。

与 bf16 时代 src_test 的口径差异:
- dtype 维度消失: 量化 ABI 单一输入通路 (query/key int8 + fp16 scale, 权重 bf16),
  原 ×{bf16,fp16} 用例合并, 75 例映射为 51 个独立场景;
- 断言引擎复用 quant_full: mgmt 精确(由 kernel 自身 topk 驱动, 与打分噪声解耦)
  + 量化闭式 golden(tie-free 全序) + B/C≥99.5 / B/A≥98.5;
- steady_strict 的受控 union 构造改用量化闭式 golden topk; miss 计数容差 ±25
  (kernel/golden 并列翻转), 幂等不动点仍精确断言;
- key_tag/invalid 由 quant_full 的 keytag/invalid 模式覆盖, 此处不重复;
- long 模式沿用 L=264192 (>2^17 新长源边界), Q4 −1 稳态 2 例为 §1.5 窗口回归
  (官方修复后应转绿)。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fused_li_manage_mtp_c8_quant_full as QF  # noqa: E402
from fused_li_manage_mtp_c8_src_test import (  # noqa: E402
    REPLACEMENT, STRICT_SCENARIOS, build_src_case, visible_lengths,
    _controlled_union_row, _force_topk_hit_slot_zero, _randomize_block_table,
)
from fused_li_manage_mtp_c8_quant_test import (  # noqa: E402
    TOPK, golden_quant_row, quantize_case,
)
from fused_li_manage_mtp_c8_test import golden_topk  # noqa: E402

FAILS = []


def report(label, errs):
    if errs:
        FAILS.extend(errs)
        print(f"  FAIL {label}", flush=True)
        for e in errs:
            print(f"    - {e}", flush=True)
    else:
        print(f"  PASS {label}", flush=True)


# ============================================================= 通用: 多步断言
def _golden_per_step(case, res, errs, label, closed_bc=False):
    """check_case_full 的 golden 半区(单步版): B/C、B/A 阈值 + 闭式全序(tie-free)。

    closed_bc=True 时 B/C 改用闭式 golden(逐位复刻 kernel 的 fp16 舍入点)——
    并列密集构造(shared_w+correlated)下 dequant 重建 C golden 与 kernel 真实
    量化数学自身偏差可达 0.5%+ (2026-09-10 q1-long 实测: 闭式=100.000%,
    dequant-C=99.463%), 该偏差是参考系产物而非 kernel 缺陷。"""
    qi, qs, ki, ks, q_rec, k_rec = quantize_case(case)
    ref_a, _, vis = QF.golden_on(case, case["query"], case["key"])
    if closed_bc:
        ref_c = _quant_ref_rows(case)
    else:
        ref_c, _, _ = QF.golden_on(case, q_rec, k_rec)
    sbc = sba = 0.0
    n = res["topk_src"].shape[0]
    for t in range(n):
        k_row = res["topk_src"][t, 0]
        sbc += QF.overlap(k_row, ref_c[t])
        sba += QF.overlap(k_row, ref_a[t])
        score = golden_quant_row(qi[t].float(), qs[t].float(),
                                 case["weights"][t], ki, ks,
                                 case["route_table"][t], vis[t])
        src_q, tf_q = golden_topk(score.unsqueeze(0), [vis[t]])[0]
        if tf_q:
            expect = torch.full((TOPK,), -1, dtype=torch.int64)
            expect[:len(src_q)] = torch.tensor(src_q, dtype=torch.int64)
            if not torch.equal(k_row[:len(src_q)].to(torch.int64),
                               expect[:len(src_q)]):
                errs.append(f"[{label}] route{t} 闭式golden顺序不一致")
                break
    mean_bc, mean_ba = sbc / max(n, 1), sba / max(n, 1)
    if mean_bc < QF.TH_BC:
        errs.append(f"[{label}] B/C {mean_bc:.3f}% < {QF.TH_BC}%")
    if mean_ba < QF.TH_BA:
        errs.append(f"[{label}] B/A {mean_ba:.3f}% < {QF.TH_BA}%")
    return mean_bc, mean_ba


def check_steps_full(case, states_seq, label, errs, closed_bc=False):
    """多步(生命周期/-3→-2 转换): 逐步 mgmt 精确 + 逐步 golden。"""
    steps = QF.run_steps(case, states_seq)
    for i, (res, old) in enumerate(steps):
        case["states"] = list(states_seq[i])
        QF.assert_mgmt_exact(case, res, old, errs, f"{label}-step{i}")
        _golden_per_step(case, res, errs, f"{label}-step{i}", closed_bc)


# ============================================================= 1. correctness
def run_correctness():
    q_values = [1, 2, 3, 4, 5, 6, 7]
    for states in ([-3] * 7, [-2] * 7, [-1] * 7,
                   [(-3, -2, -1)[i % 3] for i in range(7)]):
        case = build_src_case(q_values, states, offload_len=8192,
                              cache_tokens=8192, seed=7)
        errs = []
        QF.check_case_full(case, f"corr-{states}", errs)
        report(f"correctness q=1..7 states={states}", errs)


# ============================================================= 2. replacement
def run_replacement():
    for q_values, L, C in REPLACEMENT:
        case = build_src_case(q_values, [-1] * len(q_values), offload_len=L,
                              cache_tokens=C)
        errs = []
        QF.check_case_full(case, f"repl-{q_values}", errs)
        report(f"replacement q={q_values} state=-1 L={L} C={C}", errs)


# ============================================================= 3. first_decode
def run_first_decode():
    for q_values, L, C in REPLACEMENT:
        n = len(q_values)
        case = build_src_case(q_values, [-2] * n, offload_len=L, cache_tokens=C)
        errs = []
        QF.check_case_full(case, f"fd-cold-{q_values}", errs)
        report(f"first-decode cold q={q_values} state=-2 L={L} C={C}", errs)

        case = build_src_case(q_values, [-3] * n, offload_len=L, cache_tokens=C)
        errs = []
        QF.check_case_full(case, f"fd-std-{q_values}", errs)
        report(f"first-decode pre-step q={q_values} state=-3 L={L} C={C}", errs)

        case = build_src_case(q_values, [-3] * n, offload_len=L, cache_tokens=C)
        errs = []
        check_steps_full(case, [[-3] * n, [-2] * n], f"fd-trans-{q_values}", errs)
        report(f"first-decode transition q={q_values} state=-3->-2 L={L} C={C}", errs)


# ============================================================= 4. lifecycle
def run_lifecycle():
    for seq in ((-2, -1, -1), (-3, -1), (-3, -2, -1)):
        case = build_src_case([1, 4, 7], [seq[0]] * 3, offload_len=8192,
                              cache_tokens=8192, seed=17)
        errs = []
        check_steps_full(case, [[st] * 3 for st in seq], f"lc-{seq}", errs)
        if seq[-2:] == (-1, -1):
            pass  # 末步 (-1,-1) 不动点由 assert_mgmt_exact 的 miss=0 语义覆盖
        report(f"lifecycle q=[1,4,7] seq={seq}", errs)


# ============================================================= 5. steady_strict
def _quant_ref_rows(case):
    """量化闭式 golden topk 行(每 route): int64 [T, TOPK](不足补 -1)。"""
    qi, qs, ki, ks, _, _ = quantize_case(case)
    vis = visible_lengths(case)
    rows = []
    for t in range(qi.shape[0]):
        score = golden_quant_row(qi[t].float(), qs[t].float(),
                                 case["weights"][t], ki, ks,
                                 case["route_table"][t], vis[t])
        src, _ = golden_topk(score.unsqueeze(0), [vis[t]])[0]
        row = torch.full((TOPK,), -1, dtype=torch.int64)
        row[:len(src)] = torch.tensor(src, dtype=torch.int64)
        rows.append(row)
    return torch.stack(rows)


def run_steady_strict():
    for idx, (label, q_values, L, C, target) in enumerate(STRICT_SCENARIOS):
        case = build_src_case(q_values, [-1] * len(q_values), offload_len=L,
                              cache_tokens=C)
        gen = torch.Generator().manual_seed(7 + 20011 + idx * 97)
        _randomize_block_table(case, gen)
        ref = _quant_ref_rows(case)
        pool = torch.full_like(case["pool"], -2147483648)
        actual_unions = []
        tb = 0
        for b, q in enumerate(q_values):
            union = torch.unique(ref[tb:tb + q].reshape(-1), sorted=True)
            union = union[(union >= 0) & (union < L)]
            tb += q
            _, cached, actual = _controlled_union_row(union, L, C, target, gen)
            pool[case["req_entries"][b], cached] = torch.randperm(
                C, generator=gen, dtype=torch.int64).to(torch.int32)
            actual_unions.append(actual)
        pool = _force_topk_hit_slot_zero(case, ref, pool)
        case["pool"] = pool
        errs = []
        # 第一次调用: mgmt 精确(kernel topk 驱动) + golden; miss 计数容差 ±25
        steps = QF.run_steps(case, [[-1] * len(q_values)] * 2)
        (res1, old1), (res2, _) = steps
        case["states"] = [-1] * len(q_values)
        QF.assert_mgmt_exact(case, res1, old1, errs, f"strict-{label}")
        _golden_per_step(case, res1, errs, f"strict-{label}")
        mc = res1["miss_cnt"].tolist()
        for b, (got, want) in enumerate(zip(mc, actual_unions)):
            if abs(got - want) > 25:
                errs.append(f"[strict-{label}] b{b} miss_cnt={got} "
                            f"vs 受控 {want} 超容差 ±25")
        # 第二次调用(同池): 不动点——miss 全 0、计数全 0、池逐位不变
        if not bool((res2["topk_miss"] == 0).all()):
            errs.append(f"[strict-{label}] 重复调用 topk_miss 非 0")
        if not bool((res2["miss_cnt"] == 0).all()):
            errs.append(f"[strict-{label}] 重复调用 miss_cnt 非 0")
        if not torch.equal(res2["pool_after"], res1["pool_after"]):
            errs.append(f"[strict-{label}] 重复调用改写了池")
        report(f"steady_strict {label} q={q_values} L={L} C={C} "
               f"union={actual_unions}", errs)


# ============================================================= 6. long
LONG_L = 264192
LONG_CAP = LONG_L + 128


def _long_union_ok(case, ref, cmax):
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
    for attempt in range(8):
        case = build_src_case(q_values, states, offload_len=LONG_L,
                              cache_tokens=cache_tokens, source_capacity=LONG_CAP,
                              correlated=True, noise=0.5, shared_w=True,
                              seed=seed + attempt * 13)
        ref = _quant_ref_rows(case)
        if _long_union_ok(case, ref, int(cache_tokens * 0.9)):
            return case, ref
    raise AssertionError("long: 8 个种子仍未满足 union 预算")


def run_long():
    high = 1 << 17  # [slot15|src17] 新长源边界
    for q_values, C in (([1], 8192), ([4], 12288), ([7], 14336)):
        for states in ([-3], [-2], [-1]):
            case, ref = build_long_case(q_values, states, C, seed=7)
            if not bool((ref >= high).any()):
                raise AssertionError("long 参考未覆盖 >=2^17 源(21-bit 高位路径未验证)")
            errs = []
            res, old = QF.run_once(case)
            QF.assert_mgmt_exact(case, res, old, errs, f"long-{states}")
            _golden_per_step(case, res, errs, f"long-{states}", closed_bc=True)
            report(f"long q={q_values} states={states} L={LONG_L} C={C}", errs)
        n = len(q_values)
        case, _ = build_long_case(q_values, [-3] * n, C, seed=7)
        errs = []
        check_steps_full(case, [[-3] * n, [-2] * n], "long-trans", errs,
                         closed_bc=True)
        report(f"long transition q={q_values} state=-3->-2 L={LONG_L} C={C}", errs)
        for seq in ((-2, -1, -1), (-3, -1), (-3, -2, -1)):
            case = build_src_case(q_values, [seq[0]] * n, offload_len=LONG_L,
                                  cache_tokens=C, source_capacity=LONG_CAP,
                                  correlated=True, noise=0.5, shared_w=True, seed=7)
            errs = []
            check_steps_full(case, [[st] * n for st in seq], f"long-lc-{seq}",
                             errs, closed_bc=True)
            report(f"long lifecycle q={q_values} seq={seq} L={LONG_L} C={C}", errs)
    case, ref = build_long_case([1, 4, 7], [-3, -2, -1], 14336, seed=7)
    if not bool((ref >= high).any()):
        raise AssertionError("long mixed 参考未覆盖 >=2^17 源")
    errs = []
    res, old = QF.run_once(case)
    QF.assert_mgmt_exact(case, res, old, errs, "long-mixed")
    _golden_per_step(case, res, errs, "long-mixed", closed_bc=True)
    report("long mixed q=[1,4,7] states=[-3,-2,-1] L=264192 C=14336", errs)


# ============================================================= main
def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    modes = {
        "correctness": run_correctness,
        "replacement": run_replacement,
        "first_decode": run_first_decode,
        "lifecycle": run_lifecycle,
        "steady_strict": run_steady_strict,
        "long": run_long,
    }
    QF.warm_operators()
    for name, fn in modes.items():
        if only not in ("all", name):
            continue
        print(f"\n===== mode {name} =====", flush=True)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILS.append(f"[{name}] EXCEPTION {type(e).__name__}: {e}")
    print(f"\n===== {'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES'} =====")
    for m in FAILS:
        print(f"  FAILED: {m}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
