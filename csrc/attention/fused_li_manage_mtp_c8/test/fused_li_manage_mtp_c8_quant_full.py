#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fused_li_manage_mtp_c8（真实 C8 量化）补充验证: 管理段精确断言 + 生命周期 +
边界/q-sweep + key_tag + §1.5 窗口复测 + 六算子 A/B/C/D/E/F 精度矩阵。

在 quant_test（§7.3 A/B/C 口径）之上补齐旧版（bf16 占位期）已有的验证维度:

  mgmt       管理段逐元素精确断言（miss_cnt/miss_src 序/miss_dst 落点互异/池双射/
             -3 恒等复写/-2 top-C 选择/未参与行零改写）——全部由 kernel 自己的
             topk_src 驱动（与打分噪声解耦），topk 行另按量化闭式 golden 全序断言
  lifecycle  状态链 (-2,-1,-1)/(-3,-1)/(-3,-2,-1) × q=[1,4,7]，逐步 mgmt 精确
  warmfix    热池不动点: 第一次调用后同池二次调用 miss=0 且池零改写
  boundary   乱序 rpe + 随机 block_table + 未参与行零改写
  m3tail     -3 短可见尾部 -1 契约（vis<2048 时 topk_src/dst 尾部全 -1,
             前缀集合恰为 [0,vis); 回归 mtp commit 11a98e560 同源修复）
  c0         C=0 双态 safe-failure
  invalid    12 例元数据违例 safe-failure + fp32 scale host 拒绝 + 结构违例探针
             （量化移植, 替代已被清理的 /tmp 诊断脚本）
  qsweep     q=1..7 单批 -1 稳态全量断言 + q=8 隔离子进程探针（MTP 规格外）
  keytag     21-bit 高位源 ID: -3 解析式 golden（band 恒值并列→ID 升序截断）+
             -1 分数标签打包/解码（TagLongIndex 高 3 位重建全 ID）
  window     §1.5 源继承缺陷窗口 Σ(每路miss)∈(6144,7936] 量化 ABI 复测 +
             与源算子（D）同张量缺陷形态对拍
  matrix     六算子共享场景精度矩阵: A=golden(bf16) B=本算子(int8) C=golden(dequant)
             D=源 fused_li_manage_mtp(bf16) E=官方 LI(bf16) F=li_quant(int8)

env: D 需要 NANOVLLM_CUST_OPAPI_LIB 双算子隔离加载（无则自动 SKIP D）。
运行: python3 fused_li_manage_mtp_c8_quant_full.py [--only mgmt,lifecycle,...]
"""
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_li_manage_mtp_c8_test import (  # noqa: E402
    BLOCK, DEV, MISS_CAP, SENT, TOPK, golden_topk,
)
from fused_li_manage_mtp_c8_src_test import (  # noqa: E402
    INVALID_BASE, INVALID_CASES, INVALID_SLOT, build_src_case, make_outputs,
)
from fused_li_manage_mtp_c8_quant_test import (  # noqa: E402
    TH_BA, TH_BC, golden_on, golden_quant_row, overlap, quantize_case,
    to_dev_quant,
)
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402  注册 torch.ops._C_ascend

try:
    # D 的 OPP vendors 路径在 import 时前插进 ASCEND_CUSTOM_OPP_PATH; 必须发生在
    # torch.npu.set_device / 首次 aclnn 调用之前——CANN 运行时对 OPP 搜索路径存在
    # 初始化期快照, 惰性 import（晚于 c8 调用）会使 D 解析为 EZ1009"包不可见"
    import nanovllm  # noqa: F401
    HAVE_NANOVLLM = True
except Exception:  # noqa: BLE001
    HAVE_NANOVLLM = False

torch.npu.set_device(0)

PACKED = 1 << 17            # 21-bit source ID 的高 4 位边界 ([slot15|src17])
LONG_L, LONG_CAP = 264192, 264320
REPLACEMENT = (([1], 8320, 8192), ([1, 2, 3], 8320, 8192),
               ([4], 16256, 12288), ([7], 16256, 14336))


# ============================================================= 运行器
def call_c8(di, do):
    torch.ops._C_ascend.npu_fused_li_manage_mtp_c8(
        di["weights"], di["q_scale"], di["query"], di["k_scale"],
        di["key"], di["block_table"], di["aslq"], di["ask"],
        di["offl"], di["nct"], di["rstate"], di["rpe"], di["pool"],
        do["topk_src"], do["topk_dst"], do["topk_miss"],
        do["miss_src"], do["miss_dst"], do["miss_cnt"])
    torch.npu.synchronize()


def run_once(case):
    """量化 ABI 单次调用。返回 (res, old_pool_cpu)。"""
    di, do = to_dev_quant(case, make_outputs(case))
    old = case["pool"].clone()
    call_c8(di, do)
    res = {k: v.cpu() for k, v in do.items()}
    res["pool_after"] = di["pool"].cpu()
    return res, old


def run_steps(case, states_seq):
    """同一份 device 张量（pool 原地演进）连续调用，逐步快照。"""
    di, do = to_dev_quant(case, make_outputs(case))
    steps = []
    for st in states_seq:
        di["rstate"].copy_(torch.tensor(st, dtype=torch.int32, device=DEV))
        old = di["pool"].cpu().clone()
        for v in do.values():
            v.fill_(SENT)
        call_c8(di, do)
        res = {k: v.cpu() for k, v in do.items()}
        res["pool_after"] = di["pool"].cpu()
        steps.append((res, old))
    return steps


def src_available():
    if not HAVE_NANOVLLM:
        return False
    try:
        torch.ops.nanovllm_dsa.fused_li_manage_mtp  # noqa: B018
        return True
    except Exception:  # noqa: BLE001
        return False


D_WARNED = False


def run_src_safe(case):
    """run_src 容错版: 环境类故障(EZ1009 等)打印告警并返回 None——D 是参照算子,
    不可用降级为跳过 D 列而非判整套 FAIL。"""
    global D_WARNED
    try:
        return run_src(case)
    except RuntimeError as e:
        if not D_WARNED:
            print(f"  [D] 源算子调用失败(环境类, 本进程后续跳过 D): {str(e)[:150]}")
            D_WARNED = True
        return None


def run_src(case):
    """D = 源 fused_li_manage_mtp（bf16 + 全零 fp32 scale，scale 已证不变）。"""
    di = {k: case[k].to(DEV) for k in ("query", "weights", "key", "block_table", "pool")}
    di.update({k: v.to(DEV) for k, v in case["metadata"].items()})
    di["q_scale"] = torch.zeros(case["query"].shape[0], case["heads"],
                                dtype=torch.float32, device=DEV)
    di["k_scale"] = torch.zeros(case["key"].shape[0], BLOCK, 1,
                                dtype=torch.float32, device=DEV)
    do = {k: v.to(DEV) for k, v in make_outputs(case).items()}
    torch.ops.nanovllm_dsa.fused_li_manage_mtp(
        di["weights"], di["q_scale"], di["query"], di["k_scale"],
        di["key"], di["block_table"], di["aslq"], di["ask"],
        di["offl"], di["nct"], di["rstate"], di["rpe"], di["pool"],
        do["topk_src"], do["topk_dst"], do["topk_miss"],
        do["miss_src"], do["miss_dst"], do["miss_cnt"])
    torch.npu.synchronize()
    res = {k: v.cpu() for k, v in do.items()}
    res["pool_after"] = di["pool"].cpu()
    return res


# ============================================================= 管理段精确断言
def assert_mgmt_exact(case, res, old_pool, errs, label, skip_dst=False):
    """src_test assert_correctness_c8 的量化移植。

    miss/union/pool 全部断言由 kernel 自己的 topk_src 驱动（与量化打分噪声解耦）;
    跨 kernel 的 golden 行对比不在此（由 check_route_quant 单独做）。
    skip_dst: §1.5 窗口模式下跳过 topk_dst 池映射断言（已知源继承缺陷）。
    """
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
            if valid < TOPK and not bool((src[route, 0, valid:] == -1).all()):
                errs.append(f"[{label}] route{route} 有效域外应填 -1")
            if st == -3:
                if int(route_miss[route]) != 0:
                    errs.append(f"[{label}] -3 route{route} topk_miss 应 0, "
                                f"got {int(route_miss[route])}")
                if not torch.equal(dst[route, 0], src[route, 0]):
                    errs.append(f"[{label}] -3 route{route} topk_dst 应恒等 src")
            elif not skip_dst:
                for pos in range(TOPK):
                    s = int(src[route, 0, pos])
                    if s >= 0 and int(dst[route, 0, pos]) != int(cache_cpu[row, s]):
                        errs.append(f"[{label}] route{route} pos{pos} dst != 池终态映射")
                        break
        if st == -3:
            if int(miss_cnt[b]) != 0:
                errs.append(f"[{label}] -3 b{b} miss_cnt 应 0, got {int(miss_cnt[b])}")
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
                                f"got[:5]={miss_src[b, :5].tolist()}")
                if not bool((route_miss[tb:tb + q] == TOPK).all()):
                    errs.append(f"[{label}] -2 b{b} topk_miss 应全 {TOPK}")
        else:
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
    for r in range(case["pool_rows"]):
        if r not in case["req_entries"]:
            if not torch.equal(res["pool_after"][r], old_pool[r]):
                errs.append(f"[{label}] 未参与行 {r} 被改写")
                break


def check_case_full(case, label, errs):
    """完整口径: mgmt 精确 + 量化闭式全序(tie-free) + B/C、B/A 阈值。"""
    res, old = run_once(case)
    assert_mgmt_exact(case, res, old, errs, label)
    qi, qs, ki, ks, q_rec, k_rec = quantize_case(case)
    ref_a, _, vis = golden_on(case, case["query"], case["key"])
    ref_c, _, _ = golden_on(case, q_rec, k_rec)
    sbc = sba = 0.0
    ordered_fail = 0
    n = res["topk_src"].shape[0]
    for t in range(n):
        k_row = res["topk_src"][t, 0]
        sbc += overlap(k_row, ref_c[t])
        sba += overlap(k_row, ref_a[t])
        score = golden_quant_row(qi[t].float(), qs[t].float(),
                                 case["weights"][t], ki, ks,
                                 case["route_table"][t], vis[t])
        src_q, tf_q = golden_topk(score.unsqueeze(0), [vis[t]])[0]
        if tf_q:
            expect = torch.full((TOPK,), -1, dtype=torch.int64)
            expect[:len(src_q)] = torch.tensor(src_q, dtype=torch.int64)
            if not torch.equal(k_row[:len(src_q)].to(torch.int64), expect[:len(src_q)]):
                ordered_fail += 1
                if ordered_fail <= 2:
                    d = next(i for i in range(len(src_q))
                             if int(k_row[i]) != int(expect[i]))
                    errs.append(f"[{label}] route{t} 闭式golden顺序不一致 "
                                f"@pos{d}: kernel={int(k_row[d])} golden={int(expect[d])}")
    mean_bc, mean_ba = sbc / max(n, 1), sba / max(n, 1)
    print(f"  [{label}] B/C={mean_bc:.3f}%  B/A={mean_ba:.3f}%"
          + (f"  闭式序不一致行={ordered_fail}" if ordered_fail else ""))
    if mean_bc < TH_BC:
        errs.append(f"[{label}] B/C 重合率 {mean_bc:.3f}% < {TH_BC}%")
    if mean_ba < TH_BA:
        errs.append(f"[{label}] B/A 重合率 {mean_ba:.3f}% < {TH_BA}%")


# ============================================================= mode: mgmt
def run_mgmt(errs):
    scen = [
        ("mixed-states", dict(q_values=[1, 3, 7], states=[-3, -2, -1],
                              offload_len=8192)),
        ("two-batch", dict(q_values=[2, 2], states=[-1, -3], offload_len=8192)),
        ("first-decode", dict(q_values=[1, 4], states=[-2, -2],
                              offload_len=12288, cache_tokens=8192)),
        ("steady-L>C", dict(q_values=[4], states=[-1], offload_len=8320,
                            cache_tokens=8192)),
        ("long-2p18", dict(q_values=[1, 7], states=[-1, -3],
                           offload_len=2**18 + 2048,
                           source_capacity=2**18 + 2048 + 128)),
        ("h64-spot", dict(q_values=[3, 4], states=[-3, -1], offload_len=8192,
                          heads=64)),
        # 2026-09-10 接口跟随: 同 batch 混合 q（含 q>7, 第二 gS1 块跨 batch 组合）
        ("mixed-q-high", dict(q_values=[4, 9, 14], states=[-1, -1, -1],
                              offload_len=30720, cache_tokens=28672,
                              source_capacity=30720 + 128)),
        ("mixed-q-states", dict(q_values=[8, 4, 12], states=[-3, -2, -1],
                                offload_len=25600, cache_tokens=24576,
                                source_capacity=25600 + 128)),
        ("mixed-q-edge", dict(q_values=[1, 14], states=[-1, -3],
                              offload_len=30720, cache_tokens=28672,
                              source_capacity=30720 + 128)),
    ]
    scen += [(f"repl-q{''.join(map(str, q))}", dict(q_values=q, states=[-1] * len(q),
                                                    offload_len=L, cache_tokens=C))
             for q, L, C in REPLACEMENT]
    for name, kw in scen:
        check_case_full(build_src_case(seed=11, **kw), f"mgmt-{name}", errs)


# ============================================================= mode: lifecycle
def run_lifecycle(errs):
    """q_values=[1,4,7]（B=3）锁步演进（对齐 src_test lifecycle），逐步 mgmt 精确;
    序列末尾为 (-1,-1) 时末步自身即热池不动点（上一步 -1 已装齐），断言 miss 全 0。"""
    for seq in ((-2, -1, -1), (-3, -1), (-3, -2, -1)):
        case = build_src_case([1, 4, 7], [seq[0]] * 3, offload_len=8192, seed=17)
        steps = run_steps(case, [[st] * 3 for st in seq])
        for i, (res, old) in enumerate(steps):
            case["states"] = [seq[i]] * 3
            assert_mgmt_exact(case, res, old, errs,
                              f"lc-{''.join(map(str, seq))}-step{i}")
        if seq[-2:] == (-1, -1):
            last = steps[-1][0]
            if not bool((last["topk_miss"] == 0).all()):
                errs.append(f"[lc-{seq}] 末态 (-1,-1) topk_miss 应全 0, "
                            f"got {last['topk_miss'].tolist()}")
            if not bool((last["miss_cnt"] == 0).all()):
                errs.append(f"[lc-{seq}] 末态 (-1,-1) miss_cnt 应全 0, "
                            f"got {last['miss_cnt'].tolist()}")


# ============================================================= mode: warmfix
def run_warmfix(errs):
    q, L, C = REPLACEMENT[2]
    case = build_src_case(q, [-1] * len(q), offload_len=L, cache_tokens=C, seed=23)
    (res1, _), (res2, old2) = run_steps(case, [[-1] * len(q)] * 2)
    m1 = int(res1["miss_cnt"][0])
    if m1 == 0:
        errs.append("[warmfix] 构造失败: 第一步即 miss=0, 换 seed")
        return
    if int(res2["miss_cnt"][0]) != 0:
        errs.append(f"[warmfix] 同池二次调用 miss_cnt 应 0, got {int(res2['miss_cnt'][0])}")
    if int(res2["topk_miss"][:sum(q)].max()) != 0:
        errs.append(f"[warmfix] 同池二次调用 topk_miss 应全 0, "
                    f"got {res2['topk_miss'][:sum(q)].tolist()}")
    if not torch.equal(res2["pool_after"], res1["pool_after"]):
        diff = int((res2["pool_after"] != res1["pool_after"]).sum())
        errs.append(f"[warmfix] 同池二次调用池被改写 {diff} 处")
    print(f"  [warmfix] step1 miss={m1} -> step2 miss={int(res2['miss_cnt'][0])}, "
          f"pool 逐位不变")


# ============================================================= mode: boundary
def run_boundary(errs):
    B, q = 3, [4, 4, 4]
    rpe = [4, 1, 3]
    case = build_src_case(q, [-1] * B, offload_len=8192, seed=29, rpe=rpe,
                          pool_rows=7)
    gen = torch.Generator().manual_seed(29)
    bt = torch.stack([torch.randperm(case["source_capacity"] // BLOCK,
                                     generator=gen).to(torch.int32) for _ in range(B)])
    case["block_table"] = bt
    q2r = [r for r, qq in enumerate(q) for _ in range(qq)]
    case["route_table"] = bt[torch.tensor(q2r)].contiguous()
    check_case_full(case, "boundary-randbt-rpe413", errs)


# ============================================================= mode: m3tail
def run_m3tail(errs):
    """-3 短可见尾部 -1 契约（回归 mtp commit 11a98e560 同源修复, 2026-09-23）。

    因果掩码只把分数置 -inf, 共享 chunk 载荷里仍残留后续 route 的源 ID;
    visible < sparseCount(2048) 时这些位置必须被显式重写为 -1（对齐官方
    npu_lightning_indexer）, 否则 topk_src/topk_dst 尾部泄漏不存在的 token。

    断言（每 route）: 前缀集合恰为 [0, vis) 全体（vis<2048 时 top-k 必然全收,
    与量化噪声无关）; [vis, 2048) 尾部 topk_src/topk_dst 全 -1; 走
    assert_mgmt_exact 的 -3 精确断言（miss=0 / dst 恒等 / 池不动）。

    配置覆盖对齐与标量补写两条路径:
      q1-al    vis=1024（32B 对齐, 纯 Duplicate 批量填）
      q4-unal  vis=1149..1152（非对齐, 尾部 ≤7 项标量补写——原缺陷实测泄漏点）
      q7-cross vis=2042..2048（跨越 sparseCount, 末路 vis=2048 不触发填充）
      q4-ctrl  vis=4093..4096（vis>2048 对照组: 不触发, 行内无 -1）
    """
    for label, q, L in (("q1-al", [1], 896), ("q4-unal", [4], 1024),
                        ("q7-cross", [7], 1920), ("q4-ctrl", [4], 4096)):
        case = build_src_case(q, [-3] * len(q), offload_len=L, seed=37)
        res, old = run_once(case)
        assert_mgmt_exact(case, res, old, errs, f"m3tail-{label}")
        actk = case["actual_key"][0]
        n_route = q[0]                    # 单请求 B=1, 路数 = q_values[0]
        for t in range(n_route):
            vis = actk - n_route + t + 1
            src = res["topk_src"][t, 0]
            dst = res["topk_dst"][t, 0]
            if vis >= TOPK:
                # 对照组: 2048 位全部应为有效源 ID
                if bool((src == -1).any()):
                    errs.append(f"[m3tail-{label}] route{t} vis={vis}≥2048 不应有 -1")
                continue
            if sorted(src[:vis].tolist()) != list(range(vis)):
                errs.append(f"[m3tail-{label}] route{t} vis={vis} 前缀集合≠[0,{vis})")
            leak_src = src[vis:][src[vis:] != -1]
            leak_dst = dst[vis:][dst[vis:] != -1]
            if leak_src.numel() or leak_dst.numel():
                errs.append(f"[m3tail-{label}] route{t} vis={vis} 尾部泄漏 "
                            f"src={leak_src[:4].tolist()} dst={leak_dst[:4].tolist()}")
        print(f"  [m3tail-{label}] q={n_route} vis={actk - n_route + 1}..{actk} PASS",
              flush=True)


# ============================================================= mode: c0
def run_c0(errs):
    case = build_src_case([4, 4], [-1, -2], offload_len=12288, cache_tokens=0,
                          seed=31)          # L>q*2048 -> nct=0 生效
    assert case["cache_tokens"] == [0, 0], "C=0 构造未生效"
    res, old = run_once(case)
    for b, st in enumerate((-1, -2)):
        tb = 4 * b
        if not bool((res["topk_src"][tb:tb + 4] == -1).all()) or int(res["miss_cnt"][b]) != 0:
            errs.append(f"[c0] state={st} C=0 应 safe-failure(topk=-1, miss=0), "
                        f"got miss={int(res['miss_cnt'][b])} "
                        f"topk0={res['topk_src'][tb, 0, :3].tolist()}")
    if not torch.equal(res["pool_after"], old):
        errs.append("[c0] safe-failure 池应不动")


# ============================================================= mode: invalid
def run_invalid(errs):
    """src_test run_invalid 的量化移植: 12 例元数据违例 safe-failure + fp32 scale
    host 拒绝（quant ABI 要求 fp16, 与 src 版 fp16→fp32 方向相反）+ 结构违例子进程探针。"""
    case = build_src_case(**INVALID_BASE)
    passed = 0
    total = 0
    for field, value, expect in INVALID_CASES:
        if expect == "probe":
            continue
        total += 1
        di, do = to_dev_quant(case, make_outputs(case))
        di[field] = torch.tensor(value, dtype=torch.int32, device=DEV)
        pool_before = di["pool"].cpu().clone()
        for v in do.values():
            v.fill_(SENT)
        try:
            call_c8(di, do)
        except RuntimeError as e:
            errs.append(f"[invalid {field}={value}] 调用异常: {e}")
            continue
        src = do["topk_src"].cpu()
        mc = do["miss_cnt"].cpu()
        bad = []
        if expect == "safe":
            if not bool((src == -1).all()):
                bad.append(f"topk 应全 -1, got 唯一值 {src.unique().tolist()[:4]}")
            if int(mc[0]) != 0:
                bad.append(f"miss_cnt 应 0, got {int(mc[0])}")
        else:  # loose: ask=20000 值域外, kernel 无 actk≤capacity 守卫 -> 只断言无害
            if int(mc[0]) != 0:
                bad.append(f"loose miss_cnt 应 0(L=C 全驻留), got {int(mc[0])}")
        if not torch.equal(do_pool(di), pool_before):
            bad.append("safe-failure 不应改写池")
        if bad:
            errs.append(f"[invalid {field}={value} expect={expect}] " + "; ".join(bad))
        else:
            passed += 1
    # fp32 q_scale -> host TORCH_CHECK RuntimeError（quant ABI 要求 fp16）
    total += 1
    di, do = to_dev_quant(case, make_outputs(case))
    di["q_scale"] = di["q_scale"].to(torch.float32)
    try:
        call_c8(di, do)
        errs.append("[invalid fp32 q_scale] host 应拒绝 fp32 query_dequant_scale")
    except RuntimeError:
        passed += 1
    # 结构违例(aslq 与 T 不一致): 隔离子进程观测, 不作 PASS/FAIL 门槛
    print(f"  [invalid] structural probe: {_probe_structural_mismatch()}")
    print(f"  [invalid] PASS {passed}/{total}")


def do_pool(di):
    return di["pool"].cpu()


def _probe_structural_mismatch():
    """aslq=[5] 但 T=4: 子进程内单次调用, 观测是否崩溃/污染, 超时即终止。"""
    script = (
        "import sys, torch, torch_npu\n"
        "sys.path.insert(0, %r)\n"
        "import fused_li_manage_mtp_c8_quant_full as Q\n"
        "import vllm_ascend.vllm_ascend_C\n"
        "case = Q.build_src_case([4], [-1], offload_len=8192, cache_tokens=8192,\n"
        "                        source_capacity=16384, rpe=[0], pool_rows=2)\n"
        "di, do = Q.to_dev_quant(case, Q.make_outputs(case))\n"
        "di['aslq'] = torch.tensor([5], dtype=torch.int32, device=Q.DEV)\n"
        "Q.call_c8(di, do)\n"
        "print('probe-done miss_cnt=', int(do['miss_cnt'].cpu()[0]))\n"
    ) % os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=90)
        out = (r.stdout + r.stderr).strip().splitlines()
        tail = repr(out[-1]) if out else "''"
        return f"exit={r.returncode} tail={tail}"
    except subprocess.TimeoutExpired:
        return "timeout(90s) —— 结构违例导致设备侧挂起, 已终止子进程"


# ============================================================= mode: qsweep
def run_qsweep(errs):
    for q in range(1, 15):
        if q <= 7:
            case = build_src_case([q], [-1], offload_len=16256, cache_tokens=12288,
                                  seed=37 + q)
        else:
            # q>=8 活形态守卫线 C>=q*2048 取等, L=C+2048 (128 倍数)
            c = q * 2048
            case = build_src_case([q], [-1], offload_len=c + 2048, cache_tokens=c,
                                  source_capacity=c + 2048 + BLOCK, seed=37 + q)
        check_case_full(case, f"qsweep-q{q}", errs)
    # q=15: 超出 T<=14B 规格, 隔离子进程探针（host 必须拒绝; exit=0=拒绝成立）
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--only", "q15probe"],
        capture_output=True, text=True, timeout=600)
    tail = [ln for ln in r.stdout.splitlines() if "q15probe" in ln][-1:] or ["<no output>"]
    status = "exit=0" if r.returncode == 0 else f"exit={r.returncode}"
    print(f"  [qsweep-q15-probe] {status}  {tail[0].strip()}")
    if r.returncode != 0:
        print(f"    stderr 尾部: {r.stderr.splitlines()[-3:]}")
        errs.append("[qsweep-q15-probe] 子进程未干净退出（拒绝应表现为 RuntimeError, 非崩溃）")


def run_q15probe(errs):
    # q=15 超 T<=14B 规格: host 拒绝即为通过; 未拒绝则记入 errs
    case = build_src_case([15], [-1], offload_len=32768, cache_tokens=30720,
                          source_capacity=32768 + BLOCK, seed=45)
    try:
        check_case_full(case, "q15probe", errs)
    except RuntimeError as e:
        print(f"  [q15probe] host 拒绝（预期）: {str(e)[:70]}")
        return
    errs.append("[q15probe] q=15 未被 host 拒绝")
    print("  [q15probe] 未拒绝 FAIL")


# ============================================================= mode: keytag
def _keytag_case(state):
    upper = TOPK // 2 + BLOCK
    capacity = PACKED + upper
    offload = capacity - BLOCK
    case = build_src_case([1], [state], offload_len=offload, cache_tokens=8192,
                          source_capacity=capacity, seed=41)
    case["query"].fill_(1.0)
    case["weights"].fill_(1.0)
    key = case["key"]
    key.zero_()
    begin, end = PACKED - TOPK // 2, PACKED + upper
    # band 外留小幅随机(非零): 全零会使 per-token 量化 scale 下溢 fp16->0 导致 0/0
    outside = torch.ones(key.shape, dtype=torch.bfloat16) * 0.0625
    outside *= torch.randn(key.shape).sign()
    key.copy_(outside)
    key.reshape(capacity, 1, 128)[begin:end].fill_(1.0)
    return case, begin, end, capacity


def run_keytag(errs):
    # --- -3 标准路径: 21-bit 源 ID 直通 payload, 解析式 golden 全序
    case, begin, end, capacity = _keytag_case(-3)
    res, old = run_once(case)
    expected = torch.arange(begin, begin + TOPK, dtype=torch.int64)
    got = res["topk_src"][0, 0].to(torch.int64)
    if not torch.equal(got, expected):
        if torch.equal(torch.sort(got).values, torch.sort(expected).values):
            errs.append("[keytag-3] topk 集合一致但并列平局顺序与 ID 升序不同")
        else:
            extra = sorted(set(got.tolist()) - set(expected.tolist()))[:5]
            miss = sorted(set(expected.tolist()) - set(got.tolist()))[:5]
            errs.append(f"[keytag-3] topk 集合不一致: kernel多{extra} 期望多{miss}")
    if not bool((res["topk_dst"][0, 0] == res["topk_src"][0, 0]).all()):
        errs.append("[keytag-3] dst 应恒等 src")
    if int(res["topk_miss"][0]) != 0 or int(res["miss_cnt"][0]) != 0:
        errs.append("[keytag-3] miss 应 0")
    if not torch.equal(res["pool_after"][case["req_entries"][0]],
                       torch.arange(capacity, dtype=torch.int32)):
        errs.append("[keytag-3] 池行应整行恒等")

    # --- -1 打包路径: TagLongIndex 高 3 位写入分数 + DecodeTopkHitMiss 重建全 ID
    case1, begin, end, capacity = _keytag_case(-1)
    res1, old1 = run_once(case1)
    got1 = res1["topk_src"][0, 0].to(torch.int64)
    exp_set = set(range(begin, begin + TOPK))
    if set(got1.tolist()) != exp_set:
        extra = sorted(set(got1.tolist()) - exp_set)[:5]
        miss = sorted(exp_set - set(got1.tolist()))[:5]
        errs.append(f"[keytag-1] 21-bit 解码集合不一致: kernel多{extra} 期望多{miss}")
    if not bool((got1 >= PACKED).any()):
        errs.append("[keytag-1] 未覆盖 ≥2^18 高位源（tag 路径未验证）")
    m = int(res1["miss_cnt"][0])
    if m != TOPK:
        errs.append(f"[keytag-1] miss_cnt 应 {TOPK}（band 全部非驻留）, got {m}")
    elif not torch.equal(res1["miss_src"][0, :m], torch.arange(begin, begin + TOPK,
                                                               dtype=torch.int32)):
        errs.append("[keytag-1] miss_src 应=band 升序")
    assert_mgmt_exact(case1, res1, old1, errs, "keytag-1")
    print(f"  [keytag] band=[{begin},{end}) 跨 {PACKED}, -3 全序一致; "
          f"-1 tag 解码 miss={m} 高位覆盖 "
          f"{int((got1 >= PACKED).sum())}/{TOPK}")


# ============================================================= mode: window
def _residual_count(row_dst, row_src, pool_row):
    """§1.5 残留 miss-key 字计数: dst != 池终态映射 且按 float 解读 ∈[2.0,2.5)。"""
    hit = row_src >= 0
    exp = pool_row[row_src[hit].to(torch.int64)].to(torch.int32)
    bad = row_dst[hit] != exp
    words = row_dst[hit][bad].to(torch.int32)
    fv = words.view(torch.float32)
    return int(((fv >= 2.0) & (fv < 2.5)).sum())


def run_window(errs):
    want = "源算子(D)同张量对拍" if src_available() else "源算子不可用, 仅量化侧形态"
    print(f"  [window] {want}")
    case = None
    for attempt in range(8):
        c = build_src_case([4], [-1], offload_len=LONG_L, cache_tokens=12288,
                           source_capacity=LONG_CAP, correlated=True, noise=0.5,
                           shared_w=True, seed=7 + attempt * 13)
        res, old = run_once(c)
        total = int(res["topk_miss"].sum())
        if 6144 < total <= 7936:
            case = c
            break
    if case is None:
        errs.append("[window] 8 个种子未落入侵陷窗口 Σ∈(6144,7936]")
        return
    total = int(res["topk_miss"].sum())
    row = case["req_entries"][0]
    per_route = [int(x) for x in res["topk_miss"].tolist()]
    resid = [_residual_count(res["topk_dst"][t, 0], res["topk_src"][t, 0],
                             res["pool_after"][row]) for t in range(4)]
    # 管理段其余断言仍应精确（miss/union/池/topk_src）；topk_dst 映射断言跳过
    assert_mgmt_exact(case, res, old, errs, "window", skip_dst=True)
    print(f"  [window] quant: Σ={total} 每路={per_route} 残留={resid} 合计={sum(resid)}")
    if src_available():
        resd = run_src_safe(case)
        if resd is not None:
            perd = [int(x) for x in resd["topk_miss"].tolist()]
            residd = [_residual_count(resd["topk_dst"][t, 0], resd["topk_src"][t, 0],
                                      resd["pool_after"][row]) for t in range(4)]
            print(f"  [window] src  : Σ={sum(perd)} 每路={perd} 残留={residd} "
                  f"合计={sum(residd)}")
    if sum(resid) == 0:
        print("  [window] NOTE: 量化版未复现残留（Σ 已在窗口内）——需复核窗口边界")


# ============================================================= mode: matrix
def _grid_weights(case):
    """权重取 /8 网格: bf16/fp16 双精确表示, 使 B(bf16 权重)与 F(fp16 权重)同值。"""
    w = (case["weights"].float() * 8).round() / 8
    case["weights"] = w.to(torch.bfloat16)
    return case


def _li_rows(case, t, ask):
    """E = 官方 LI（bf16, qlen=1 逐 route, -3 按 causal ask）。"""
    bt_row = case["route_table"][t].view(1, -1).to(DEV)
    out = torch_npu.npu_lightning_indexer(
        case["query"][t:t + 1].to(DEV), case["key"].to(DEV),
        case["weights"][t:t + 1].to(DEV),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device=DEV),
        actual_seq_lengths_key=torch.tensor([ask], dtype=torch.int32, device=DEV),
        block_table=bt_row, layout_query="TND", layout_key="PA_BSND",
        sparse_count=TOPK, sparse_mode=0)
    idx = out[0] if isinstance(out, (tuple, list)) else out
    return idx.cpu().flatten()


def _liq_rows(case, qi, qs, ki, ks, t, ask):
    """F = li_quant（int8+fp16 scale, 同 B 的有效输入）。"""
    bt_row = case["route_table"][t].view(1, -1).to(DEV)
    out = torch.ops._C_ascend.npu_lightning_indexer_quant(
        qi[t:t + 1].to(DEV), ki.to(DEV),
        case["weights"][t:t + 1].to(torch.float16).to(DEV),
        query_dequant_scale=qs[t:t + 1].to(DEV), key_dequant_scale=ks.to(DEV),
        query_quant_mode=0, key_quant_mode=0,
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device=DEV),
        actual_seq_lengths_key=torch.tensor([ask], dtype=torch.int32, device=DEV),
        block_table=bt_row, layout_query="TND", layout_key="PA_BSND",
        sparse_count=TOPK, sparse_mode=0)
    idx = out[0] if isinstance(out, (tuple, list)) else out
    return idx.cpu().flatten()


def run_matrix(errs):
    have_d = src_available()
    print(f"  [matrix] D(源算子) {'可用' if have_d else 'SKIP(NANOVLLM_CUST_OPAPI_LIB 未配置)'}")
    scen = [
        ("mix8k", dict(q_values=[1, 3, 7], states=[-3, -2, -1], offload_len=8192)),
        ("steady16k", dict(q_values=[4, 4], states=[-1, -1], offload_len=16256,
                           cache_tokens=12288)),
        ("long264k", dict(q_values=[1, 4, 7], states=[-3, -2, -1],
                          offload_len=LONG_L, source_capacity=LONG_CAP,
                          correlated=True, noise=0.5, shared_w=True)),
    ]
    for name, kw in scen:
        case = _grid_weights(build_src_case(seed=43, **kw))
        qi, qs, ki, ks, q_rec, k_rec = quantize_case(case)
        ref_a, tie_a, vis = golden_on(case, case["query"], case["key"])
        ref_c, _, _ = golden_on(case, q_rec, k_rec)
        res, _ = run_once(case)
        rows_b = [res["topk_src"][t, 0] for t in range(res["topk_src"].shape[0])]
        rows_d = None
        if have_d:
            resd = run_src_safe(case)
            if resd is not None:
                rows_d = [resd["topk_src"][t, 0] for t in range(resd["topk_src"].shape[0])]
        rows_e, rows_f = [], []
        for t in range(len(rows_b)):
            ask = vis[t]
            rows_e.append(_li_rows(case, t, ask))
            rows_f.append(_liq_rows(case, qi, qs, ki, ks, t, ask))
        n = len(rows_b)

        def avg(fn):
            return sum(fn(t) for t in range(n)) / n

        m = {
            "B/C": avg(lambda t: overlap(rows_b[t], ref_c[t])),
            "B/A": avg(lambda t: overlap(rows_b[t], ref_a[t])),
            "B/F": avg(lambda t: overlap(rows_b[t], rows_f[t])),
            "B/E": avg(lambda t: overlap(rows_b[t], rows_e[t])),
            "E/F": avg(lambda t: overlap(rows_e[t], rows_f[t])),
            "E/A": avg(lambda t: overlap(rows_e[t], ref_a[t])),
            "F/C": avg(lambda t: overlap(rows_f[t], ref_c[t])),
        }
        if rows_d is not None:
            m["B/D"] = avg(lambda t: overlap(rows_b[t], rows_d[t]))
            m["D/E"] = avg(lambda t: overlap(rows_d[t], rows_e[t]))
            m["D/A"] = avg(lambda t: overlap(rows_d[t], ref_a[t]))
        order = ["B/C", "B/A", "B/D", "B/E", "B/F", "D/A", "D/E", "E/A", "E/F", "F/C"]
        print(f"  [{name}] " + "  ".join(f"{k}={m[k]:.3f}%" for k in order if k in m))
        if m["B/C"] < TH_BC:
            errs.append(f"[matrix-{name}] B/C {m['B/C']:.3f}% < {TH_BC}%")
        if m["B/A"] < TH_BA:
            errs.append(f"[matrix-{name}] B/A {m['B/A']:.3f}% < {TH_BA}%")
        if m["B/F"] < TH_BC:
            errs.append(f"[matrix-{name}] B/F {m['B/F']:.3f}% < {TH_BC}% "
                        f"(两段 mma 通路等价度)")
        if "D/E" in m and m["D/E"] < 99.0:
            errs.append(f"[matrix-{name}] D/E {m['D/E']:.3f}% < 99.0%")


# ============================================================= main
def warm_operators():
    """任何测试逻辑前预热全部算子的 host 侧惰性解析（二进制匹配/executor 创建）。

    实测: 进程内元数据违例序列 + 子进程 tiling 失败的组合会毒化**尚未解析**算子
    的 EZ1009 二进制匹配（D 在 window/matrix 首调时转"包不可见"）；预热后免疫
    （B/D/E/F 解析状态进程内缓存）。B 预热失败直接抛（后续全部无意义）。"""
    case = build_src_case([4], [-1], offload_len=8192, cache_tokens=8192,
                          source_capacity=16384, rpe=[0], pool_rows=2)
    run_once(case)                                   # B
    qi, qs, ki, ks, _, _ = quantize_case(case)
    ask = int(case["offload_key"][0])
    if src_available():
        run_src_safe(case)                           # D
    try:
        _li_rows(case, 0, ask)                       # E
        _liq_rows(case, qi, qs, ki, ks, 0, ask)      # F
    except Exception as e:  # noqa: BLE001
        print(f"warm: E/F 预热失败（matrix 模式将受影响）: {e}")
    print("warm: B/D/E/F host 解析完成", flush=True)


MODES = {
    "mgmt": run_mgmt,
    "lifecycle": run_lifecycle,
    "warmfix": run_warmfix,
    "boundary": run_boundary,
    "m3tail": run_m3tail,
    "c0": run_c0,
    "invalid": run_invalid,
    "qsweep": run_qsweep,
    "q15probe": run_q15probe,
    "keytag": run_keytag,
    "window": run_window,
    "matrix": run_matrix,
}
# 子进程探针目标模式: 主进程内 tiling 失败会毒化后续算子二进制缓存匹配
# （实测 q8 tiling EZ1008 后同进程内 D 算子转 EZ1009 不可见），仅显式 --only 时执行
HELPER_MODES = {"q15probe"}


def main():
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        only = set(sys.argv[2].split(","))
    errs = []
    warm_operators()
    for name, fn in MODES.items():
        if only and name not in only:
            continue
        if not only and name in HELPER_MODES:
            continue
        print(f"== {name} ==", flush=True)
        try:
            fn(errs)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            errs.append(f"[{name}] EXCEPTION {type(e).__name__}: {e}")
    print("=" * 60)
    if errs:
        print(f"结果: FAIL ({len(errs)} 项)")
        for e in errs:
            print(" -", e)
        return 1
    print("结果: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
