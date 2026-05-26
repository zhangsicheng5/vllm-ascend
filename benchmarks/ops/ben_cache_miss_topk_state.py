import itertools

import numpy as np
import torch
import torch_npu  # noqa: F401
import vllm  # noqa: F401

import vllm_ascend.platform  # noqa: F401
from vllm_ascend.ops.triton.get_topk_indices import (
    CacheMissTopKState,
    get_cache_miss_topk_indices_triton,
    get_cache_miss_topk_indices_triton_state,
)


def benchmark_with_reset(reset_fn, fn, num_iterations=100, num_warmup=50):
    """Benchmark with state reset before each call. Reset is NOT included in timing."""
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_iterations)

    for i in range(num_warmup + num_iterations):
        reset_fn()
        torch.npu.synchronize()
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        if i >= num_warmup:
            times[i - num_warmup] = start.elapsed_time(end)

    return np.amin(times) / 1000  # seconds


def generate_test_data(
    num_reqs: int,
    topk: int,
    token_limit: int,
    overlap: float,
    device: str = "npu",
):
    """Generate non-overlapping topk data with a guaranteed overlap ratio.

    - common tokens appear in both old and new (identical values)
    - old_only tokens appear only in old
    - new_only tokens appear only in new
    - all tokens within a request are unique
    - overlap=0.0 means all tokens differ, overlap=1.0 means identical sets
    """
    assert 0.0 <= overlap <= 1.0

    num_common = int(topk * overlap)
    num_unique = topk - num_common
    num_needed = num_common + 2 * num_unique  # common + old_only + new_only

    old_rows = []
    new_rows = []
    for _ in range(num_reqs):
        # Sample unique tokens without replacement
        pool = torch.randperm(token_limit, device=device)[:num_needed]
        common = pool[:num_common]
        old_only = pool[num_common:num_common + num_unique]
        new_only = pool[num_common + num_unique:]

        old_row = torch.cat([common, old_only])
        new_row = torch.cat([common, new_only])
        old_rows.append(old_row[torch.randperm(topk, device=device)])
        new_rows.append(new_row[torch.randperm(topk, device=device)])

    old_topk = torch.stack(old_rows).to(torch.int64)
    new_topk = torch.stack(new_rows).to(torch.int32)
    return old_topk, new_topk


def run_benchmark():
    has_npu_op = hasattr(torch.ops.npu, "get_cache_miss_topk_indices")

    num_reqs_vals = [1, 4, 8]
    topk_vals = [128, 512, 1024, 2048]
    token_limits = [32768, 65536]
    overlaps = [0.95, 0.90, 0.75, 0.50, 0.0]

    configs = list(
        itertools.product(num_reqs_vals, topk_vals, token_limits, overlaps)
    )

    print(
        f"{'num_reqs':>8} {'topk':>6} {'token_limit':>12} {'overlap':>8} "
        f"{'hit_rate':>8} {'miss_count':>10} {'stateful_us':>12} "
        f"{'fallback_us':>12} {'speedup':>8}"
    )
    print("-" * 95)

    for num_reqs, topk, token_limit, overlap in configs:
        old_topk, new_topk = generate_test_data(
            num_reqs, topk, token_limit, overlap
        )
        req_ids = torch.arange(num_reqs, dtype=torch.int32, device="npu")

        # Pre-compute initial token_to_slot rows for fast reset
        initial_slots = []
        for r in range(num_reqs):
            tokens = old_topk[r]
            valid = tokens >= 0
            slots = torch.arange(topk, device="npu", dtype=torch.int32)[valid]
            initial_slots.append((tokens[valid].long(), slots))

        # --- Stateful benchmark ---
        state = CacheMissTopKState(
            max_num_reqs=num_reqs, topk=topk, token_limit=token_limit, device="npu"
        )

        def reset_stateful():
            state.slot_to_token[:num_reqs] = old_topk
            state.token_to_slot.fill_(-1)
            for r in range(num_reqs):
                tok, slot = initial_slots[r]
                state.token_to_slot[r, tok] = slot

        def stateful_fn():
            return get_cache_miss_topk_indices_triton_state(
                req_ids, state, new_topk
            )

        # One warmup to compile Triton kernels
        reset_stateful()
        _ = stateful_fn()

        stateful_time = benchmark_with_reset(reset_stateful, stateful_fn)

        # Compute hit/miss stats from a single fresh run
        reset_stateful()
        result = stateful_fn()
        num_valid_new = int((new_topk >= 0).sum().item())
        num_miss = int((result >= 0).sum().item())
        hit_rate = 1.0 - num_miss / max(num_valid_new, 1)

        # --- Fallback benchmark ---
        if has_npu_op:
            fallback_old = old_topk.clone()

            def reset_fallback():
                fallback_old.copy_(old_topk)

            def fallback_fn():
                r = torch.ops.npu.get_cache_miss_topk_indices(
                    new_topk, fallback_old, req_ids
                )
                return r

            reset_fallback()
            _ = fallback_fn()
            fallback_time = benchmark_with_reset(reset_fallback, fallback_fn)
            fallback_label = "ascendc"
        else:
            fallback_old = old_topk.clone()

            def reset_fallback():
                fallback_old.copy_(old_topk)

            def fallback_fn():
                return get_cache_miss_topk_indices_triton(
                    req_ids, fallback_old, new_topk
                )

            reset_fallback()
            _ = fallback_fn()
            fallback_time = benchmark_with_reset(reset_fallback, fallback_fn)
            fallback_label = "triton"

        speedup = fallback_time / stateful_time if stateful_time > 0 else float("inf")

        print(
            f"{num_reqs:>8} {topk:>6} {token_limit:>12} {overlap:>8.2f} "
            f"{hit_rate:>8.4f} {num_miss:>10} {stateful_time * 1e6:>12.1f} "
            f"{fallback_time * 1e6:>12.1f} {speedup:>8.2f}x"
        )

    print(f"\nFallback type: {fallback_label}")
    if not has_npu_op:
        print(
            "Note: torch.ops.npu.get_cache_miss_topk_indices not available, "
            "falling back to triton non-stateful path."
        )


if __name__ == "__main__":
    run_benchmark()
