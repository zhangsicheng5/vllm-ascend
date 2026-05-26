import itertools
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import vllm  # noqa: F401

import vllm_ascend.platform  # noqa: F401
from vllm_ascend.ops.triton.get_topk_indices import (
    TOKEN_LIMIT_PER_REQ,
    CacheMissTopKState,
    get_cache_miss_topk_indices_triton,
    get_cache_miss_topk_indices_triton_state,
)


def benchmark_npu(fn, num_iterations=100, num_warmup_iterations=50):
    """Benchmark a function on NPU using event timing."""
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_iterations + num_warmup_iterations)

    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            fn()
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)

    times = times[num_warmup_iterations:]
    return np.amin(times) / 1000  # seconds


def generate_test_data(
    num_reqs: int,
    topk: int,
    token_limit: int,
    overlap: float,
    device: str = "npu",
):
    """Generate synthetic topk data with a given overlap ratio.

    overlap=0.0 means completely new tokens (all miss).
    overlap=1.0 means identical tokens (all hit).
    """
    assert 0.0 <= overlap <= 1.0

    num_common = int(topk * overlap)
    num_new = topk - num_common

    # Common tokens: shared between old and new
    common_tokens = torch.randint(0, token_limit, (num_reqs, num_common), device=device)

    # Old-only tokens: present in old but not new
    old_only = torch.randint(0, token_limit, (num_reqs, num_new), device=device)

    # New-only tokens: present in new but not old
    new_only = torch.randint(0, token_limit, (num_reqs, num_new), device=device)

    # Build old and new tensors
    old_topk_list = []
    new_topk_list = []
    for r in range(num_reqs):
        old_row = torch.cat([common_tokens[r], old_only[r]])
        new_row = torch.cat([common_tokens[r], new_only[r]])
        # Shuffle each row independently
        old_row = old_row[torch.randperm(topk, device=device)]
        new_row = new_row[torch.randperm(topk, device=device)]
        old_topk_list.append(old_row)
        new_topk_list.append(new_row)

    old_topk = torch.stack(old_topk_list).to(torch.int64)
    new_topk = torch.stack(new_topk_list).to(torch.int32)
    return old_topk, new_topk


def run_benchmark():
    has_npu_op = hasattr(torch.ops.npu, "get_cache_miss_topk_indices")

    num_reqs = [1, 4, 8]
    topk_vals = [128, 512, 1024, 2048]
    token_limits = [32768, 65536]
    overlaps = [0.95, 0.90, 0.75, 0.50, 0.0]

    configs = list(
        itertools.product(num_reqs, topk_vals, token_limits, overlaps)
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

        # Stateful benchmark
        state = CacheMissTopKState(
            max_num_reqs=num_reqs, topk=topk, token_limit=token_limit, device="npu"
        )
        # Pre-populate state with old data
        state.slot_to_token[:num_reqs] = old_topk
        for r in range(num_reqs):
            tokens = old_topk[r]
            valid = tokens >= 0
            state.token_to_slot[r, tokens[valid].long()] = torch.arange(
                topk, device="npu"
            )[valid]

        def stateful_fn():
            return get_cache_miss_topk_indices_triton_state(
                req_ids, state, new_topk
            )

        # Warm up and benchmark stateful
        _ = stateful_fn()
        stateful_time = benchmark_npu(stateful_fn)

        # Compute hit/miss stats
        with torch.no_grad():
            result = stateful_fn()
            num_valid_new = (new_topk >= 0).sum().item()
            num_miss = (result >= 0).sum().item()
            hit_rate = 1.0 - num_miss / max(num_valid_new, 1)

        # Fallback benchmark
        if has_npu_op:
            fallback_old = old_topk.clone()

            def fallback_fn():
                r = torch.ops.npu.get_cache_miss_topk_indices(
                    new_topk, fallback_old, req_ids
                )
                return r

            _ = fallback_fn()
            fallback_time = benchmark_npu(fallback_fn)
            fallback_label = "ascendc"
        else:
            fallback_old = old_topk.clone()

            def fallback_fn():
                return get_cache_miss_topk_indices_triton(
                    req_ids, fallback_old, new_topk
                )

            _ = fallback_fn()
            fallback_time = benchmark_npu(fallback_fn)
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
