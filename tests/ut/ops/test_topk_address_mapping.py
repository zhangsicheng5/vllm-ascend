import os
import time

import numpy as np
import pytest


SEQ_LEN = 131072
TOPK = 2048
BLOCK_SIZE = 16
VALID_RATIOS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.7, 1.0)
NUM_REQS_VALUES = (1, 4, 16, 32)
THREAD_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_REPEAT = 30

_BACKEND = None


CPP_SOURCE = r"""
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <torch/extension.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void check_int32_cpu_contiguous(const torch::Tensor& tensor,
                                const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
                name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void map_topk_indices_to_slots(torch::Tensor topk_indices,
                               torch::Tensor block_table,
                               torch::Tensor physical_block_ids,
                               torch::Tensor offsets_in_block,
                               int64_t seq_len,
                               int64_t block_size,
                               int64_t requested_threads) {
    check_int32_cpu_contiguous(topk_indices, "topk_indices");
    check_int32_cpu_contiguous(block_table, "block_table");
    check_int32_cpu_contiguous(physical_block_ids, "physical_block_ids");
    check_int32_cpu_contiguous(offsets_in_block, "offsets_in_block");

    TORCH_CHECK(topk_indices.dim() == 2,
                "topk_indices must have shape [num_reqs, topk]");
    TORCH_CHECK(block_table.dim() == 2,
                "block_table must have shape [num_reqs, max_blocks]");
    TORCH_CHECK(physical_block_ids.sizes() == topk_indices.sizes(),
                "physical_block_ids shape must match topk_indices");
    TORCH_CHECK(offsets_in_block.sizes() == topk_indices.sizes(),
                "offsets_in_block shape must match topk_indices");
    TORCH_CHECK(block_table.size(0) == topk_indices.size(0),
                "block_table num_reqs must match topk_indices");
    TORCH_CHECK(seq_len > 0, "seq_len must be positive");
    TORCH_CHECK(block_size > 0, "block_size must be positive");

    const int64_t num_reqs = topk_indices.size(0);
    const int64_t topk = topk_indices.size(1);
    const int64_t max_blocks = block_table.size(1);
    const int64_t total = num_reqs * topk;
    const int64_t max_required_blocks = (seq_len + block_size - 1) / block_size;
    TORCH_CHECK(max_blocks >= max_required_blocks,
                "block_table does not cover seq_len");

    const auto* topk_ptr = topk_indices.data_ptr<int32_t>();
    const auto* block_table_ptr = block_table.data_ptr<int32_t>();
    auto* physical_ptr = physical_block_ids.data_ptr<int32_t>();
    auto* offset_ptr = offsets_in_block.data_ptr<int32_t>();

    const int active_threads = std::max<int64_t>(
        1, std::min<int64_t>(requested_threads, 128));

#ifdef _OPENMP
    omp_set_num_threads(active_threads);
#pragma omp parallel for schedule(static)
#endif
    for (int64_t linear = 0; linear < total; ++linear) {
        const int64_t req = linear / topk;
        const int32_t token = topk_ptr[linear];

        if (token < 0 || static_cast<int64_t>(token) >= seq_len) {
            physical_ptr[linear] = -1;
            offset_ptr[linear] = -1;
            continue;
        }

        const int64_t block_index = static_cast<int64_t>(token) / block_size;
        if (block_index < 0 || block_index >= max_blocks) {
            physical_ptr[linear] = -1;
            offset_ptr[linear] = -1;
            continue;
        }

        physical_ptr[linear] =
            block_table_ptr[req * max_blocks + block_index];
        offset_ptr[linear] =
            static_cast<int32_t>(static_cast<int64_t>(token) % block_size);
    }
}
"""


def _load_topk_address_mapping_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    torch = pytest.importorskip("torch")
    from torch.utils.cpp_extension import load_inline

    try:
        _BACKEND = load_inline(
            name="topk_address_mapping_omp_test",
            cpp_sources=[CPP_SOURCE],
            functions=["map_topk_indices_to_slots"],
            extra_cflags=["-O3", "-std=c++17", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    except Exception as exc:
        pytest.skip(f"topk address mapping OpenMP extension unavailable: {exc}")

    return _BACKEND


def test_topk_address_mapping_handles_boundary_tokens():
    backend = _load_topk_address_mapping_backend()
    torch = pytest.importorskip("torch")
    seq_len = SEQ_LEN
    max_blocks = seq_len // BLOCK_SIZE
    topk_indices = torch.tensor(
        [[0, BLOCK_SIZE - 1, BLOCK_SIZE, seq_len - 1, -1, seq_len]],
        dtype=torch.int32,
    )
    block_table = torch.arange(max_blocks, dtype=torch.int32).reshape(1, -1)
    physical_block_ids = torch.empty_like(topk_indices)
    offsets_in_block = torch.empty_like(topk_indices)

    backend.map_topk_indices_to_slots(
        topk_indices,
        block_table,
        physical_block_ids,
        offsets_in_block,
        seq_len,
        BLOCK_SIZE,
        1,
    )

    np.testing.assert_array_equal(
        physical_block_ids.numpy(),
        np.array([[0, 0, 1, max_blocks - 1, -1, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        offsets_in_block.numpy(),
        np.array(
            [[0, BLOCK_SIZE - 1, 0, BLOCK_SIZE - 1, -1, -1]],
            dtype=np.int32,
        ),
    )


def _make_case(num_reqs: int, valid_ratio: float, seed: int = 0):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    valid_count = int(round(TOPK * valid_ratio))
    topk_indices = np.full((num_reqs, TOPK), -1, dtype=np.int32)
    for req in range(num_reqs):
        if valid_count > 0:
            tokens = rng.choice(SEQ_LEN, size=valid_count, replace=False)
            rng.shuffle(tokens)
            topk_indices[req, :valid_count] = tokens.astype(
                np.int32,
                copy=False,
            )
            rng.shuffle(topk_indices[req])

    max_blocks = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = np.empty((num_reqs, max_blocks), dtype=np.int32)
    for req in range(num_reqs):
        block_table[req] = req * max_blocks + np.arange(
            max_blocks,
            dtype=np.int32,
        )

    return (
        torch.from_numpy(topk_indices),
        torch.from_numpy(block_table),
        valid_count,
    )


def _reference_address_mapping(topk_indices, block_table):
    topk_np = topk_indices.numpy()
    block_table_np = block_table.numpy()
    physical = np.full(topk_np.shape, -1, dtype=np.int32)
    offsets = np.full(topk_np.shape, -1, dtype=np.int32)
    num_reqs, topk = topk_np.shape
    for req in range(num_reqs):
        for slot in range(topk):
            token = int(topk_np[req, slot])
            if token < 0 or token >= SEQ_LEN:
                continue
            block_index = token // BLOCK_SIZE
            physical[req, slot] = block_table_np[req, block_index]
            offsets[req, slot] = token % BLOCK_SIZE
    return physical, offsets


@pytest.mark.parametrize("valid_ratio", VALID_RATIOS)
@pytest.mark.parametrize("num_reqs", NUM_REQS_VALUES)
def test_topk_address_mapping_matches_python_reference(num_reqs, valid_ratio):
    backend = _load_topk_address_mapping_backend()
    torch = pytest.importorskip("torch")
    topk_indices, block_table, _ = _make_case(
        num_reqs=num_reqs,
        valid_ratio=valid_ratio,
        seed=17 + num_reqs + int(valid_ratio * 100),
    )
    physical_block_ids = torch.empty_like(topk_indices)
    offsets_in_block = torch.empty_like(topk_indices)

    backend.map_topk_indices_to_slots(
        topk_indices,
        block_table,
        physical_block_ids,
        offsets_in_block,
        SEQ_LEN,
        BLOCK_SIZE,
        8,
    )

    expected_physical, expected_offsets = _reference_address_mapping(
        topk_indices,
        block_table,
    )
    np.testing.assert_array_equal(
        physical_block_ids.numpy(),
        expected_physical,
    )
    np.testing.assert_array_equal(offsets_in_block.numpy(), expected_offsets)


def _time_mapping(backend,
                  topk_indices,
                  block_table,
                  physical_block_ids,
                  offsets_in_block,
                  threads: int,
                  repeat: int):
    best_ms = float("inf")
    total_ms = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        backend.map_topk_indices_to_slots(
            topk_indices,
            block_table,
            physical_block_ids,
            offsets_in_block,
            SEQ_LEN,
            BLOCK_SIZE,
            threads,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        best_ms = min(best_ms, elapsed_ms)
        total_ms += elapsed_ms
    return total_ms / repeat, best_ms


def test_topk_address_mapping_openmp_performance(capsys):
    backend = _load_topk_address_mapping_backend()
    torch = pytest.importorskip("torch")
    repeat = int(os.environ.get("TOPK_ADDR_MAP_REPEAT", DEFAULT_REPEAT))
    rows = []

    for num_reqs in NUM_REQS_VALUES:
        for valid_ratio in VALID_RATIOS:
            topk_indices, block_table, valid_count = _make_case(
                num_reqs=num_reqs,
                valid_ratio=valid_ratio,
                seed=1000 + num_reqs + int(valid_ratio * 1000),
            )
            physical_block_ids = torch.empty_like(topk_indices)
            offsets_in_block = torch.empty_like(topk_indices)

            for threads in THREAD_VALUES:
                avg_ms, best_ms = _time_mapping(
                    backend,
                    topk_indices,
                    block_table,
                    physical_block_ids,
                    offsets_in_block,
                    threads=threads,
                    repeat=repeat,
                )
                rows.append((
                    num_reqs,
                    valid_ratio,
                    valid_count,
                    threads,
                    avg_ms,
                    best_ms,
                ))

    with capsys.disabled():
        for (
            num_reqs,
            valid_ratio,
            valid_count,
            threads,
            avg_ms,
            best_ms,
        ) in rows:
            print(
                "TOPK_ADDR_MAP_RESULT "
                f"num_reqs={num_reqs} seq_len={SEQ_LEN} topk={TOPK} "
                f"valid_ratio={valid_ratio:.2f} valid_count={valid_count} "
                f"threads={threads} avg_ms={avg_ms:.6f} "
                f"best_ms={best_ms:.6f}",
                flush=True,
            )
