import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "ops"
    / "bench_cpu_cache_miss_topk.py"
)
SPEC = importlib.util.spec_from_file_location("bench_cpu_cache_miss_topk",
                                              MODULE_PATH)
bench_cpu = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_cpu
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_cpu)

_CPP_BACKEND = None
_CPP_BACKEND_LOAD_ATTEMPTED = False
LEGACY_REQ_ID_OFFSET_STRIDE = 1 << 16
V0_32K_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "cache_miss_topk"
    / "v0_32k"
)
V0_32K_FIXTURE_RE = re.compile(
    r"pid_(?P<pid>\d+)_tp_(?P<tp>\d+)_model_layers_(?P<layer>\d+)"
    r"_self_attn_attn_case_(?P<case>\d+)\.pt$")


def _v0_32k_fixture_paths():
    return sorted(V0_32K_FIXTURE_DIR.glob("pid_*.pt"))


def _cpu_token_indices(topk_indices, num_offloaded_blocks, block_size):
    threshold = num_offloaded_blocks[:, None].astype(np.int64) * int(block_size)
    valid = topk_indices >= 0
    return np.where((topk_indices < threshold) & valid, topk_indices,
                    -1).astype(np.int32)


def _legacy_get_cache_miss_topk_indices_reference(
    req_ids: np.ndarray,
    last_req_ids: np.ndarray,
    old_raw: np.ndarray,
    new_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the old offset-based get_cache_miss_topk_indices behavior.

    This is only a parity oracle for valid topk inputs whose valid tokens are
    unique within each row. Repeated new tokens are deliberately covered by
    separate robustness tests because the optimized CPU path deduplicates them.
    """
    req_offsets = req_ids[:, None] * LEGACY_REQ_ID_OFFSET_STRIDE
    last_offsets = last_req_ids[:, None] * LEGACY_REQ_ID_OFFSET_STRIDE
    old_encoded = np.where(old_raw >= 0, old_raw + last_offsets,
                           -1).astype(np.int64)
    new_encoded = np.where(new_raw >= 0, new_raw + req_offsets,
                           -1).astype(np.int64)
    out_encoded = new_encoded
    bs, topk = old_encoded.shape

    for row in range(bs):
        old_row = old_encoded[row]
        new_row = out_encoded[row]

        cache_miss_mask = np.logical_not(np.isin(new_row, old_row))
        available_slot_mask = np.logical_not(np.isin(old_row, new_row))
        num_tokens_to_load = int(cache_miss_mask.sum())
        num_available_slot = int(available_slot_mask.sum())
        shortage = num_tokens_to_load - num_available_slot

        empty_slot_mask = old_row == -1
        selected_empty_slot_mask = (
            np.cumsum(empty_slot_mask) <= shortage) & empty_slot_mask
        available_slot_mask = np.where(selected_empty_slot_mask, True,
                                       available_slot_mask)

        tokens_to_load = new_row[cache_miss_mask]
        if tokens_to_load.size != int(available_slot_mask.sum()):
            raise AssertionError(
                "legacy parity helper only supports unique valid topk inputs")

        new_row.fill(-1)
        new_row[available_slot_mask] = tokens_to_load
        old_encoded[row] = np.where(available_slot_mask, new_row, old_row)

    out_raw = np.where(out_encoded >= 0, out_encoded - req_offsets,
                       -1).astype(np.int32)
    old_decoded = np.where(old_encoded >= 0, old_encoded - req_offsets,
                           -1).astype(np.int64)
    return old_decoded, out_raw


def _canonical_v0_get_cache_miss_topk_indices_reference(
    req_ids: np.ndarray,
    old_encoded: np.ndarray,
    new_raw: np.ndarray,
    max_token: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the original v0 torch get_cache_miss_topk_indices.

    The canonical v0 path encoded current-step tokens with
    req_id * (1 << 16), performed set-diff in encoded space, updated the
    encoded history buffer, then returned the output tokens decoded back to
    raw token ids. It is only unambiguous for max_token <= 65536.
    """
    assert max_token <= LEGACY_REQ_ID_OFFSET_STRIDE
    req_offsets = req_ids[:, None].astype(
        np.int64) * LEGACY_REQ_ID_OFFSET_STRIDE
    old_work = old_encoded.copy().astype(np.int64, copy=False)
    out_encoded = np.where(
        new_raw >= 0,
        new_raw.astype(np.int64) + req_offsets,
        -1,
    )

    for row in range(old_work.shape[0]):
        old_row = old_work[row]
        new_row = out_encoded[row]

        cache_miss_mask = np.logical_not(np.isin(new_row, old_row))
        available_slot_mask = np.logical_not(np.isin(old_row, new_row))
        num_tokens_to_load = int(cache_miss_mask.sum())
        num_available_slot = int(available_slot_mask.sum())
        shortage = num_tokens_to_load - num_available_slot

        empty_slot_mask = old_row == -1
        selected_empty_slot_mask = (
            np.cumsum(empty_slot_mask) <= shortage) & empty_slot_mask
        available_slot_mask = np.where(selected_empty_slot_mask, True,
                                       available_slot_mask)

        tokens_to_load = new_row[cache_miss_mask]
        if tokens_to_load.size != int(available_slot_mask.sum()):
            raise AssertionError(
                "canonical v0 helper requires valid unique topk inputs")

        new_row.fill(-1)
        new_row[available_slot_mask] = tokens_to_load
        old_work[row] = np.where(available_slot_mask, new_row, old_row)

    out_raw = np.where(out_encoded >= 0, out_encoded - req_offsets,
                       -1).astype(np.int32)
    return old_work, out_raw


def _derive_valid_v2_input_from_canonical_a(
    req_ids: np.ndarray,
    old_encoded: np.ndarray,
    max_token: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode a valid v0 encoded state into v2 raw state.

    Dirty old slots are not normalized here. If any valid encoded slot cannot
    be decoded with the current row req_id, the case is outside v2 exact parity
    input and must be skipped.
    """
    assert max_token <= LEGACY_REQ_ID_OFFSET_STRIDE
    old_raw = np.full(old_encoded.shape, -1, dtype=np.int64)
    for row in range(old_encoded.shape[0]):
        offset = int(req_ids[row]) * LEGACY_REQ_ID_OFFSET_STRIDE
        valid = old_encoded[row] >= 0
        decoded = old_encoded[row, valid] - offset
        if np.any((decoded < 0) | (decoded >= max_token)):
            return None
        old_raw[row, np.flatnonzero(valid)] = decoded
    last_req_ids = req_ids.copy().astype(np.int64)
    return old_raw, last_req_ids


def _decode_valid_canonical_old_to_raw(
    req_ids: np.ndarray,
    old_encoded: np.ndarray,
    max_token: int,
) -> np.ndarray:
    decoded = _derive_valid_v2_input_from_canonical_a(
        req_ids,
        old_encoded,
        max_token=max_token,
    )
    assert decoded is not None
    old_raw, _ = decoded
    return old_raw


def _assert_current_reference_matches_legacy(
    req_ids: np.ndarray,
    last_req_ids: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
):
    expected_old, expected_out = _legacy_get_cache_miss_topk_indices_reference(
        req_ids, last_req_ids, old, new)
    ref_last_req_ids = last_req_ids.copy()
    ref_old = old.copy()
    ref_new = new.copy()

    actual = bench_cpu.reference_update_topk_indices(
        req_ids, ref_last_req_ids, ref_old, ref_new)

    np.testing.assert_array_equal(actual, expected_out)
    np.testing.assert_array_equal(ref_old, expected_old)
    np.testing.assert_array_equal(ref_last_req_ids, req_ids)
    assert actual is ref_new
    return expected_old, expected_out


def _assert_numba_matches_legacy(
    req_ids: np.ndarray,
    last_req_ids: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    max_token: int,
):
    if bench_cpu.NUMBA_AVAILABLE is False:
        pytest.skip("numba is not installed in this environment")

    expected_old, expected_out = _legacy_get_cache_miss_topk_indices_reference(
        req_ids, last_req_ids, old, new)
    nb_last_req_ids = last_req_ids.copy()
    nb_old = old.copy()
    nb_new = new.copy()
    workspace = bench_cpu.make_topk_workspace(
        topk=new.shape[1],
        max_token=max_token,
    )

    actual = bench_cpu.update_topk_indices_cpu(
        req_ids,
        nb_last_req_ids,
        nb_old,
        nb_new,
        workspace,
    )

    np.testing.assert_array_equal(actual, expected_out)
    np.testing.assert_array_equal(nb_old, expected_old)
    np.testing.assert_array_equal(nb_last_req_ids, req_ids)
    assert actual is nb_new


def _assert_cpp_matches_legacy(
    req_ids: np.ndarray,
    last_req_ids: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    max_token: int,
):
    expected_old, expected_out = _legacy_get_cache_miss_topk_indices_reference(
        req_ids, last_req_ids, old, new)

    cpp_last_req_ids, cpp_old, actual = _cpp_update_topk_indices(
        req_ids,
        last_req_ids,
        old,
        new,
        max_token=max_token,
        requested_threads=64,
        workspace_threads=64,
    )

    np.testing.assert_array_equal(actual, expected_out)
    np.testing.assert_array_equal(cpp_old, expected_old)
    np.testing.assert_array_equal(cpp_last_req_ids, req_ids)


def _load_cpp_backend():
    global _CPP_BACKEND, _CPP_BACKEND_LOAD_ATTEMPTED
    if _CPP_BACKEND_LOAD_ATTEMPTED:
        if _CPP_BACKEND is None:
            pytest.skip("cpu_sparse_attn extension is unavailable")
        return _CPP_BACKEND

    _CPP_BACKEND_LOAD_ATTEMPTED = True
    pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")

    from torch.utils.cpp_extension import load

    src_path = (Path(__file__).resolve().parents[3] / "vllm_ascend" /
                "distributed" / "kv_transfer" / "kv_pool" /
                "ascend_store" / "cpu_sparse_attn.cpp")
    ascend_home = os.environ.get("ASCEND_HOME_PATH",
                                 "/usr/local/Ascend/ascend-toolkit/latest")
    npu_include_path = os.path.join(ascend_home, "include")
    npu_lib_path = os.path.join(ascend_home, "lib64")
    if not os.path.exists(npu_lib_path):
        npu_lib_path = os.path.join(ascend_home, "lib")
    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
    os.environ.setdefault("CXX", "clang++")
    os.environ.setdefault("CC", "clang")

    try:
        _CPP_BACKEND = load(
            name="cpu_sparse_attn_cache_miss_topk_test",
            sources=[str(src_path)],
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-funroll-loops",
                "-fomit-frame-pointer",
                "-fopenmp",
                "-march=armv8.2-a+sve+fp16+bf16",
                "-fPIC",
                f"-I{npu_include_path}",
                f"-I{torch_npu_include}",
            ],
            extra_ldflags=[
                "-fopenmp",
                f"-L{npu_lib_path}",
                "-lascendcl",
                f"-L{torch_npu_lib_path}",
                "-ltorch_npu",
            ],
            verbose=False,
        )
    except Exception as exc:
        pytest.skip(f"cpu_sparse_attn extension is unavailable: {exc}")

    if not hasattr(_CPP_BACKEND, "cache_miss_topk"):
        pytest.skip("cpu_sparse_attn.cache_miss_topk is unavailable")
    return _CPP_BACKEND


def _cpp_update_topk_indices(req_ids,
                             last_req_ids,
                             old,
                             new,
                             max_token,
                             requested_threads=1,
                             workspace_threads=4):
    torch = pytest.importorskip("torch")
    backend = _load_cpp_backend()

    req_ids_tensor = torch.from_numpy(req_ids.copy())
    last_req_ids_tensor = torch.from_numpy(last_req_ids.copy())
    old_tensor = torch.from_numpy(old.copy())
    new_tensor = torch.from_numpy(new.copy())
    mark_workspace = torch.zeros([workspace_threads, max_token],
                                 dtype=torch.int32)
    miss_workspace = torch.empty([workspace_threads, new.shape[1]],
                                 dtype=torch.int32)
    epochs = torch.zeros([workspace_threads], dtype=torch.int32)

    backend.cache_miss_topk(
        req_ids_tensor.data_ptr(),
        last_req_ids_tensor.data_ptr(),
        old_tensor.data_ptr(),
        new_tensor.data_ptr(),
        mark_workspace.data_ptr(),
        miss_workspace.data_ptr(),
        epochs.data_ptr(),
        req_ids.shape[0],
        new.shape[1],
        max_token,
        workspace_threads,
        requested_threads,
    )
    return last_req_ids_tensor.numpy(), old_tensor.numpy(), new_tensor.numpy()


def test_reference_matches_spec_example_with_raw_tokens():
    req_ids = np.array([3], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, last_req_ids, old,
                                                  new)

    np.testing.assert_array_equal(out,
                                  np.array([[5, -1, 18, -1, 30, 33]],
                                           dtype=np.int32))
    np.testing.assert_array_equal(
        old, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int64))
    np.testing.assert_array_equal(last_req_ids, req_ids)
    assert out is new


def test_baseline_parity_matches_spec_example():
    req_ids = np.array([3], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)

    _assert_current_reference_matches_legacy(req_ids, last_req_ids, old, new)
    _assert_numba_matches_legacy(req_ids, last_req_ids, old, new, max_token=128)
    _assert_cpp_matches_legacy(req_ids, last_req_ids, old, new, max_token=128)


def test_baseline_parity_matches_all_empty_old():
    req_ids = np.array([5], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[-1, -1, -1, -1]], dtype=np.int64)
    new = np.array([[8, 9, 10, 11]], dtype=np.int32)

    _assert_current_reference_matches_legacy(req_ids, last_req_ids, old, new)
    _assert_numba_matches_legacy(req_ids, last_req_ids, old, new, max_token=128)
    _assert_cpp_matches_legacy(req_ids, last_req_ids, old, new, max_token=128)


def test_baseline_parity_matches_mixed_req_owner_rows():
    req_ids = np.array([3, 9, 11], dtype=np.int64)
    last_req_ids = np.array([3, 7, 11], dtype=np.int64)
    old = np.array([
        [3, 7, 15, 22, -1, -1],
        [1, 2, 3, -1, -1, -1],
        [4, 8, 12, 16, -1, -1],
    ], dtype=np.int64)
    new = np.array([
        [5, 7, 18, 22, 30, 33],
        [2, 6, 9, 10, 11, 12],
        [4, 8, 13, 16, 17, 18],
    ], dtype=np.int32)

    _assert_current_reference_matches_legacy(req_ids, last_req_ids, old, new)
    _assert_numba_matches_legacy(req_ids, last_req_ids, old, new, max_token=128)
    _assert_cpp_matches_legacy(req_ids, last_req_ids, old, new, max_token=128)


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [32768, 131072])
@pytest.mark.parametrize("hit_rate", [0.0, 0.5, 0.9, 1.0])
def test_baseline_parity_matches_random_unique_topk(num_reqs, seq_len,
                                                    hit_rate):
    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=num_reqs,
        seq_len=seq_len,
        topk=2048,
        hit_rate=hit_rate,
        seed=1000 + num_reqs + int(hit_rate * 100),
    )

    _assert_current_reference_matches_legacy(req_ids, last_req_ids, old, new)
    _assert_numba_matches_legacy(req_ids,
                                 last_req_ids,
                                 old,
                                 new,
                                 max_token=seq_len)
    _assert_cpp_matches_legacy(req_ids,
                               last_req_ids,
                               old,
                               new,
                               max_token=seq_len)


def test_reference_treats_changed_req_id_as_all_miss():
    req_ids = np.array([7], dtype=np.int64)
    last_req_ids = np.array([3], dtype=np.int64)
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, last_req_ids, old,
                                                  new)

    np.testing.assert_array_equal(
        out, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32))
    np.testing.assert_array_equal(
        old, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int64))
    np.testing.assert_array_equal(last_req_ids, req_ids)


def test_reference_keeps_same_raw_token_per_row():
    req_ids = np.array([3, 7], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([
        [5, 10, -1, -1],
        [5, 20, -1, -1],
    ], dtype=np.int64)
    new = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, last_req_ids, old,
                                                  new)

    np.testing.assert_array_equal(out,
                                  np.array([
                                      [-1, 30, 31, -1],
                                      [-1, 40, 41, -1],
                                  ], dtype=np.int32))
    expected_old = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ], dtype=np.int64)
    np.testing.assert_array_equal(old, expected_old)
    np.testing.assert_array_equal(last_req_ids, req_ids)


def test_reference_deduplicates_repeated_new_tokens():
    # Repeated new tokens are robustness coverage, not legacy parity coverage.
    req_ids = np.array([11], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[1, 2, 3, -1, -1, -1]], dtype=np.int64)
    new = np.array([[2, 4, 4, 5, 5, 6]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, last_req_ids, old,
                                                  new)

    assert sorted(out[out >= 0].tolist()) == [4, 5, 6]
    assert sorted(old[old >= 0].tolist()) == [2, 4, 5, 6]
    assert old[0, 0] == 4
    np.testing.assert_array_equal(last_req_ids, req_ids)


def test_v0_32k_fixture_grid_is_complete():
    paths = _v0_32k_fixture_paths()
    assert len(paths) == 150

    seen = set()
    pids = set()
    tps = set()
    for path in paths:
        match = V0_32K_FIXTURE_RE.match(path.name)
        assert match is not None
        layer = int(match.group("layer"))
        case_id = int(match.group("case"))
        assert 0 <= layer < 10
        assert 0 <= case_id < 15
        seen.add((layer, case_id))
        pids.add(match.group("pid"))
        tps.add(match.group("tp"))

    assert seen == {(layer, case_id) for layer in range(10)
                    for case_id in range(15)}
    assert len(pids) == 1
    assert tps == {"0"}


def test_cpp_v2_matches_canonical_v0_baseline_on_filtered_v0_32k_inputs(
):
    torch = pytest.importorskip("torch")
    paths = _v0_32k_fixture_paths()
    assert paths

    used_fixture_count = 0
    parity_max_token = LEGACY_REQ_ID_OFFSET_STRIDE
    for path in paths:
        case = torch.load(path, map_location="cpu")
        req_ids = case["req_ids"].numpy().astype(np.int64)
        old_before_encoded = case[
            "last_step_topk_indices_before"].numpy().astype(np.int64)
        new_before = case["topk_indices_new_before"].numpy().astype(np.int32)
        valid_new = new_before[new_before >= 0]
        if valid_new.size and int(valid_new.max()) >= parity_max_token:
            continue

        decoded_input = _derive_valid_v2_input_from_canonical_a(
            req_ids,
            old_before_encoded,
            max_token=parity_max_token,
        )
        if decoded_input is None:
            continue
        old_raw_before, last_req_ids = decoded_input

        expected_old_encoded, expected_new_after = (
            _canonical_v0_get_cache_miss_topk_indices_reference(
                req_ids,
                old_before_encoded,
                new_before,
                max_token=parity_max_token,
            ))
        expected_old_raw = _decode_valid_canonical_old_to_raw(
            req_ids,
            expected_old_encoded,
            max_token=parity_max_token,
        )

        updated_last_req_ids, actual_old_raw, actual_new_after = (
            _cpp_update_topk_indices(
                req_ids,
                last_req_ids,
                old_raw_before,
                new_before,
                max_token=parity_max_token,
                requested_threads=64,
                workspace_threads=64,
            ))

        np.testing.assert_array_equal(
            actual_new_after,
            expected_new_after,
            err_msg=f"topk_indices_new_after mismatch for {path.name}",
        )
        np.testing.assert_array_equal(
            actual_old_raw,
            expected_old_raw,
            err_msg=f"raw old state mismatch for {path.name}",
        )
        np.testing.assert_array_equal(
            updated_last_req_ids,
            req_ids,
            err_msg=f"last_req_ids state mismatch for {path.name}",
        )

        expected_cpu_token_indices = _cpu_token_indices(
            expected_new_after,
            case["num_offloaded_blocks"].numpy().astype(np.int32),
            int(case["block_size"]),
        )
        actual_cpu_token_indices = _cpu_token_indices(
            actual_new_after,
            case["num_offloaded_blocks"].numpy().astype(np.int32),
            int(case["block_size"]),
        )
        np.testing.assert_array_equal(
            actual_cpu_token_indices,
            expected_cpu_token_indices,
            err_msg=f"cpu_token_indices mismatch for {path.name}",
        )
        used_fixture_count += 1

    assert used_fixture_count > 0


def test_req_id_raw_state_fixes_legacy_stride_collision():
    req_ids = np.array([0], dtype=np.int64)
    last_req_ids = np.array([1], dtype=np.int64)
    old = np.array([[0, -1]], dtype=np.int64)
    new = np.array([[65536, 1]], dtype=np.int32)

    legacy_old, legacy_out = _legacy_get_cache_miss_topk_indices_reference(
        req_ids, last_req_ids, old, new)
    assert legacy_out.tolist() == [[-1, 1]]
    assert legacy_old.tolist() == [[65536, 1]]

    ref_last_req_ids = last_req_ids.copy()
    ref_old = old.copy()
    ref_new = new.copy()
    actual = bench_cpu.reference_update_topk_indices(
        req_ids,
        ref_last_req_ids,
        ref_old,
        ref_new,
    )

    np.testing.assert_array_equal(actual,
                                  np.array([[65536, 1]], dtype=np.int32))
    np.testing.assert_array_equal(ref_old,
                                  np.array([[65536, 1]], dtype=np.int64))
    np.testing.assert_array_equal(ref_last_req_ids, req_ids)


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [32768, 131072])
@pytest.mark.parametrize("hit_rate", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_generate_case_exact_overlap(num_reqs, seq_len, hit_rate):
    topk = 2048

    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=num_reqs,
        seq_len=seq_len,
        topk=topk,
        hit_rate=hit_rate,
        seed=123,
    )

    assert req_ids.shape == (num_reqs,)
    assert last_req_ids.shape == (num_reqs,)
    assert old.shape == (num_reqs, topk)
    assert new.shape == (num_reqs, topk)
    assert req_ids.dtype == np.int64
    assert last_req_ids.dtype == np.int64
    assert old.dtype == np.int64
    assert new.dtype == np.int32
    assert old.flags.c_contiguous
    assert new.flags.c_contiguous
    assert int(new.min()) >= 0
    assert int(new.max()) < seq_len

    expected_overlap = int(round(topk * hit_rate))
    for row in range(num_reqs):
        old_raw = old[row]
        assert int(old_raw.min()) >= 0
        assert int(old_raw.max()) < seq_len
        overlap = len(set(old_raw.tolist()) & set(new[row].tolist()))
        assert overlap == expected_overlap


def test_numpy_baseline_matches_reference_with_req_ids():
    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=256,
        hit_rate=0.3,
        seed=7,
    )
    ref_old = old.copy()
    ref_last_req_ids = last_req_ids.copy()
    ref_new = new.copy()
    np_old = old.copy()
    np_last_req_ids = last_req_ids.copy()
    np_new = new.copy()

    expected = bench_cpu.reference_update_topk_indices(
        req_ids, ref_last_req_ids, ref_old, ref_new)
    actual = bench_cpu.numpy_isin_update_topk_indices(
        req_ids, np_last_req_ids, np_old, np_new)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np_old, ref_old)
    np.testing.assert_array_equal(np_last_req_ids, ref_last_req_ids)
    assert actual is np_new


def test_numba_stamp_matches_reference_when_numba_available():
    if bench_cpu.NUMBA_AVAILABLE is False:
        pytest.skip("numba is not installed in this environment")

    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=512,
        hit_rate=0.7,
        seed=11,
    )
    ref_old = old.copy()
    ref_last_req_ids = last_req_ids.copy()
    ref_new = new.copy()
    nb_old = old.copy()
    nb_last_req_ids = last_req_ids.copy()
    nb_new = new.copy()

    expected = bench_cpu.reference_update_topk_indices(
        req_ids, ref_last_req_ids, ref_old, ref_new)
    workspace = bench_cpu.make_topk_workspace(topk=512, max_token=32768)
    actual = bench_cpu.update_topk_indices_cpu(req_ids, nb_last_req_ids,
                                               nb_old, nb_new, workspace)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(nb_old, ref_old)
    np.testing.assert_array_equal(nb_last_req_ids, ref_last_req_ids)
    assert actual is nb_new


def test_cpp_backend_matches_reference_when_available():
    cases = []

    req_ids = np.array([3], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32),
        128,
    ))

    req_ids = np.array([3, 7], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([
        [5, 10, -1, -1],
        [5, 20, -1, -1],
    ], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([
            [5, 30, 31, -1],
            [5, 40, 41, -1],
        ], dtype=np.int32),
        128,
    ))

    req_ids = np.array([11], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[1, 2, 3, -1, -1, -1]], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([[2, 4, 4, 5, 5, 6]], dtype=np.int32),
        128,
    ))

    req_ids = np.array([5], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[-1, -1, -1, -1]], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([[8, 9, 10, 11]], dtype=np.int32),
        128,
    ))

    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=512,
        hit_rate=0.7,
        seed=13,
    )
    cases.append((req_ids, last_req_ids, old, new, 32768))

    req_ids = np.array([7], dtype=np.int64)
    last_req_ids = np.array([3], dtype=np.int64)
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32),
        128,
    ))

    for req_ids, last_req_ids, old, new, max_token in cases:
        ref_old = old.copy()
        ref_last_req_ids = last_req_ids.copy()
        ref_new = new.copy()
        expected = bench_cpu.reference_update_topk_indices(
            req_ids, ref_last_req_ids, ref_old, ref_new)

        cpp_last_req_ids, cpp_old, actual = _cpp_update_topk_indices(
            req_ids,
            last_req_ids,
            old,
            new,
            max_token=max_token,
        )

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(cpp_old, ref_old)
        np.testing.assert_array_equal(cpp_last_req_ids, ref_last_req_ids)


@pytest.mark.parametrize("requested_threads", [-1, 1, 2, 4, 8, 16, 32, 64])
def test_cpp_backend_thread_counts_match_reference_when_available(
        requested_threads):
    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=512,
        hit_rate=0.7,
        seed=17,
    )
    ref_old = old.copy()
    ref_last_req_ids = last_req_ids.copy()
    ref_new = new.copy()
    expected = bench_cpu.reference_update_topk_indices(
        req_ids, ref_last_req_ids, ref_old, ref_new)

    cpp_last_req_ids, cpp_old, actual = _cpp_update_topk_indices(
        req_ids,
        last_req_ids,
        old,
        new,
        max_token=32768,
        requested_threads=requested_threads,
        workspace_threads=64,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cpp_old, ref_old)
    np.testing.assert_array_equal(cpp_last_req_ids, ref_last_req_ids)
