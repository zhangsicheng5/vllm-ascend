# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the consumer-owned A5 attention plan, not kernel proof."""

import pytest
import torch

from vllm_ascend.models.deepseek_v41.mixed_quant_metadata import prepare_smla_lengths


@pytest.mark.parametrize("heads", [64, 128])
def test_plan_keeps_current_ori_and_selected_cmp_lengths_separate(heads):
    q = torch.empty(3, heads, 512, dtype=torch.bfloat16)
    ori = torch.tensor([4, 6, 9], dtype=torch.int32)
    selected = torch.tensor([[[5, -1, -1]], [[2, 3, 4]], [[-1, -1, -1]]], dtype=torch.int32)
    supplied = torch.tensor([1, 3, 0], dtype=torch.int32)

    ori_rows, cmp_rows = prepare_smla_lengths(q, ori, selected, supplied)
    inferred_ori, inferred_cmp = prepare_smla_lengths(q, ori, selected)

    torch.testing.assert_close(ori_rows, torch.tensor([[4], [6], [9]], dtype=torch.int32))
    torch.testing.assert_close(cmp_rows, torch.tensor([[1], [3], [0]], dtype=torch.int32))
    torch.testing.assert_close(inferred_ori, ori_rows)
    torch.testing.assert_close(inferred_cmp, cmp_rows)
    assert cmp_rows.data_ptr() != ori_rows.data_ptr()


def test_ori_only_uses_zero_compressed_lengths_without_aliasing_ori():
    q = torch.empty(3, 128, 512, dtype=torch.bfloat16)
    ori = torch.tensor([[4, 99], [6, 99], [9, 99]], dtype=torch.int32)[:, :1]
    assert not ori.is_contiguous()

    ori_rows, cmp_rows = prepare_smla_lengths(q, ori)

    assert ori_rows.is_contiguous() and cmp_rows.is_contiguous()
    torch.testing.assert_close(ori_rows[:, 0], torch.tensor([4, 6, 9], dtype=torch.int32))
    torch.testing.assert_close(cmp_rows, torch.zeros((3, 1), dtype=torch.int32))
    assert cmp_rows.data_ptr() != ori_rows.data_ptr()
    torch.testing.assert_close(ori[:, 0], torch.tensor([4, 6, 9], dtype=torch.int32))


@pytest.mark.parametrize("shape", [(3, 32, 512), (3, 128, 256), (3, 128, 1, 512)])
def test_rejects_query_outside_private64_public128_contract(shape):
    with pytest.raises(ValueError, match="query"):
        prepare_smla_lengths(torch.empty(shape), torch.ones(3, dtype=torch.int32))


@pytest.mark.parametrize(
    "bad_lengths",
    [torch.zeros(1, 3, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), torch.zeros(3, dtype=torch.int64)],
)
def test_rejects_length_rows_with_wrong_axis_count_or_dtype(bad_lengths):
    q = torch.empty(3, 128, 512)
    with pytest.raises(ValueError, match="ori_lengths"):
        prepare_smla_lengths(q, bad_lengths)
    with pytest.raises(ValueError, match="cmp_lengths"):
        prepare_smla_lengths(q, torch.ones(3, dtype=torch.int32), torch.zeros(3, 1, 4, dtype=torch.int32), bad_lengths)


@pytest.mark.parametrize("shape,dtype", [((3, 4), torch.int32), ((3, 2, 4), torch.int32), ((3, 1, 4), torch.float32)])
def test_rejects_malformed_compressed_indices(shape, dtype):
    with pytest.raises(ValueError, match="compressed indices"):
        prepare_smla_lengths(
            torch.empty(3, 128, 512), torch.ones(3, dtype=torch.int32), torch.zeros(shape, dtype=dtype)
        )


def test_compressed_lengths_require_a_compressed_index_source():
    with pytest.raises(ValueError, match="require compressed indices"):
        prepare_smla_lengths(
            torch.empty(3, 128, 512),
            torch.ones(3, dtype=torch.int32),
            cmp_lengths=torch.ones(3, dtype=torch.int32),
        )


def test_rejects_lengths_on_another_device_without_reading_values():
    # The meta device lets this contract run without allocating an NPU tensor.
    with pytest.raises(ValueError, match="query device"):
        prepare_smla_lengths(torch.empty(3, 128, 512), torch.empty(3, dtype=torch.int32, device="meta"))
