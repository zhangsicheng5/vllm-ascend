# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side lengths for the packed A5 MQSMLA planner."""

import torch

SUPPORTED_QUERY_HEADS = (64, 128)
QUERY_HEAD_DIM = 512


def prepare_smla_lengths(
    q: torch.Tensor,
    ori_lengths: torch.Tensor,
    cmp_indices: torch.Tensor | None = None,
    cmp_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the current window and selected compressed prefix, without host reads.

    Index selection precedes this call on the consumer stream. The prepared
    indexer output has valid indices first and a negative padded suffix.
    Separate lengths are required even when ORI/CMP happen to have equal sizes.
    """
    if q.ndim != 3 or q.shape[1] not in SUPPORTED_QUERY_HEADS or q.shape[2] != QUERY_HEAD_DIM:
        raise ValueError("A5 MQSMLA requires query [T,64|128,512]")
    rows = q.shape[0]

    def normalize(value: torch.Tensor | None, name: str) -> torch.Tensor:
        if value is None or value.dtype != torch.int32 or value.device != q.device:
            raise ValueError(f"{name} must be INT32 on the query device")
        if value.shape not in ((rows,), (rows, 1)):
            raise ValueError(f"{name} must have shape [T] or [T,1]")
        return value.reshape(rows, 1).contiguous()

    ori = normalize(ori_lengths, "ori_lengths")
    if cmp_indices is None:
        if cmp_lengths is not None:
            raise ValueError("compressed lengths require compressed indices")
        return ori, torch.zeros_like(ori)
    if (
        cmp_indices.device != q.device
        or cmp_indices.dtype != torch.int32
        or cmp_indices.ndim != 3
        or cmp_indices.shape[:2] != (rows, 1)
    ):
        raise ValueError("compressed indices must be INT32 [T,1,K] on the query device")
    if cmp_lengths is None:
        cmp_lengths = (cmp_indices >= 0).sum(dim=-1, dtype=torch.int32)
    return ori, normalize(cmp_lengths, "cmp_lengths")
