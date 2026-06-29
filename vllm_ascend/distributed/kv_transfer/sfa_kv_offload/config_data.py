from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class RequestTracker:
    req_id: str
    allocated_block_ids_npu: list[int]
    allocated_block_ids_cpu: list[int]

    def update(
        self,
        new_block_ids_npu: list[int],
        new_block_ids_cpu: list[int],
    ) -> None:
        """Update the request tracker when a running request is scheduled again."""
        self.allocated_block_ids_npu.extend(new_block_ids_npu)
        self.allocated_block_ids_cpu.extend(new_block_ids_cpu)


@dataclass
class ReqMeta:
    req_id: str
    block_ids_npu: list[int]
    block_ids_cpu: list[int]
    num_new_offload_blocks: int = 0

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        num_new_offload_blocks: int = 0,
    ) -> ReqMeta | None:
        """Create the request metadata from a request tracker."""
        return ReqMeta(
            req_id=tracker.req_id,
            block_ids_npu=tracker.allocated_block_ids_npu,
            block_ids_cpu=tracker.allocated_block_ids_cpu,
            num_new_offload_blocks=num_new_offload_blocks,
        )


class SFAKVOffloadConnectorMetadata(KVConnectorMetadata):
    def __init__(
            self,
            unfinished_request_ids: set[str],
            preempted_req_ids: set[str] | None,
        ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)


@dataclass
class LayerMultiBlockReqMeta:
    req_id: str
    layer_id: int
    block_ids_npu: list[int] | None = None
    block_ids_cpu: list[int] | None = None
    cache_npu: tuple[torch.Tensor, torch.Tensor] | None = None
    cache_cpu: tuple[torch.Tensor, torch.Tensor] | None = None
