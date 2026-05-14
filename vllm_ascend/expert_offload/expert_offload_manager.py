"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import torch
from vllm.config import VllmConfig


class ExpertOffloadManager:
    """Singleton manager for expert weight offloading.

    Stores all expert weights on CPU and pages the needed experts to NPU
    during forward based on routing topk_ids.
    """

    _instance: "ExpertOffloadManager | None" = None

    @classmethod
    def get_instance(cls) -> "ExpertOffloadManager":
        assert cls._instance is not None, "ExpertOffloadManager not initialized"
        return cls._instance

    def __init__(self, vllm_config=None):
        from vllm_ascend.ascend_config import get_ascend_config

        self.offload_config = get_ascend_config().expert_offload_config
        self.num_device_experts = self.offload_config.num_device_experts

        # CPU weight buffers (post-transpose format, matching device after
        # process_weights_after_loading):
        #   w13 per expert: [hidden_size, w13_up_dim]
        #   w2 per expert:  [intermediate_size_per_partition, hidden_size]
        self.w13_weights_cpu: list[list[torch.Tensor]] = []
        self.w2_weights_cpu: list[list[torch.Tensor]] = []

        # Registered AscendFusedMoE layers, indexed by moe_instance_id order
        self.moe_layers: list = []

        # Temporary storage for weights loaded before create_weights()
        self._pending_weights: dict = {}

        ExpertOffloadManager._instance = self

    # ------------------------------------------------------------------ #
    #  Lifecycle: called from NPUModelRunner during model loading         #
    # ------------------------------------------------------------------ #

    def create_weights(
        self,
        num_moe_layers: int,
        num_total_experts: int,
        w13_up_dim: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
    ):
        """Allocate CPU buffers for all MoE layers."""
        for _ in range(num_moe_layers):
            w13_list = [
                torch.empty(hidden_size, w13_up_dim, dtype=params_dtype, device="cpu")
                for _ in range(num_total_experts)
            ]
            w2_list = [
                torch.empty(intermediate_size_per_partition, hidden_size,
                            dtype=params_dtype, device="cpu")
                for _ in range(num_total_experts)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)
        self._drain_pending_weights()

    def register_moe_layer(self, layer):
        self.moe_layers.append(layer)

    def load_w13(self, layer_moe_idx: int, expert_id: int,
                 loaded_weight: torch.Tensor, shard_id: str):
        """Store w1/w3 shard to CPU buffer (with transpose to post format)."""
        if not self.w13_weights_cpu:
            key = (layer_moe_idx, expert_id)
            self._pending_weights.setdefault(key, {})[f"w13_{shard_id}"] = \
                loaded_weight.cpu().clone()
            return
        cpu = self.w13_weights_cpu[layer_moe_idx][expert_id]
        intermed = cpu.shape[1] // 2
        w = loaded_weight.cpu()
        if shard_id == "w1":
            cpu[:, :intermed].copy_(w.t())
        elif shard_id == "w3":
            cpu[:, intermed: intermed + w.shape[0]].copy_(w.t())

    def load_w2(self, layer_moe_idx: int, expert_id: int,
                loaded_weight: torch.Tensor):
        """Store w2 weight to CPU buffer (with transpose to post format)."""
        if not self.w2_weights_cpu:
            key = (layer_moe_idx, expert_id)
            self._pending_weights.setdefault(key, {})["w2"] = \
                loaded_weight.cpu().clone()
            return
        self.w2_weights_cpu[layer_moe_idx][expert_id].copy_(loaded_weight.cpu().t())

    def init_device_experts(self):
        """Copy the first num_device_experts experts from CPU to NPU."""
        for i, layer in enumerate(self.moe_layers):
            dev = layer.w13_weight.device
            dt = layer.w13_weight.dtype
            for j in range(min(self.num_device_experts,
                               layer.w13_weight.shape[0])):
                layer.w13_weight.data[j].copy_(
                    self.w13_weights_cpu[i][j].to(dev).to(dt))
                layer.w2_weight.data[j].copy_(
                    self.w2_weights_cpu[i][j].to(dev).to(dt))

    # ------------------------------------------------------------------ #
    #  Forward path: page in experts based on topk_ids                    #
    # ------------------------------------------------------------------ #

    def update_weights(self, layer, topk_ids: torch.Tensor,
                        log2phy: torch.Tensor) -> int:
        """Incrementally page in needed experts, overwriting unused slots.

        Only copies experts that are NOT already on device.  Experts
        already mapped to a device slot (log2phy[eid] >= 0) are left
        untouched.  Reusable slots come from experts not in the current
        topk_ids set.

        Args:
            layer: AscendFusedMoE instance.
            topk_ids: [num_tokens, top_k] routed expert indices.
            log2phy: [global_num_experts] CPU tensor, modified in-place.

        Returns: number of CPU→NPU copies performed.
        """
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return 0

        unique_experts = topk_ids.unique().cpu().tolist()
        needed = set(unique_experts)

        # Build reverse map: slot → expert_id currently occupying it
        slot_owner: dict[int, int] = {}
        for eid in range(len(log2phy)):
            s = log2phy[eid].item()
            if s >= 0:
                slot_owner[s] = eid

        on_device = set(slot_owner.values())
        already_there = needed & on_device           # no-op
        need_to_load = needed - already_there          # CPU→NPU copy
        reusable_slots = [s for s, e in slot_owner.items()
                          if e not in needed]          # slots to recycle

        if not need_to_load:
            return 0

        dev = layer.w13_weight.device
        dt = layer.w13_weight.dtype
        n_copies = 0

        for eid in need_to_load:
            if not reusable_slots:
                break  # no free slots — should not happen in normal usage
            slot = reusable_slots.pop()
            # Copy from CPU to NPU
            layer.w13_weight.data[slot].copy_(
                self.w13_weights_cpu[layer_idx][eid].to(dev).to(dt))
            layer.w2_weight.data[slot].copy_(
                self.w2_weights_cpu[layer_idx][eid].to(dev).to(dt))
            # Update mapping
            log2phy[slot_owner[slot]] = -1   # evict old occupant
            log2phy[eid] = slot               # assign slot to new expert
            slot_owner[slot] = eid
            n_copies += 1

        return n_copies

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _drain_pending_weights(self):
        if not self._pending_weights:
            return
        for (layer_idx, eid), weights in self._pending_weights.items():
            if layer_idx >= len(self.w13_weights_cpu):
                continue
            if eid >= len(self.w13_weights_cpu[layer_idx]):
                continue
            cpu_w13 = self.w13_weights_cpu[layer_idx][eid]
            intermed = cpu_w13.shape[1] // 2
            for key, w in weights.items():
                w_cpu = w if w.device.type == "cpu" else w.cpu()
                if key.startswith("w13_"):
                    shard = key.split("_")[1]
                    if shard == "w1":
                        cpu_w13[:, :intermed].copy_(w_cpu.t())
                    elif shard == "w3":
                        cpu_w13[:, intermed: intermed + w_cpu.shape[0]].copy_(w_cpu.t())
                elif key == "w2":
                    self.w2_weights_cpu[layer_idx][eid].copy_(w_cpu.t())
        self._pending_weights.clear()


_EXPERT_OFFLOAD_MANAGER: ExpertOffloadManager = None


def maybe_init_expert_offload_manager(vllm_config: VllmConfig):
    # if no need to init offload manager:
    #     return
    global _EXPERT_OFFLOAD_MANAGER
    if _EXPERT_OFFLOAD_MANAGER is None:
        _EXPERT_OFFLOAD_MANAGER = ExpertOffloadManager(vllm_config)


def has_expert_offload_manager():
    return _EXPERT_OFFLOAD_MANAGER is not None


def get_expert_offload_manager():
    assert _EXPERT_OFFLOAD_MANAGER is not None, (
        "Expert Offload Manager is not initialized"
    )
    return _EXPERT_OFFLOAD_MANAGER