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

        # CPU buffers for quantized model scale/offset parameters.
        # Keyed by attr_name (e.g. "w13_weight_scale", "w2_weight_offset").
        # Each value is a list of layers, each layer is a list of expert tensors.
        self.scale_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.offset_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}

        # Temporary storage for scale/offset weights loaded before
        # maybe_create_scale_buffers runs.
        self._pending_scales: dict[tuple, dict[str, torch.Tensor]] = {}

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

    # ------------------------------------------------------------------ #
    #  Scale / offset helpers (quantized models only)                     #
    # ------------------------------------------------------------------ #

    def _add_pending_scale(self, layer_moe_idx: int, expert_id: int,
                           attr_name: str, shard_id: str,
                           loaded_weight: torch.Tensor):
        """Store a scale/offset weight that arrived before CPU buffers exist."""
        key = (layer_moe_idx, expert_id)
        sub_key = f"{attr_name}_{shard_id}"
        self._pending_scales.setdefault(key, {})[sub_key] = \
            loaded_weight.cpu().clone()

    def maybe_create_scale_buffers(self, layer, layer_moe_idx: int):
        """Inspect layer for scale/offset params and allocate CPU buffers.

        Called from _register_offload_layers AFTER process_weights_after_loading
        has transformed device tensor shapes, so we detect the final per-expert
        shape from the device tensor.
        """
        attr_names = [
            ("scale_cpu_buffers", "w13_weight_scale"),
            ("scale_cpu_buffers", "w2_weight_scale"),
            ("offset_cpu_buffers", "w13_weight_offset"),
            ("offset_cpu_buffers", "w2_weight_offset"),
        ]
        created_any = False
        global_num_experts = len(self.w13_weights_cpu[layer_moe_idx])

        for buffer_dict_name, attr_name in attr_names:
            if not hasattr(layer, attr_name):
                continue
            dev_tensor = getattr(layer, attr_name)
            per_expert_shape = dev_tensor.shape[1:]
            dtype = dev_tensor.dtype
            buffer_dict: dict = getattr(self, buffer_dict_name)
            if attr_name not in buffer_dict:
                buffer_dict[attr_name] = []
            buffers = buffer_dict[attr_name]
            while len(buffers) <= layer_moe_idx:
                buffers.append([])
            for _ in range(global_num_experts):
                buffers[layer_moe_idx].append(
                    torch.empty(per_expert_shape, dtype=dtype, device="cpu"))
            created_any = True

        if created_any:
            self._drain_pending_scales()

    def _drain_pending_scales(self):
        """Drain _pending_scales into CPU buffers, assembling w1/w3 shards.

        Only removes entries that were successfully copied to CPU buffers.
        Entries for layers whose buffers haven't been created yet are left
        in _pending_scales for the next call.
        """
        if not self._pending_scales:
            return
        processed_keys: list[tuple] = []
        for (layer_idx, eid), items in self._pending_scales.items():
            if layer_idx >= len(self.w13_weights_cpu):
                continue
            if eid >= len(self.w13_weights_cpu[layer_idx]):
                continue
            # Group shards by attr_name
            attr_shards: dict[str, dict[str, torch.Tensor]] = {}
            for sub_key, w in items.items():
                # sub_key format: "{attr_name}_{shard_id}"
                # attr_name may contain underscores (e.g. "w13_weight_scale")
                # shard_id is always "w1", "w2", or "w3" (no underscores)
                parts = sub_key.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in ("w1", "w2", "w3"):
                    attr_name, shard = parts[0], parts[1]
                else:
                    attr_name, shard = parts[0], parts[1] if len(parts) > 1 else ""
                attr_shards.setdefault(attr_name, {})[shard] = w

            copied_any = False
            for attr_name, shards in attr_shards.items():
                target_dict = None
                if "scale" in attr_name:
                    target_dict = self.scale_cpu_buffers
                elif "offset" in attr_name:
                    target_dict = self.offset_cpu_buffers
                if target_dict is None or attr_name not in target_dict:
                    continue
                buffers = target_dict[attr_name]
                if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                    continue
                target = buffers[layer_idx][eid]

                if attr_name.startswith("w13_"):
                    # w13 scale/offset: assemble w1 + w3 shards along dim 0
                    if "w1" in shards and "w3" in shards:
                        assembled = torch.cat(
                            [shards["w1"].cpu(), shards["w3"].cpu()], dim=0)
                        # squeeze trailing dim-1 if present (W8A8_DYNAMIC)
                        assembled = assembled.reshape(target.shape)
                        target.copy_(assembled)
                        copied_any = True
                elif attr_name.startswith("w2_"):
                    # w2 scale/offset: single shard
                    if "w2" in shards:
                        w_cpu = shards["w2"]
                        if w_cpu.device.type != "cpu":
                            w_cpu = w_cpu.cpu()
                        w_cpu = w_cpu.reshape(target.shape)
                        target.copy_(w_cpu)
                        copied_any = True
            if copied_any:
                processed_keys.append((layer_idx, eid))
        # Only remove successfully processed entries
        for key in processed_keys:
            del self._pending_scales[key]

    def init_device_experts(self):
        """Copy the first num_device_experts experts from CPU to NPU."""
        for i, layer in enumerate(self.moe_layers):
            dev = layer.w13_weight.device
            dt = layer.w13_weight.dtype
            ndev = min(self.num_device_experts, layer.w13_weight.shape[0])
            for j in range(ndev):
                layer.w13_weight.data[j].copy_(
                    self.w13_weights_cpu[i][j].to(dev).to(dt))
                layer.w2_weight.data[j].copy_(
                    self.w2_weights_cpu[i][j].to(dev).to(dt))

            # Copy initial scales/offsets for first N experts
            for attr_name, buffers in self.scale_cpu_buffers.items():
                if i >= len(buffers):
                    continue
                dev_tensor = getattr(layer, attr_name, None)
                if dev_tensor is None:
                    continue
                for j in range(min(ndev, len(buffers[i]))):
                    dev_tensor.data[j].copy_(buffers[i][j].to(dev))

            for attr_name, buffers in self.offset_cpu_buffers.items():
                if i >= len(buffers):
                    continue
                dev_tensor = getattr(layer, attr_name, None)
                if dev_tensor is None:
                    continue
                for j in range(min(ndev, len(buffers[i]))):
                    dev_tensor.data[j].copy_(buffers[i][j].to(dev))

            # Refresh derived fp32 scale if present (W8A8_DYNAMIC)
            if hasattr(layer, 'w13_weight_scale_fp32'):
                for j in range(ndev):
                    layer.w13_weight_scale_fp32[j].copy_(
                        layer.w13_weight_scale.data[j].to(torch.float32))

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

        # Debug: print routing info for first 30 calls
        if not hasattr(self, '_dbg_call'):
            self._dbg_call = 0
        if self._dbg_call < 10000:
            import logging
            _dbg = logging.getLogger(__name__)
            _dbg.warning("[UPDATE-W] l=%d call=%d topk_shape=%s |needed|=%d |on_dev|=%d |to_load|=%d reusable=%d needed=%s",
                         layer_idx, self._dbg_call, tuple(topk_ids.shape),
                         len(needed), len(on_device),
                         len(need_to_load), len(reusable_slots),
                         sorted(needed)[:30])
            if need_to_load and len(need_to_load) > len(reusable_slots):
                _dbg.warning("[UPDATE-W] l=%d SHORTFALL: need %d load but only %d slots, to_load=%s",
                             layer_idx, len(need_to_load), len(reusable_slots),
                             sorted(need_to_load)[:20])
            self._dbg_call += 1

        if not need_to_load:
            return 0

        dev = layer.w13_weight.device
        dt = layer.w13_weight.dtype
        n_copies = 0

        for eid in need_to_load:
            if not reusable_slots:
                import logging
                logging.getLogger(__name__).warning(
                    "[UPDATE-W] l=%d NO SLOTS: %d experts could not be loaded, missed=%s",
                    layer_idx, len(need_to_load) - n_copies,
                    sorted(list(need_to_load))[n_copies:][:20])
                break  # no free slots — should not happen in normal usage
            slot = reusable_slots.pop()
            # Copy weights from CPU to NPU
            layer.w13_weight.data[slot].copy_(
                self.w13_weights_cpu[layer_idx][eid].to(dev).to(dt))
            layer.w2_weight.data[slot].copy_(
                self.w2_weights_cpu[layer_idx][eid].to(dev).to(dt))
            # Copy scales/offsets from CPU to NPU
            for attr_name, buffers in self.scale_cpu_buffers.items():
                if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                    continue
                dev_tensor = getattr(layer, attr_name, None)
                if dev_tensor is None:
                    continue
                dev_tensor.data[slot].copy_(buffers[layer_idx][eid].to(dev))
            for attr_name, buffers in self.offset_cpu_buffers.items():
                if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                    continue
                dev_tensor = getattr(layer, attr_name, None)
                if dev_tensor is None:
                    continue
                dev_tensor.data[slot].copy_(buffers[layer_idx][eid].to(dev))
            # Refresh derived fp32 scale if present (W8A8_DYNAMIC)
            if hasattr(layer, 'w13_weight_scale_fp32'):
                layer.w13_weight_scale_fp32[slot].copy_(
                    layer.w13_weight_scale.data[slot].to(torch.float32))
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
