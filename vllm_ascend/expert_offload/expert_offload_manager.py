import torch

from vllm.config import VllmConfig


class ExpertOffloadManager():
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        # cpu weights buffer, size = num_layers, element tensor size = [num_experts, dim_in, dim_out]
        self.w13_weights_cpu: list[torch.Tensor] = []
        self.w2_weights_cpu: list[torch.Tensor] = []

    def create_weights(self):
        # 根据 config 初始化 host 侧权重 self.w13/w2_weights_cpu, __init__ 初始化时调用
        pass

    def load_weights(self):
        # 模型权重加载到 self.w13/w2_weights_cpu，加载权重时由 moe weight_loader 调用
        pass

    def update_weights(self, topk_ids: torch.Tensor):
        # 模型前向时根据所需专家列表 topk_ids 更新 npu weight buffer
        # 待补充入参：layer_id, npu weight buffer，开发过程中根据情况补充
        pass


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
