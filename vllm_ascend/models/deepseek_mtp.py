#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/deepseek_mtp.py
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter,
                              get_context_model_parallel_world_size,
                              get_cp_group,
                              get_cp_group,
                              divide)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP, DeepSeekMultiTokenPredictor, DeepSeekMultiTokenPredictorLayer,
    SharedHead)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.lmhead import CustomParallelLMHead
from vllm_ascend.ops.logits_processor import CustomLogitsProcessor

from .deepseek_v2 import CustomDeepseekV2DecoderLayer, VocabParallelEmbeddingwithSP
from collections.abc import Sequence


class CustomDeepSeekShareHead(SharedHead):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = CustomParallelLMHead(config.vocab_size,
                                         config.hidden_size,
                                         quant_config=quant_config,
                                         prefix=maybe_prefix(prefix, "head"))


def split_tensor_along_first_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """ Split a tensor along its first dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    first_dim = 0
    first_dim_size = divide(tensor.size()[first_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, first_dim_size, dim=first_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class CustomDeepSeekMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        enable_sp: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.embed_tokens = VocabParallelEmbeddingwithSP(
            config.vocab_size,
            config.hidden_size,
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2,
                                 config.hidden_size,
                                 bias=False)
        self.shared_head = CustomDeepSeekShareHead(config=config,
                                                   quant_config=quant_config,
                                                   prefix=maybe_prefix(
                                                       prefix, "shared_head"))
        self.enable_sp = enable_sp
        self.mtp_block = CustomDeepseekV2DecoderLayer(config, prefix,
                                                      model_config,
                                                      cache_config,
                                                      quant_config,
                                                      enable_sp = self.enable_sp,
                                                      otp_compatible=False)

    def forward(
        self,
        original_len: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_index: int = 0,
        is_prefill: bool = False,
        positions_splitted: torch.tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids, enable_sp=(self.enable_sp and is_prefill))
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        forward_context = get_forward_context()
        if positions_splitted is None:
            positions_splitted = positions
        if forward_context.with_prefill:
            inputs_embeds = torch.where((positions_splitted == 0).unsqueeze(-1),
                                        torch.zeros_like(inputs_embeds),
                                        inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states, residual = self.mtp_block(original_len=original_len,
                                                 positions=positions,
                                                 hidden_states=hidden_states,
                                                 kv_cache=kv_cache,
                                                 attn_metadata=attn_metadata,
                                                 residual=None)
        hidden_states = residual + hidden_states
        return hidden_states


class CustomDeepSeekMultiTokenPredictor(DeepSeekMultiTokenPredictor):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.enable_sp = vllm_config.parallel_config.enable_sequence_parallel
        self.cp_size = get_context_model_parallel_world_size()
        self.cp_group = get_cp_group().device_group
        self.sp_size = get_tensor_model_parallel_world_size()
        self.sp_group = get_tp_group().device_group
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict({
            str(idx): CustomDeepSeekMultiTokenPredictorLayer(
                config,
                f"{prefix}.layers.{idx}",
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                enable_sp=self.enable_sp,
            )
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })

        # Note: torch._dynamo.exc.Unsupported: builtin: str
        self.layers_list = [
            self.layers[str(idx)]
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        ]
        self.logits_processor = CustomLogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: torch.Tensor,
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        original_len = 1
        is_prefill = False
        # NOTE influence graph mode, need to check why
        if self.enable_sp or self.cp_size > 1:
            attn_metadata = get_forward_context().attn_metadata if attn_metadata is None else attn_metadata
            if attn_metadata:
                is_prefill = attn_metadata.num_prefills > 0 # TODO checkout is_prefill while mtp

        if inputs_embeds is None:
            original_len = input_ids.shape[0]

        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        step_kv_cache = kv_caches[
            current_step_idx] if kv_caches is not None else None

        positions_splitted = positions
        if self.enable_sp and is_prefill:  # previous_hidden_states should be splited as well
            reminder = original_len % self.sp_size
            if reminder > 0:
                padding_len = self.sp_size - reminder
                previous_hidden_states = F.pad(previous_hidden_states, (0, 0, 0, padding_len), mode='constant', value=0)
                positions_splitted = F.pad(positions_splitted, (0, padding_len), mode='constant', value=0)
            sp_rank = get_tensor_model_parallel_rank()
            previous_hidden_states = split_tensor_along_first_dim(
                previous_hidden_states, num_partitions=self.sp_size)[sp_rank].contiguous()
            positions_splitted = split_tensor_along_first_dim(
                positions_splitted, num_partitions=self.sp_size)[sp_rank].contiguous()
        hidden_states = self.layers_list[current_step_idx](
            original_len,
            input_ids,
            positions,
            step_kv_cache,
            attn_metadata,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
            is_prefill=is_prefill,
            positions_splitted=positions_splitted,
        )
        if self.enable_sp and is_prefill:
            chunk_hidden_states = [torch.empty_like(hidden_states) for _ in range(self.sp_size)]
            dist.all_gather(list(chunk_hidden_states), hidden_states, self.sp_group)
            hidden_states = torch.cat(chunk_hidden_states, dim=0)
            hidden_states = hidden_states[:original_len]
        if self.cp_size > 1 and is_prefill:
            chunk_hidden_states = [torch.empty_like(hidden_states) for _ in range(self.cp_size)]
            dist.all_gather(list(chunk_hidden_states), hidden_states, self.cp_group)
            hidden_states = torch.cat(chunk_hidden_states, dim=0)
            hidden_states = torch.index_select(hidden_states, 0, attn_metadata.prefill.cp_metadata.cp_kv_recover_idx)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        mtp_layer = self.layers_list[current_step_idx]
        logits = self.logits_processor(mtp_layer.shared_head.head,
                                       mtp_layer.shared_head(hidden_states),
                                       sampling_metadata)
        return logits


class CustomDeepSeekMTP(DeepSeekMTP):
    # NOTE 1.The quantized MTP layer of deepseek on the NPU is not quantized;
    # NOTE 2.The description file generated by the current msmodelslim tool does not have
    # MTP layer info. Please manually add it and set the value to FLOAT.
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.model = CustomDeepSeekMultiTokenPredictor(vllm_config=vllm_config,
                                                       prefix=maybe_prefix(
                                                           prefix, "model"))

        self.sampler = get_sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, previous_hidden_states,
                                   inputs_embeds, spec_step_idx)
        return hidden_states
