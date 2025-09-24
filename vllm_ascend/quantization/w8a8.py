#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

from typing import Any, Dict, Optional

import torch
import torch_npu
from vllm_ascend import envs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def quant_per_tensor(in_tensor: torch.Tensor, input_scale: torch.Tensor,
                     input_offset: torch.Tensor):
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, False)


class AscendW8A8LinearMethod:
    """Linear method for Ascend W8A8.

    Args:
        w_sym: whether the linear weight is symmetrically quantized.
    """

    def __init__(self) -> None:
        # aclnn quant matmul requires to transpose matrix B, set to true by default.
        self.transpose_weight = True
        ascend_config = get_ascend_config()
        self.enable_weight_nz_layout = ascend_config.enable_weight_nz_layout

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        return params_dict

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            x = quant_per_tensor(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
            )
        quant_bias = layer.quant_bias if tp_rank == 0 else None
        return torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=layer.params_dtype,
        )

    def process_weights_after_loading(self, layer):
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False).to(layer.aclnn_input_scale.dtype)
        # Modified begin
        module_name = layer.prefix.split(".")[-1] if hasattr(layer, 'prefix') else ""
        if envs.VLLM_ASCEND_ROPE_OPT and module_name == "q_b_proj":
            temp_weight = layer.weight.data
            temp_weight = temp_weight.view(-1, 192, temp_weight.shape[-1])
            weight_1 = temp_weight[..., -64:: 2, :].contiguous()
            weight_2 = temp_weight[..., -64 + 1:: 2, :].contiguous()
            temp_weight[..., -64:, :] = torch.cat([weight_1, weight_2], dim=-2)
            layer.weight.data = temp_weight.view(layer.weight.shape) 
            deq_scale = layer.deq_scale.data 
            weight_scale = deq_scale.view(-1, 192, 1)
            weight_1 = weight_scale[..., -64:: 2, :].contiguous()
            weight_2 = weight_scale[..., -64 + 1:: 2, :].contiguous()
            weight_scale[..., -64:, :] = torch.cat([weight_1, weight_2], dim=-2)
            layer.deq_scale.data  = weight_scale.view(layer.deq_scale.shape).flatten()

            weight_offset = layer.quant_bias.data
            weight_offset = weight_offset.view(-1, 192, 1)
            weight_1 = weight_offset[..., -64:: 2, :].contiguous()
            weight_2 = weight_offset[..., -64 + 1:: 2, :].contiguous()
            weight_offset[..., -64:, :] = torch.cat([weight_1, weight_2], dim=-2)
            layer.quant_bias.data = weight_offset.view(layer.quant_bias.shape).flatten()
        
        # Modified
        if envs.VLLM_ASCEND_ROPE_OPT and module_name == "kv_a_proj_with_mqa":
            temp_weight = layer.weight.data
            temp_weight = temp_weight.view(-1, temp_weight.shape[-1])
            weight_1 = temp_weight[-64:: 2, :].contiguous()
            weight_2 = temp_weight[-64 + 1:: 2, :].contiguous()
            temp_weight[-64:, :] = torch.cat([weight_1, weight_2], dim=0)
            layer.weight.data = temp_weight.view(layer.weight.shape)
            deq_scale = layer.deq_scale.data 
            weight_scale = deq_scale.view(-1, 1)
            weight_1 = weight_scale[-64:: 2, :].contiguous()
            weight_2 = weight_scale[-64 + 1:: 2, :].contiguous()
            weight_scale[-64:, :] = torch.cat([weight_1, weight_2], dim=0)
            layer.deq_scale.data  = weight_scale.view(layer.deq_scale.shape).flatten()

            weight_offset = layer.quant_bias.data
            weight_offset = weight_offset.view(-1, 1)
            weight_1 = weight_offset[-64:: 2, :].contiguous()
            weight_2 = weight_offset[-64 + 1:: 2, :].contiguous()
            weight_offset[-64:, :] = torch.cat([weight_1, weight_2], dim=0)
            layer.quant_bias.data = weight_offset.view(layer.quant_bias.shape).flatten()

        # Modified， 除了o_proj和q_b_proj外其余的权重要转为NZ 
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        if self.enable_weight_nz_layout:
            # cast quantized weight tensors in NZ layout for higher inference speed
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)

        torch.npu.empty_cache()
