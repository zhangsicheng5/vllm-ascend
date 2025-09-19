"""
Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
This file is a part of the vllm-ascend project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter, UninitializedParameter
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.forward_context import get_forward_context
from vllm.distributed import divide, split_tensor_along_last_dim, tensor_model_parallel_reduce_scatter, \
    tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (get_tp_group, get_dp_group, get_tensor_model_parallel_world_size,
                                             get_tensor_model_parallel_rank)
from vllm.lora.utils import LinearBase
from vllm.model_executor.layers.linear import (WEIGHT_LOADER_V2_SUPPORTED,
                                               QuantizeMethodBase,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.distributed.parallel_state import get_otp_group
from vllm_ascend.utils import oproj_tp_enable
from vllm_ascend.ascend_config import get_ascend_config

_HCOMM_INFO = None

class AscendRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.
    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        enable_sp: bool = False,
        otp_compatible: bool = True,
    ):
        self.prefix = prefix

        if prefix.find("o_proj") != -1 and oproj_tp_enable() and otp_compatible:
            comm_group = get_otp_group()
            self.forward_type = "oproj_tp"
        else:
            comm_group = get_tp_group()
            self.forward_type = "normal"
        self.comm_group = comm_group

        self.tp_size = self.comm_group.world_size
        self.tp_rank = self.comm_group.rank_in_group
        self.dp_rank = get_dp_group().rank_in_group

        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(self,
                                  input_size,
                                  output_size,
                                  skip_bias_add,
                                  params_dtype,
                                  quant_config,
                                  prefix,
                                  return_bias=return_bias)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

        self.torchair_graph_enabled = get_ascend_config().torchair_graph_config.enabled
        self.enable_sp = enable_sp
        self.rank = torch.distributed.get_rank()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = self.tp_rank
        tp_size = self.tp_size
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    @staticmethod
    def get_hcomm_info(group: ProcessGroup) -> str:
        """Get the HCCL communication information for the given group."""
        global _HCOMM_INFO
        if _HCOMM_INFO is not None:
            return _HCOMM_INFO

        rank = torch.distributed.get_rank(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            _HCOMM_INFO = group._get_backend(
                torch.device("npu")).get_hccl_comm_name(global_rank)
        else:
            _HCOMM_INFO = group.get_hccl_comm_name(rank)
        return _HCOMM_INFO

    def forward(
        self,
        input_,
        is_prefill: bool = True,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        # Choose different forward function according to the type of TP group
        if self.forward_type == "oproj_tp":
            return self._forward_oproj_tp(input_)
        sp_size = get_tensor_model_parallel_world_size()
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.enable_sp and is_prefill:
            original_len = input_.shape[0]
            reminder = original_len % sp_size
            if reminder != 0:
                padding_len = sp_size - reminder
                output_parallel = F.pad(output_parallel, (0, 0, 0, padding_len), mode='constant', value=0)
            output = tensor_model_parallel_reduce_scatter(output_parallel.movedim(0, -1)).movedim(-1, 0)
        elif self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    # enable custom Oproj tensor parallel
    def _forward_oproj_tp(
        self,
        input_: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:

        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Prepare tensors for all-to-all communication
        local_batch_size = input_parallel.size(0)
        chunk_size = self.input_size_per_partition
        total_batch_size = local_batch_size * self.tp_size

        enable_graph_mode = self.torchair_graph_enabled
        forward_context = get_forward_context()

        if enable_graph_mode:
            # Reshape tensor for efficient cross-device transfer:
            # [batch, dim] -> [tp_size, batch, chunk] -> flattened
            send_buf = (input_parallel.reshape(-1,
                                            self.tp_size, chunk_size).transpose(
                                                0, 1).contiguous().view(-1))

            # Create receive buffer
            recv_buf = torch.empty(total_batch_size * chunk_size,
                                dtype=input_parallel.dtype,
                                device=input_parallel.device)

            # Perform all-to-all communication
            dist.all_to_all_single(recv_buf,
                                send_buf,
                                group=self.comm_group.device_group)
            input_parallel = recv_buf.view(total_batch_size, chunk_size)


        else:
            cu_tokens_across_dp_cpu = forward_context.dp_metadata.cu_tokens_across_dp_cpu
            prefix_array = cu_tokens_across_dp_cpu.cpu().numpy()
            global_batch_size = np.concatenate(
                ([prefix_array[0]], np.diff(prefix_array)))
            tp_group_id = self.dp_rank // self.tp_size
            tp_group_batchsize = global_batch_size[tp_group_id * self.tp_size: tp_group_id * self.tp_size + self.tp_size]
            tp_total_batchsize = sum(tp_group_batchsize)

            # Reshape for all-to-all communication
            send_buf = (
                input_parallel.reshape(-1, self.tp_size, chunk_size)
                .transpose(0, 1)
                .contiguous()
                .view(-1))

            # Create receive buffer
            recv_buf = torch.empty(
                tp_total_batchsize * chunk_size,
                dtype=input_parallel.dtype,
                device=input_parallel.device)

            # Create split array
            recv_splits = [size * chunk_size for size in tp_group_batchsize]
            send_splits = [local_batch_size * chunk_size] * self.tp_size

            # Perform all-to-all communication
            dist.all_to_all_single(
                recv_buf,
                send_buf,
                recv_splits,
                send_splits,
                group=self.comm_group.device_group)

            input_parallel = recv_buf.view(tp_total_batchsize, chunk_size)


        # Only fuse bias add for rank 0 to avoid duplicate bias addition in TP>1
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if enable_graph_mode:
            # otp-specific: Combine partial results across devices
            output = self.comm_group.reduce_scatter(output_parallel, dim=0)

        else:
            # prepare all-reduce data
            output = torch.empty(
                    local_batch_size,
                    output_parallel.size(1),
                    dtype=output_parallel.dtype,
                    device=output_parallel.device)

            recv_chunks = []
            start_idx = 0
            for size in tp_group_batchsize:
                chunk = output_parallel[start_idx:start_idx + size, :]
                recv_chunks.append(chunk.contiguous())
                start_idx += size

            # Reduce-scatter the results across devices
            dist.reduce_scatter(
                    output,
                    recv_chunks,
                    op=dist.ReduceOp.SUM,
                    group=self.comm_group.device_group)

        # Handle bias return based on configuration
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias


class AscendLinearBase(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        nn.Module.__init__(self)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        if quant_config is None:
            self.quant_method: Optional[
                QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp