
import vllm
from vllm_ascend.ops.linear import AscendRowParallelLinear

vllm.model_executor.layers.linear.RowParallelLinear = AscendRowParallelLinear