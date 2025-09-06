#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass, fields
from typing import Type, Union

from vllm.config import SchedulerConfig


@dataclass
class AscendSchedulerConfig(SchedulerConfig):
    enable_chunked_prefill: bool = False
    policy: str = "fcfs"
    num_scheduler_steps: int = 1
    scheduler_cls: Union[str, Type[object]] = (
        "vllm_ascend.core.scheduler.AscendScheduler")

    @classmethod
    def initialize_from_config(
        cls,
        vllm_scheduler_config: SchedulerConfig,
        ascend_scheduler_config,
    ):
        scheduler_config = {
            field.name: getattr(vllm_scheduler_config, field.name)
            for field in fields(vllm_scheduler_config) if field.init
        }
        # Override default values into original SchedulerConfig
        scheduler_config["enable_chunked_prefill"] = False
        scheduler_config["policy"] = "fcfs"
        scheduler_config["num_scheduler_steps"] = 1
        scheduler_config["scheduler_cls"] = (
            "vllm_ascend.core.scheduler.AscendScheduler")
        # Override params in original SchedulerConfig with params in ascend_scheduler_config
        for k, _ in scheduler_config.items():
            if hasattr(ascend_scheduler_config, k):
                scheduler_config[k] = getattr(ascend_scheduler_config, k)
        return cls(**scheduler_config)

    def __post_init__(self) -> None:
        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens
        self.chunked_prefill_enabled = self.enable_chunked_prefill
        # with prefill cp & sp, max_num_batched_tokens should be less than max_model_len
        # if (self.max_num_batched_tokens < self.max_model_len
        #         and not self.chunked_prefill_enabled):
        #     raise ValueError(
        #         "Ascend scheduler is enabled without chunked prefill feature. "
        #         f"Argument max_num_batched_tokens ({self.max_num_batched_tokens}) is "
        #         f"smaller than max_model_len ({self.max_model_len}). "
        #         "This effectively limits the maximum sequence length to "
        #         "max_num_batched_tokens and makes vLLM reject longer "
        #         "sequences. Please increase max_num_batched_tokens or "
        #         "decrease max_model_len.")
        if self.policy != "fcfs":
            raise NotImplementedError(
                f"currently AscendScheduler only supports fcfs policy, got {self.policy}"
            )
        if self.is_multimodal_model:
            raise NotImplementedError(
                "currently AscendScheduler only supports LLM models.")
        if self.num_scheduler_steps > 1:
            raise NotImplementedError(
                "currently AscendScheduler doesn't support multi-step.")
        if self.send_delta_data:
            raise NotImplementedError(
                "currently AscendScheduler doesn't support send_delta_data.")
        if self.delay_factor > 0:
            raise NotImplementedError(
                "currently AscendScheduler doesn't support scheduler_delay_factor."
            )
