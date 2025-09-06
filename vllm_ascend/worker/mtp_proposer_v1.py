import types

import numpy as np
import torch
import torch.nn as nn
import torchair
import vllm.envs as envs_vllm
from torchair import patch_for_hcom
from vllm.attention.layer import Attention
from vllm.config import (VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.forward_context import get_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import (
    process_weights_after_loading, set_default_torch_dtype)
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, AscendCommonLongSequenceMetadata
from vllm_ascend.distributed.utils import is_lmhead_tp
from vllm_ascend.models.deepseek_mtp import CustomDeepSeekMTP
from vllm_ascend.utils import TORCHAIR_CACHE_DIR, ProfileExecuteDuration


# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    is_greedy = sampling_metadata.temperature == -1
    temperature = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    next_token_ids = probs.div_(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(
            is_greedy,
            greedy_token_ids,
            next_token_ids,
        )
    return next_token_ids, probs


class MtpProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        runner,
    ):
        self.vllm_config = vllm_config
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens)
        self.block_size = vllm_config.cache_config.block_size
        self.runner = runner
        self.sp_size = self.runner.sp_size
        self.cp_size = self.runner.cp_size
        self.cp_rank = self.runner.cp_rank
        # persistent buffers for graph
        self.input_ids = torch.zeros(self.runner.max_num_tokens * self.cp_size * self.sp_size,
                                     dtype=torch.int32,
                                     device=self.runner.device)
        self.positions = torch.zeros(self.runner.max_num_tokens * self.sp_size,
                                     dtype=torch.int64,
                                     device=self.runner.device)
        self.hidden_states = torch.zeros(
            (self.runner.max_num_tokens * self.sp_size, self.runner.hidden_size),
            dtype=self.runner.dtype,
            device=self.runner.device)
        self.torchair_compiled_model = None  # type: ignore
        self.torchair_compiled_models = {}  # type: ignore

    @staticmethod
    def prepare_inputs(
            # [batch_size + 1]
            cu_target_query_lens: torch.Tensor,
            # [batch_size]
            num_rejected_tokens: torch.Tensor,
            is_torchair_graph: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cu_target_query_lens: [0, a, a + b, a + b + c]
        # num_rejected_tokens: [n1, n2, n3]
        # num_tokens_per_req: [a - n1, b - n2, c - n3]
        # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        # token_indices: [0, 1, ..., a - n1 - 1,
        #                 a, a + 1, ..., a + b - n2 - 1,
        #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]
        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = (cu_target_query_lens[1:] -
                             cu_target_query_lens[:-1])
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens
        if is_torchair_graph:
            cu_num_tokens = cu_target_query_lens
            relative_index = query_len_per_req - num_rejected_tokens - 1
            token_indices = cu_num_tokens[:-1] + relative_index
        else:
            cu_num_tokens = torch.empty_like(cu_target_query_lens)
            torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
            cu_num_tokens[0] = 0

            # FIXME(woosuk): Avoid synchronization.
            num_tokens = cu_num_tokens[-1].item()
            token_indices = torch.zeros(
                num_tokens,
                dtype=torch.int32,
                device=cu_num_tokens.device,
            )

            BLOCK_SIZE = 1024
            prepare_input_kernel(
                token_indices,
                cu_target_query_lens,
                cu_num_tokens,
                block_size=BLOCK_SIZE,
            )
        return cu_num_tokens, token_indices

    def propose(
            self,
            # [num_tokens]
            target_token_ids: torch.Tensor,
            # [num_tokens]
            target_positions: torch.Tensor,
            # [num_tokens, hidden_size]
            target_hidden_states: torch.Tensor,
            # [num_tokens]
            target_slot_mapping: torch.Tensor,
            # [batch_size]
            next_token_ids: torch.Tensor,
            # [batch_size + 1] starting with 0
            cu_num_tokens: torch.Tensor,
            # [batch_size, max_num_blocks_per_req]
            block_table: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            token_indices=None,
            req_scheduled_tokens: dict=None,
            long_seq_metadata: AscendCommonLongSequenceMetadata=None,
            is_prefill: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        if token_indices is not None and self.runner.torchair_graph_enabled:
            last_token_indices = token_indices

        self.input_ids[last_token_indices] = next_token_ids

        query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
        max_query_len = query_lens.max().item()

        # FIXME: reorder_batch() needs to be called before build()
        # because fields of attn_metadata_builder needs to be updated.
        # However, currently reorder_batch() takes input_batch and
        # scheduler_output as arguments, we should probably refactor
        # the method to use new data structures which are independent
        # from input_batch and scheduler_output.
        # self.runner.attn_metadata_builder.reorder_batch(
        #     input_batch=self.runner.input_batch,
        #     scheduler_output=self.runner.scheduler_output,
        # )
        extra_builder_kwargs = self.runner.extra_builder_kwargs

        is_running_torchair = self.runner.torchair_graph_enabled and \
            not self.runner.with_prefill

        if is_running_torchair:
            extra_builder_kwargs[
                'num_reqs_pad_size'] = self.runner.num_reqs_pad_size
            extra_builder_kwargs[
                'num_token_pad_size'] = self.runner.num_token_pad_size
            num_input_tokens = num_tokens + self.runner.num_token_pad_size
        else:
            extra_builder_kwargs['num_token_pad_size'] = -1
            extra_builder_kwargs['num_reqs_pad_size'] = 0
            num_input_tokens = num_tokens

        seq_lens = target_positions[last_token_indices] + 1
        seq_lens = seq_lens.int()

        if self.cp_size > 1 and is_prefill:
            # target_token_ids: ori full input_ids without pad
            # target_positions: cp pad and split position
            # target_hidden_states: cp pad, split, all-gather full hidden states
            # target_slot_mapping: cp pad full slot_mapping
            # cu_num_tokens: ori full cum_sum without pad
            num_cp_scheduled_tokens = []
            input_ids_list = self.input_ids[:num_tokens].tolist()
            ori_start_index = 0
            pad_start_index = 0
            cp_split_input_ids_list = np.array([], dtype=np.int32)
            cp_split_hidden_states_list = []
            for i, req_id in enumerate(req_scheduled_tokens):
                ori_num_tokens = req_scheduled_tokens[req_id]
                req_position_cp, num_cp_padded_scheduled_tokens, num_cp_pad = self.runner._num_scheduled_tokens_prefill_cp(
                    ori_num_tokens, 0, set_cp_kv_recover_idx=False) # TODO consider computed tokens in prefill
                actual_num_tokens = len(req_position_cp)
                num_cp_scheduled_tokens.append(actual_num_tokens)
                pad_input_ids = np.array(input_ids_list[ori_start_index : ori_start_index + ori_num_tokens] + [0] * num_cp_pad, dtype=np.int32)
                ori_start_index += ori_num_tokens
                cp_chunk_indices = [pad_start_index + pos for pos in req_position_cp]
                cp_split_input_ids = pad_input_ids[req_position_cp]
                cp_split_hidden_states = target_hidden_states[cp_chunk_indices]
                cp_split_input_ids_list = np.append(cp_split_input_ids_list, cp_split_input_ids)
                cp_split_hidden_states_list.append(cp_split_hidden_states)
                pad_start_index += num_cp_padded_scheduled_tokens
            num_tokens = sum(num_cp_scheduled_tokens)
            num_input_tokens = num_tokens # TODO consider graph pad
            self.input_ids[:num_input_tokens].copy_(torch.tensor(cp_split_input_ids_list, dtype=torch.int32))
            target_hidden_states = torch.cat(cp_split_hidden_states_list, dim=0)
            max_query_len = max(num_cp_scheduled_tokens)
            seq_lens = torch.tensor(num_cp_scheduled_tokens, dtype=torch.int32)
            cu_num_tokens = torch.tensor(np.insert(np.cumsum(np.array(num_cp_scheduled_tokens)), 0, 0))

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=cu_num_tokens[:batch_size + 1],
            query_start_loc_cpu=cu_num_tokens[:batch_size + 1].cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=self.runner.input_batch.block_table[0].
            get_device_tensor(),
            slot_mapping_cpu=target_slot_mapping,
            positions=target_positions,
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            max_num_blocks_per_req=self.runner.max_num_blocks_per_req,
            common_long_seq_metadata=long_seq_metadata,
        )
        attn_metadata = self.runner.attn_metadata_builder.build(
            common_attn_metadata, **extra_builder_kwargs)

        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states

        if not self.runner.torchair_graph_enabled:
            # TODO: adapt enable_dbo later
            (num_input_tokens, num_tokens_across_dp, with_prefill,
             _) = self.runner._get_forward_metadata_across_dp(
                 num_input_tokens, num_tokens, self.runner.with_prefill, False)
            attn_metadata.slot_mapping = target_slot_mapping
        else:
            num_tokens_across_dp = self.runner.num_tokens_across_dp
            with_prefill = self.runner.with_prefill

        with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                with_prefill=with_prefill,
                num_tokens_across_dp=num_tokens_across_dp,
                in_profile_run=self.runner.in_profile_run,
                num_actual_tokens=num_tokens):
            with ProfileExecuteDuration().capture_async('mtp_forward'):
                model_kwargs = {}
                model_kwargs["attn_metadata"] = attn_metadata
                if self.runner.torchair_graph_enabled:
                    model_kwargs["kv_caches"] = self.runner.kv_caches[-1:]
                if is_running_torchair:
                    torchair_compiled_model = self._get_torchair_lazy_compiled_model(
                        num_input_tokens)
                    hidden_states = torchair_compiled_model(
                        input_ids=self.input_ids[:num_input_tokens],
                        positions=self.positions[:num_input_tokens],
                        previous_hidden_states=self.
                        hidden_states[:num_input_tokens],
                        inputs_embeds=None,
                        intermediate_tensors=None,
                        spec_step_idx=0,
                        **model_kwargs)
                else:
                    hidden_states = self.model(
                        input_ids=self.input_ids[:num_input_tokens],
                        positions=self.positions[:num_input_tokens],
                        previous_hidden_states=self.
                        hidden_states[:num_input_tokens],
                        kv_caches=self.runner.kv_caches[-1:])

        num_indices = last_token_indices.shape[0]
        if is_lmhead_tp():
            if not self.runner.with_prefill:
                padded_num_indices = num_input_tokens // self.runner.decode_token_per_req
            else:
                padded_num_indices = self.vllm_config.scheduler_config.max_num_seqs
            last_token_indices = nn.functional.pad(
                last_token_indices, (0, padded_num_indices - num_indices))

        sample_hidden_states = hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)

        if is_lmhead_tp() and num_indices < logits.shape[0]:
            logits = logits[:num_indices]

        draft_token_ids = logits.argmax(dim=-1)

        # [batch_size, 1]
        return draft_token_ids.view(-1, 1)

    def load_model(self) -> None:
        loader = get_model_loader(self.vllm_config.load_config)

        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_device = self.vllm_config.device_config.device

        with set_default_torch_dtype(
                draft_model_config.dtype), set_current_vllm_config(
                    self.vllm_config):
            self.model = CustomDeepSeekMTP(
                vllm_config=self.vllm_config).to(target_device)

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)

        assert len(draft_attn_layer_names) == 1
        self.attn_layer_name = next(iter(draft_attn_layer_names))

        self.model.load_weights(
            loader.get_all_weights(
                self.vllm_config.speculative_config.draft_model_config,
                self.model))
        process_weights_after_loading(self.model, draft_model_config,
                                      target_device)

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens: int,
                  with_prefill: bool = False,
                  skip_attn: bool = False,
                  num_reqs: int = 0,
                  num_tokens_across_dp=None) -> None:
        if not self.runner.torchair_graph_enabled:
            # TODO: adapt enable_dbo later
            (num_tokens, num_tokens_across_dp, with_prefill,
             _) = self.runner._get_forward_metadata_across_dp(
                 num_tokens, num_tokens, with_prefill, False)
        is_running_torchair = self.runner.torchair_graph_enabled and \
            not with_prefill

        if is_running_torchair:
            skip_attn = False
        if skip_attn:
            attn_metadata = None
        else:
            attn_metadata = self.runner.attn_metadata_builder.build_torchair_graph_dummy(
                num_reqs=num_reqs, num_actual_tokens=1)

        input_ids = self.input_ids[:num_tokens]
        positions = self.positions[:num_tokens]
        previous_hidden_states = self.hidden_states[:num_tokens]
        with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                num_tokens_across_dp=num_tokens_across_dp,
                in_profile_run=self.runner.in_profile_run,
                num_actual_tokens=0):
            if is_running_torchair:
                assert attn_metadata is not None
                torch._dynamo.mark_static(input_ids)
                torch._dynamo.mark_static(positions)
                torch._dynamo.mark_static(previous_hidden_states)
                torch._dynamo.mark_static(attn_metadata.decode.block_table)
                torch._dynamo.mark_static(attn_metadata.decode.input_positions)
                torch._dynamo.mark_static(attn_metadata.decode.sin)
                torch._dynamo.mark_static(attn_metadata.decode.cos)
                torch._dynamo.mark_static(get_forward_context().mc2_mask)
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
                torch._dynamo.mark_static(attn_metadata.decode.attn_mask)
                torchair_compiled_model = self._get_torchair_lazy_compiled_model(
                    num_tokens)
                torchair_compiled_model(
                    input_ids=input_ids,
                    positions=positions,
                    previous_hidden_states=previous_hidden_states,
                    inputs_embeds=None,
                    intermediate_tensors=None,
                    attn_metadata=attn_metadata,
                    kv_caches=self.runner.kv_caches[-1:],
                    spec_step_idx=0)
            else:
                self.model(input_ids=input_ids,
                           positions=positions,
                           previous_hidden_states=previous_hidden_states)

    def _get_torchair_lazy_compiled_model(self, batch_size: int):
        if batch_size < 0 or batch_size > self.runner.torchair_graph_batch_sizes[
                -1]:
            raise ValueError(
                f"Bad graph batch size:{batch_size}! max_graph_batch_sizes:{self.runner.torchair_graph_batch_sizes[-1]}"
            )

        compiled_model = self.torchair_compiled_models.get(
            batch_size
        ) if self.runner.use_cached_npu_graph else self.torchair_compiled_model

        if compiled_model:
            return compiled_model

        patch_for_hcom()
        config = torchair.CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        config.experimental_config.enable_view_optimize = \
        get_ascend_config().torchair_graph_config.enable_view_optimize
        torch.npu.set_compile_mode(jit_compile=False)
        if not self.runner.use_cached_npu_graph:
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.torchair_compiled_model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=npu_backend)
            return self.torchair_compiled_model
        else:
            # Generate a new forward proxy code object to prevent the invalidation of
            # compilation cache caused by dynamo retracing
            forward_proxy_name = f"{self.model.__class__.__name__}_forward_with_batch_size_{batch_size}"
            forward_fn = self.model.forward
            code = forward_fn.__code__
            # Mark code object with a new proxy name
            modified_code = code.replace(co_name=forward_proxy_name, )

            modified_func = types.FunctionType(modified_code,
                                               forward_fn.__globals__,
                                               name=forward_proxy_name,
                                               argdefs=forward_fn.__defaults__)

            self.model.__dict__[forward_proxy_name] = modified_func.__get__(
                self.model, nn.Module)
            self.torchair_compiled_models[
                batch_size] = torchair.inference.cache_compile(
                    self.model.__dict__[forward_proxy_name],
                    dynamic=True,
                    fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    cache_dir=TORCHAIR_CACHE_DIR,
                    config=config,
                    ge_cache=False)
            return self.torchair_compiled_models[batch_size]


# TODO Using torch instead of triton may result in poor performance
def prepare_input_kernel(out_ptr: torch.Tensor, cu_query_lens: torch.Tensor,
                         cu_num_tokens: torch.Tensor, block_size: int):
    device = cu_query_lens.device
    dtype = out_ptr.dtype

    offsets = torch.arange(block_size, device=device, dtype=dtype)
    start_pos = cu_num_tokens[:-1]
    end_pos = cu_num_tokens[1:]
    num_tokens = end_pos - start_pos

    global_indices = (start_pos.view(-1, 1) + offsets.view(1, -1))
    values = (cu_query_lens[:-1].view(-1, 1) + offsets.view(1, -1))

    mask = (offsets.view(1, -1) < num_tokens.view(-1, 1))

    global_indices_flat = global_indices[mask]
    values_flat = values[mask]
    out_ptr[global_indices_flat] = values_flat
