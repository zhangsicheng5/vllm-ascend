import sys
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm.distributed.parallel_state import GroupCoordinator

if 'torch_npu._inductor' not in sys.modules:
    sys.modules['torch_npu._inductor'] = MagicMock()

from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    _compute_sparse_softmax_lse,
)
from vllm_ascend.utils import enable_dsa_cp


class TestSFAOffloadHelpers(TestBase):

    def test_compute_sparse_softmax_lse_uses_log_sum_plus_max(self):
        softmax_max = torch.tensor([[1.0, -2.0], [0.5, 3.0]],
                                   dtype=torch.float16)
        softmax_sum = torch.tensor([[2.0, 4.0], [1.0, 0.5]],
                                   dtype=torch.float16)

        actual = _compute_sparse_softmax_lse(softmax_max, softmax_sum)

        expected = torch.log(
            softmax_sum.to(torch.float32)) + softmax_max.to(torch.float32)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected)

    def test_compute_sparse_softmax_lse_masks_rows_without_valid_kv(self):
        softmax_max = torch.tensor([[13.0, -7.0], [0.5, 3.0]],
                                   dtype=torch.float16)
        softmax_sum = torch.tensor([[-2.0, 4.0], [1.0, 0.5]],
                                   dtype=torch.float16)
        no_valid_kv_mask = torch.tensor([[True], [False]])

        actual = _compute_sparse_softmax_lse(
            softmax_max,
            softmax_sum,
            no_valid_kv_mask,
        )

        expected_valid_row = torch.log(
            softmax_sum[1:].to(torch.float32)) + softmax_max[1:].to(
                torch.float32)
        assert torch.isneginf(actual[0]).all()
        torch.testing.assert_close(actual[1:], expected_valid_row)

    def test_npu_sparse_flash_attention_all_negative_indices_returns_safe_lse(
            self):
        torch_npu_module = getattr(torch, "npu", None)
        is_npu_available = getattr(torch_npu_module, "is_available",
                                   lambda: False)
        if not is_npu_available():
            self.skipTest("NPU is not available")
        ascend_ops = getattr(torch.ops, "_C_ascend", None)
        if ascend_ops is None or not hasattr(ascend_ops,
                                             "npu_sparse_flash_attention"):
            self.skipTest("npu_sparse_flash_attention is not available")

        cases = (
            ("npu_direct_kv_cache", 100, 10, 1280),
            ("cpu_topk_buffer", 16, 16, 2048),
        )
        for name, num_blocks, block_table_width, seq_len_kv in cases:
            with self.subTest(name=name):
                sparse_indices = torch.full([1, 1, 2048],
                                            -1,
                                            dtype=torch.int32,
                                            device="npu")
                query = torch.randn([1, 64, 512],
                                    dtype=torch.bfloat16,
                                    device="npu")
                key = torch.randn([num_blocks, 128, 1, 512],
                                  dtype=torch.bfloat16,
                                  device="npu")
                key_rope = torch.randn([num_blocks, 128, 1, 64],
                                       dtype=torch.bfloat16,
                                       device="npu")
                block_table = torch.zeros([1, block_table_width],
                                          dtype=torch.int32,
                                          device="npu")
                seq_lens_q = torch.tensor([1], dtype=torch.int32, device="npu")
                seq_lens_kv = torch.tensor([seq_len_kv],
                                           dtype=torch.int32,
                                           device="npu")

                attn_out, softmax_max, softmax_sum = (
                    torch.ops._C_ascend.npu_sparse_flash_attention(
                        query=query,
                        key=key,
                        value=key,
                        sparse_indices=sparse_indices,
                        block_table=block_table,
                        actual_seq_lengths_query=seq_lens_q,
                        actual_seq_lengths_kv=seq_lens_kv,
                        query_rope=query[..., :64],
                        key_rope=key_rope,
                        scale_value=0.0625,
                        sparse_block_size=1,
                        layout_query="TND",
                        layout_kv="PA_BSND",
                        sparse_mode=3,
                    ))
                raw_lse = (torch.log(softmax_sum.to(torch.float32)) +
                           softmax_max.to(torch.float32))
                protected_lse = _compute_sparse_softmax_lse(
                    softmax_max,
                    softmax_sum,
                    torch.ones([1, 1], dtype=torch.bool, device="npu"),
                )
                stats = (
                    f"name={name} softmax_max={softmax_max.detach().cpu()} "
                    f"softmax_sum={softmax_sum.detach().cpu()} "
                    f"raw_lse={raw_lse.detach().cpu()}")

                assert not torch.isnan(attn_out).any(), stats
                assert not torch.isnan(softmax_max).any(), stats
                assert not torch.isnan(softmax_sum).any(), stats
                assert torch.isneginf(raw_lse).all(), stats
                assert torch.isneginf(protected_lse).all(), stats

    def test_sfa_v1_no_empty_lse_stubs_remain(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        source = (repo_root / "vllm_ascend" / "attention" /
                  "sfa_v1.py").read_text()

        assert "softmax_lse_decode_cpu = torch.empty" not in source
        assert "softmax_lse_npu = torch.empty" not in source
        assert "no_valid_kv_mask = torch.all" not in source
        assert (
            "torch.where(cpu_mask, cpu_token_indices, -1).unsqueeze(1)"
            not in source)
        assert "cpu_mask & valid_sparse_slots" in source
        assert "~torch.any(npu_mask, dim=-1, keepdim=True)" in source
        assert "~torch.any(cpu_mask, dim=-1, keepdim=True)" in source


class TestAscendSFABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(),
                         AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):

    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestAscendSFAMetadataBuilder(TestBase):

    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config",
                             return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(self, kv_cache_spec, layer_names, vllm_config,
                             device, metadata_cls, supports_dcp_with_varlen):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__",
            mock_parent_init)
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
            self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly
