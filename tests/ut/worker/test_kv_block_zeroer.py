from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.base import TestBase
from vllm_ascend.worker.utils import AscendKVBlockZeroer


class TestAscendKVBlockZeroer(TestBase):
    def setUp(self):
        self.zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
        self.spec = FullAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.int8,
        )
        self.group = SimpleNamespace(
            kv_cache_spec=self.spec,
            kv_cache_group_id=0,
            layer_names=["layer_0"],
        )

    def _init_meta(self, kv_cache: tuple[torch.Tensor, ...]) -> None:
        self.zeroer.init_meta(
            attn_groups_iter=[self.group],
            kernel_block_sizes=[[1]],
            cache_dtype="int8",
            runner_only_attn_layers=set(),
            static_forward_context={"layer_0": SimpleNamespace(kv_cache=kv_cache)},
        )

    def test_regular_kv_pair_uses_one_page_size_bucket(self):
        self._init_meta(
            (
                torch.empty((4, 16), dtype=torch.uint8),
                torch.empty((4, 16), dtype=torch.uint8),
            )
        )

        self.assertEqual(len(self.zeroer._metas), 1)
        _, page_size_el, _, num_segments = self.zeroer._metas[0]
        self.assertEqual(page_size_el, 8)
        self.assertEqual(num_segments, 2)

    def test_c8_mxfp_preserves_auxiliary_scale_caches(self):
        self._init_meta(
            (
                torch.empty((4, 16), dtype=torch.uint8),
                torch.empty((4, 16), dtype=torch.uint8),
                torch.empty((4, 4), dtype=torch.uint8),
                torch.empty((4, 4), dtype=torch.uint8),
            )
        )

        self.assertEqual(len(self.zeroer._metas), 1)
        _, page_size_el, _, num_segments = self.zeroer._metas[0]
        self.assertEqual(page_size_el, 8)
        self.assertEqual(num_segments, 2)

    @patch("vllm_ascend.worker.utils.get_vectorcore_num", return_value=32)
    @patch("vllm_ascend.worker.utils._zero_kv_blocks_kernel")
    def test_c8_mxfp_zeroes_only_kv_payload_bucket(
        self,
        mock_zero_kernel: MagicMock,
        _mock_vectorcore_num: MagicMock,
    ):
        self._init_meta(
            (
                torch.empty((4, 16), dtype=torch.uint8),
                torch.empty((4, 16), dtype=torch.uint8),
                torch.empty((4, 4), dtype=torch.uint8),
                torch.empty((4, 4), dtype=torch.uint8),
            )
        )

        self.zeroer.zero_block_ids([1])

        self.assertEqual(mock_zero_kernel.__getitem__.call_count, 1)
        self.assertEqual(mock_zero_kernel.__getitem__.return_value.call_count, 1)

    def test_full_attention_cache_requires_k_and_v(self):
        with self.assertRaisesRegex(ValueError, "at least K and V"):
            self._init_meta((torch.empty((4, 16), dtype=torch.uint8),))
