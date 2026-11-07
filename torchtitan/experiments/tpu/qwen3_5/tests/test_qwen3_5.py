"""Tests for Qwen3.5 (text-only Dense and MoE) on CPU and TPU devices.

Covers:
1. HuggingFace reference parity (``Qwen3_5ForCausalLM`` and
   ``Qwen3_5MoeForCausalLM``) through ``Qwen35StateDictAdapter`` for forward
   logits, cross-entropy loss, and every parameter gradient (with negative
   controls), for dense, MoE, and MoE with ``TpuGroupedExpertsConverter``;
   plus the adapter round trip.
2. ``chunked_gated_delta`` against the exact per-token recurrence, also with
   packed documents, and bitwise against its earlier inter-chunk loops (one
   indexing every chunk, whose backward writes a full-size buffer per chunk,
   and one taking ``qk`` per chunk, whose backward stacks the per-chunk ``qk``
   gradients); and the
   slice-based causal convolution's backward against autograd of ``conv1d``,
   also with packed documents, and in bf16 against the accuracy of the
   autograd it replaces.
3. Row isolation in the GatedDeltaNet and packed-document isolation in the
   causal convolution, the GatedDeltaNet, the attention and the whole model
   (with negative controls), plus Splash parity on TPU.
4. Flavor, registry and recipe checks.
"""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.nn.functional as F
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.muse_glimmer.attention import SlidingWindowScaledDotProductInnerAttention
import torchtitan.experiments.tpu.qwen3_5 as qwen3_5_tpu
from torchtitan.experiments.tpu.qwen3_5 import config_registry
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.experiments.tpu.qwen3_5 import model as tpu_model
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import parallelize_qwen3_5
from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.qwen3_5 import model as qwen3_5_model
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from transformers import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)


def _recurrent_gated_delta_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
  """Per-token reference for the gated delta rule recurrence in fp32.

  With ``(bs, seq_len)`` ``segment_ids``, the state restarts from zero at the
  first token of every document.
  """
  q = gated_delta._l2norm(q.float()) * (q.shape[-1] ** -0.5)  # pylint: disable=protected-access
  k = gated_delta._l2norm(k.float())  # pylint: disable=protected-access
  v, g, beta = v.float(), g.float(), beta.float()
  bs, seq_len, n_heads, d_k = k.shape
  state = torch.zeros(
      bs, n_heads, d_k, v.shape[-1], dtype=torch.float32, device=q.device
  )
  outs = []
  for t in range(seq_len):
    if segment_ids is not None and t > 0:
      new_document = segment_ids[:, t] != segment_ids[:, t - 1]
      state = torch.where(
          new_document.to(q.device)[:, None, None, None],
          torch.zeros_like(state),
          state,
      )
    state = state * g[:, t].exp()[..., None, None]
    kv_mem = (state * k[:, t, :, :, None]).sum(-2)
    delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
    state = state + k[:, t, :, :, None] * delta[:, :, None, :]
    outs.append((state * q[:, t, :, :, None]).sum(-2))
  return torch.stack(outs, dim=1)


def _indexed_inter_chunk_recurrence(
    wq: torch.Tensor,
    u: torch.Tensor,
    qk: torch.Tensor,
    k_decayed_t: torch.Tensor,
    chunk_decay: torch.Tensor,
) -> torch.Tensor:
  """``gated_delta._inter_chunk_recurrence`` indexing each chunk in the loop.

  The loop as it was before it took its chunks from ``unbind``: autograd turns
  every ``t[:, :, i]`` into a ``select_backward`` of the size of all of ``t``.
  """
  bs, n_heads, num_chunks, chunk, d_v = u.shape
  state = torch.zeros(
      bs, n_heads, wq.shape[-1], d_v, dtype=wq.dtype, device=wq.device
  )
  outs = []
  for i in range(num_chunks):
    wq_state = wq[:, :, i] @ state
    v_new = u[:, :, i] - wq_state[..., :chunk, :]
    outs.append(wq_state[..., chunk:, :] + qk[:, :, i] @ v_new)
    state = state * chunk_decay[:, :, i] + k_decayed_t[:, :, i] @ v_new
  return torch.stack(outs, dim=2)


def _unbind_inter_chunk_recurrence(
    wq: torch.Tensor,
    u: torch.Tensor,
    qk: torch.Tensor,
    k_decayed_t: torch.Tensor,
    chunk_decay: torch.Tensor,
) -> torch.Tensor:
  """``gated_delta._inter_chunk_recurrence`` with ``qk @ v_new`` in the loop.

  The loop as it was before the intra-chunk term moved after it: every chunk
  takes ``qk`` from one ``unbind``, so ``qk``'s gradient is a ``stack`` of
  ``num_chunks`` per-chunk gradients.
  """
  bs, n_heads, num_chunks, chunk, d_v = u.shape
  state = torch.zeros(
      bs, n_heads, wq.shape[-1], d_v, dtype=wq.dtype, device=wq.device
  )
  wq_c, u_c, qk_c = wq.unbind(2), u.unbind(2), qk.unbind(2)
  k_decayed_t_c, chunk_decay_c = k_decayed_t.unbind(2), chunk_decay.unbind(2)
  outs = []
  for i in range(num_chunks):
    wq_state = wq_c[i] @ state
    v_new = u_c[i] - wq_state[..., :chunk, :]
    outs.append(wq_state[..., chunk:, :] + qk_c[i] @ v_new)
    state = state * chunk_decay_c[i] + k_decayed_t_c[i] @ v_new
  return torch.stack(outs, dim=2)


def _gated_delta_inputs(
    bs: int,
    seq_len: int,
    n_heads: int,
    d_k: int,
    d_v: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
  """Random ``chunked_gated_delta`` inputs ``(q, k, v, g, beta)``."""
  return (
      torch.randn(bs, seq_len, n_heads, d_k, dtype=dtype),
      torch.randn(bs, seq_len, n_heads, d_k, dtype=dtype),
      torch.randn(bs, seq_len, n_heads, d_v, dtype=dtype),
      (-F.softplus(torch.randn(bs, seq_len, n_heads))).to(dtype),
      torch.sigmoid(torch.randn(bs, seq_len, n_heads)).to(dtype),
  )


def _packed_positions(doc_lens: tuple[int, ...]) -> torch.Tensor:
  """Positions of documents of ``doc_lens`` tokens packed into one row."""
  return torch.cat([torch.arange(n) for n in doc_lens])


def _relative_l2_error(test: torch.Tensor, reference: torch.Tensor) -> float:
  test_f = test.detach().float().cpu()
  ref_f = reference.detach().float().cpu()
  denom = torch.linalg.vector_norm(ref_f).clamp_min(1e-6)
  return (torch.linalg.vector_norm(test_f - ref_f) / denom).item()


class Qwen35Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for Qwen3.5 hybrid GatedDeltaNet + Attention + MoE on CPU and TPU."""

  def setUp(self):
    super().setUp()
    use_cpu_safe_histc_patch()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous_precision)

  def _is_tpu(self) -> bool:
    return self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU

  def _build_hf_reference(self, is_moe: bool) -> torch.nn.Module:
    # The geometry of upstream `debugmodel` (first 4 layers) / `debugmodel_moe`
    # with vocab_size=2048 for fast CPU execution.
    common_kwargs = dict(
        vocab_size=2048,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
        },
        attn_implementation="sdpa",
    )
    torch.manual_seed(42)
    if is_moe:
      hf_cfg = Qwen3_5MoeTextConfig(
          **common_kwargs,
          moe_intermediate_size=256,
          shared_expert_intermediate_size=256,
          num_experts=8,
          num_experts_per_tok=2,
          tie_word_embeddings=False,
      )
      hf_model = Qwen3_5MoeForCausalLM(hf_cfg).float().eval()
    else:
      hf_cfg = Qwen3_5TextConfig(
          **common_kwargs,
          intermediate_size=512,
          tie_word_embeddings=True,
      )
      hf_model = Qwen3_5ForCausalLM(hf_cfg).float().eval()

    # Perturb unit/zero-initialized parameters so dropped scales cannot hide.
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
      for name, param in hf_model.named_parameters():
        if "norm" in name or "A_log" in name or "dt_bias" in name:
          param.copy_(0.2 * torch.randn(param.shape, generator=gen))
    return hf_model

  @parameterized.named_parameters(
      dict(
          testcase_name="_dense",
          flavor="debugmodel",
          use_tpu_grouped_experts=False,
      ),
      dict(
          testcase_name="_moe",
          flavor="debugmodel_moe",
          use_tpu_grouped_experts=False,
      ),
      dict(
          testcase_name="_moe_tpu_grouped_experts",
          flavor="debugmodel_moe",
          use_tpu_grouped_experts=True,
      ),
      dict(
          testcase_name="_moe_pallas_paired_grouped_mm",
          flavor="debugmodel_moe",
          use_tpu_grouped_experts=True,
          use_pallas_paired_grouped_mm=True,
      ),
  )
  def test_qwen3_5_hf_reference_parity(
      self,
      flavor: str,
      use_tpu_grouped_experts: bool,
      use_pallas_paired_grouped_mm: bool = False,
  ):
    """Tests forward logits, loss, and all gradients against HF Qwen3.5."""
    if use_pallas_paired_grouped_mm and not self._is_tpu():
      self.skipTest("Pallas grouped matmuls are TPU-only.")
    is_moe = flavor == "debugmodel_moe"
    hf_model = self._build_hf_reference(is_moe=is_moe)
    converters = (
        [TpuGroupedExpertsConverter.Config()] if use_tpu_grouped_experts else []
    )
    tt_config = qwen3_5_tpu.model_registry(
        flavor, seq_len=128, converters=converters
    ).model
    if use_pallas_paired_grouped_mm:
      # The expert kernels of `_use_tpu_moe_kernels`, on canonical weights so
      # that the gradients compare by name.
      for _, experts, _, _ in tt_config.traverse(TpuGroupedExperts.Config):
        experts.use_pallas_grouped_mm = True
        experts.use_pallas_paired_grouped_mm = True
        experts.pallas_grouped_mm_tile_m = 256
        experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    tt_config.vocab_size = 2048
    tt_config.tok_embeddings.num_embeddings = 2048
    tt_config.lm_head.out_features = 2048
    tt_config.enable_weight_tying = not is_moe
    tt_config.layers = tt_config.layers[:4]
    adapter = Qwen35StateDictAdapter(tt_config, None)

    tt_model = tt_config.build()
    tt_model.init_states(buffer_device=torch.device("cpu"))

    # Check state_dict mapping both ways.
    tt_sd = adapter.from_hf(copy.deepcopy(hf_model.state_dict()))
    missing, unexpected = tt_model.load_state_dict(tt_sd, strict=False)
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    if not use_tpu_grouped_experts:
      round_trip = adapter.from_hf(adapter.to_hf(tt_model.state_dict()))
      self.assertEqual(
          set(round_trip.keys()), set(tt_model.state_dict().keys())
      )
      for k, v in tt_model.state_dict().items():
        torch.testing.assert_close(round_trip[k], v, msg=f"Round-trip key {k}")

    tt_model = tt_model.to(self.accelerator_device)
    if tt_config.enable_weight_tying:
      tt_model.tok_embeddings.weight = tt_model.lm_head.weight
      self.assertIs(tt_model.tok_embeddings.weight, tt_model.lm_head.weight)

    batch, seq_len = 2, 16
    gen = torch.Generator().manual_seed(99)
    tokens_2d = torch.randint(
        0, tt_config.vocab_size, (batch, seq_len), generator=gen
    )
    labels_2d = torch.randint(
        0, tt_config.vocab_size, (batch, seq_len), generator=gen
    )
    positions_1d = torch.arange(seq_len, dtype=torch.long).repeat(batch)

    # Reference forward + backward on CPU in fp32.
    hf_model.zero_grad()
    hf_logits = hf_model(tokens_2d).logits
    hf_loss = F.cross_entropy(
        hf_logits.view(-1, tt_config.vocab_size), labels_2d.view(-1)
    )
    hf_loss.backward()

    # TorchTitan forward + backward on accelerator_device in packed [T] layout.
    tt_model.zero_grad()
    tokens_dev = tokens_2d.view(-1).to(self.accelerator_device)
    labels_dev = labels_2d.view(-1).to(self.accelerator_device)
    positions_dev = positions_1d.to(self.accelerator_device)

    # libtpu 0.0.48 rejects HIGHEST matmul precision for bf16/fp32 ragged dot
    # inside torch._grouped_mm on TPU; use "high" precision when running MoE on
    # TPU.
    precision = "high" if (self._is_tpu() and is_moe) else "highest"
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
      with set_unflatten_seq_len(seq_len):
        tt_logits = tt_model(tokens_dev, positions=positions_dev)
        tt_loss = F.cross_entropy(
            tt_logits.float().view(-1, tt_config.vocab_size), labels_dev
        )
        tt_loss.backward()
    finally:
      torch.set_float32_matmul_precision(prev_prec)

    # GroupedExperts runs expert matmuls in bfloat16 even when the surrounding
    # model is float32 on CPU, so MoE uses bfloat16 tolerances on both CPU and
    # TPU.
    logit_atol = 5e-2 if (self._is_tpu() or is_moe) else 5e-3
    logit_rtol = 1e-2 if (self._is_tpu() or is_moe) else 5e-3
    loss_atol = 5e-3 if (self._is_tpu() or is_moe) else 1e-5
    loss_rtol = 1e-3 if (self._is_tpu() or is_moe) else 1e-5
    grad_rel_bound = 6e-2 if (self._is_tpu() or is_moe) else 5e-4

    test_utils.check_equivalence(
        tt_logits.cpu().view(batch, seq_len, -1),
        hf_logits.detach(),
        atol=logit_atol,
        rtol=logit_rtol,
        check_name="HF Parity Logits",
    )
    test_utils.check_equivalence(
        tt_loss.cpu(),
        hf_loss.detach(),
        atol=loss_atol,
        rtol=loss_rtol,
        check_name="HF Parity Loss",
    )

    hf_grads = {
        name: param.grad.detach().clone()
        for name, param in hf_model.named_parameters()
        if param.grad is not None
    }
    expected_tt_grads = adapter.from_hf(hf_grads)
    if tt_config.enable_weight_tying:
      expected_tt_grads.pop("lm_head.weight", None)

    actual_tt_grads = {
        name: param.grad.detach().float().cpu()
        for name, param in tt_model.named_parameters()
        if param.grad is not None
    }

    self.assertEqual(set(actual_tt_grads.keys()), set(expected_tt_grads.keys()))
    for name, ref_grad in expected_tt_grads.items():
      rel_err = _relative_l2_error(actual_tt_grads[name], ref_grad)
      self.assertLessEqual(
          rel_err,
          grad_rel_bound,
          msg=f"Gradient relative L2 error too high for {name}: {rel_err:.3e}",
      )
      # Negative controls: zeroed and sign-flipped gradients must exceed bound.
      if torch.linalg.vector_norm(ref_grad).item() > 1e-4:
        self.assertGreater(
            _relative_l2_error(torch.zeros_like(ref_grad), ref_grad),
            grad_rel_bound,
        )
        self.assertGreater(
            _relative_l2_error(-ref_grad, ref_grad),
            grad_rel_bound,
        )

  @parameterized.named_parameters(
      # Several chunks, the last one padded.
      dict(testcase_name="_random", correlated_keys=False, chunk_size=16),
      # Nearly parallel keys with beta near 1 and little decay push the UT
      # transform's triangular system toward its worst conditioning.
      dict(
          testcase_name="_correlated_chunk64",
          correlated_keys=True,
          chunk_size=64,
      ),
      dict(
          testcase_name="_correlated_chunk128",
          correlated_keys=True,
          chunk_size=128,
      ),
  )
  def test_chunked_gated_delta_matches_recurrent(
      self, correlated_keys: bool, chunk_size: int
  ):
    """Tests chunked_gated_delta against the per-token recurrent reference."""
    torch.manual_seed(0)
    if correlated_keys:
      bs, seq_len, n_heads, d_k, d_v = 1, 160, 2, 32, 32
      base = torch.randn(bs, 1, n_heads, d_k)
      k = base + 0.05 * torch.randn(bs, seq_len, n_heads, d_k)
      g_raw = torch.full((bs, seq_len, n_heads), -4.0)
      beta_raw = torch.full((bs, seq_len, n_heads), 4.0)
    else:
      bs, seq_len, n_heads, d_k, d_v = 2, 50, 4, 16, 16
      k = torch.randn(bs, seq_len, n_heads, d_k)
      g_raw = torch.randn(bs, seq_len, n_heads)
      beta_raw = torch.randn(bs, seq_len, n_heads)
    q = torch.randn(bs, seq_len, n_heads, d_k)
    v = torch.randn(bs, seq_len, n_heads, d_v)
    leaves_cpu = tuple(x.requires_grad_() for x in (q, k, v, g_raw, beta_raw))
    leaves_dev = tuple(
        x.detach().to(self.accelerator_device).requires_grad_()
        for x in leaves_cpu
    )

    def run(fn, q, k, v, g_raw, beta_raw, **kwargs):
      return fn(q, k, v, -F.softplus(g_raw), torch.sigmoid(beta_raw), **kwargs)

    expected = run(_recurrent_gated_delta_reference, *leaves_cpu)
    out_dev = run(
        gated_delta.chunked_gated_delta, *leaves_dev, chunk_size=chunk_size
    )
    tol = 5e-3 if self._is_tpu() else 1e-4
    test_utils.check_equivalence(
        out_dev.cpu(),
        expected,
        atol=tol,
        rtol=tol,
        check_name="Chunked vs Recurrent GDN Forward",
    )
    grad_out = torch.randn_like(expected)
    expected.backward(grad_out)
    out_dev.backward(grad_out.to(self.accelerator_device))
    grad_tol = 2e-2 if self._is_tpu() else 5e-4
    for name, x_cpu, x_dev in zip(
        ("q", "k", "v", "g", "beta"), leaves_cpu, leaves_dev, strict=True
    ):
      self.assertLessEqual(
          _relative_l2_error(x_dev.grad, x_cpu.grad),
          grad_tol,
          msg=f"GDN {name} grad mismatch",
      )

  def test_causal_conv_silu_matches_autograd(self):
    """Tests the slice-based causal conv backward against autograd/conv1d."""
    torch.manual_seed(0)
    channels, kernel_size = 8, 4
    x = torch.randn(3, 37, channels, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(
        channels, 1, kernel_size, dtype=torch.float64, requires_grad=True
    )
    grad = torch.randn(3, 37, channels, dtype=torch.float64)
    # pylint: disable=protected-access
    conv = gated_delta._causal_conv(x, weight)
    expected_conv = F.conv1d(
        F.pad(x.transpose(1, 2), (kernel_size - 1, 0)),
        weight,
        groups=channels,
    ).transpose(1, 2)
    # pylint: enable=protected-access
    torch.testing.assert_close(conv, expected_conv)
    out = gated_delta.causal_conv_silu(x, weight)
    expected = F.silu(expected_conv)
    torch.testing.assert_close(out, expected)
    grads = torch.autograd.grad(out, (x, weight), grad)
    expected_grads = torch.autograd.grad(expected, (x, weight), grad)
    for actual_grad, expected_grad in zip(grads, expected_grads, strict=True):
      torch.testing.assert_close(actual_grad, expected_grad)

    # On the accelerator in bf16, as the model runs it.
    dev = self.accelerator_device
    x_bf16 = x.detach().to(dev, torch.bfloat16).requires_grad_()
    w_bf16 = weight.detach().to(dev, torch.bfloat16).requires_grad_()
    out_bf16 = gated_delta.causal_conv_silu(x_bf16, w_bf16)
    grads_bf16 = torch.autograd.grad(
        out_bf16, (x_bf16, w_bf16), grad.to(dev, torch.bfloat16)
    )
    self.assertLess(_relative_l2_error(out_bf16.double().cpu(), expected), 1e-2)
    for actual_grad, expected_grad in zip(
        grads_bf16, expected_grads, strict=True
    ):
      self.assertEqual(actual_grad.dtype, torch.bfloat16)
      self.assertLess(
          _relative_l2_error(actual_grad.double().cpu(), expected_grad), 2e-2
      )

  @parameterized.named_parameters(
      dict(testcase_name="_seq1", seq_len=1),
      dict(testcase_name="_seq2", seq_len=2),
      dict(testcase_name="_seq3", seq_len=3),
  )
  def test_causal_conv_silu_sequences_shorter_than_kernel(self, seq_len: int):
    """Tests the conv backward when taps reach before the sequence start."""
    torch.manual_seed(0)
    channels, kernel_size = 8, 4
    x = torch.randn(
        2, seq_len, channels, dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn(
        channels, 1, kernel_size, dtype=torch.float64, requires_grad=True
    )
    grad = torch.randn(2, seq_len, channels, dtype=torch.float64)
    expected = F.silu(
        F.conv1d(
            F.pad(x.transpose(1, 2), (kernel_size - 1, 0)),
            weight,
            groups=channels,
        ).transpose(1, 2)
    )
    out = gated_delta.causal_conv_silu(x, weight)
    torch.testing.assert_close(out, expected)
    grads = torch.autograd.grad(out, (x, weight), grad)
    expected_grads = torch.autograd.grad(expected, (x, weight), grad)
    for actual_grad, expected_grad in zip(grads, expected_grads, strict=True):
      torch.testing.assert_close(actual_grad, expected_grad)

  def test_causal_conv_silu_partial_input_grads(self):
    """Tests the conv backward when only the input or the weight needs grad."""
    torch.manual_seed(0)
    channels, kernel_size = 8, 4
    x = torch.randn(2, 11, channels, dtype=torch.float64)
    weight = torch.randn(channels, 1, kernel_size, dtype=torch.float64)
    grad = torch.randn(2, 11, channels, dtype=torch.float64)
    for name, x_grad in (("x", True), ("weight", False)):
      with self.subTest(grad_of=name):
        x_in = x.clone().requires_grad_(x_grad)
        w_in = weight.clone().requires_grad_(not x_grad)
        leaf = x_in if x_grad else w_in
        expected = F.silu(
            F.conv1d(
                F.pad(x_in.transpose(1, 2), (kernel_size - 1, 0)),
                w_in,
                groups=channels,
            ).transpose(1, 2)
        )
        (actual_grad,) = torch.autograd.grad(
            gated_delta.causal_conv_silu(x_in, w_in), leaf, grad
        )
        (expected_grad,) = torch.autograd.grad(expected, leaf, grad)
        torch.testing.assert_close(actual_grad, expected_grad)

  @parameterized.named_parameters(
      dict(testcase_name="_rows", packed=False),
      dict(testcase_name="_packed_documents", packed=True),
  )
  def test_causal_conv_silu_bf16_backward_accuracy(self, packed: bool):
    """Tests the bf16 conv backward against fp64 over 16K-token dw sums.

    The slice-based backward must be as accurate as the autograd it replaces,
    which multiplies in fp32 and sums dw in fp32. Adding the dw products in
    bf16 is not: torch_tpu disallows XLA excess precision, so a bf16 sum rounds
    every add.
    """
    torch.manual_seed(0)
    bs, seq_len, channels, kernel_size = 8, 2048, 128, 4
    # Biased activations and gradients (correlated, as in the model) make the
    # dw sums grow with the number of tokens.
    x = (torch.randn(bs, seq_len, channels) + 0.5).bfloat16()
    weight = (0.5 * torch.randn(channels, 1, kernel_size)).bfloat16()
    grad = (torch.randn(bs, seq_len, channels) + 0.25).bfloat16()
    segment_ids = None
    if packed:
      doc_lens = (700, 1, 3, 500, 844)
      segment_ids = segment_ids_from_positions(_packed_positions(doc_lens))
      segment_ids = segment_ids.expand(bs, -1)

    def conv_grads(fn, device, dtype):
      x_in = x.to(device, dtype).requires_grad_()
      w_in = weight.to(device, dtype).requires_grad_()
      seg = None if segment_ids is None else segment_ids.to(device)
      return torch.autograd.grad(
          fn(x_in, w_in, seg), (x_in, w_in), grad.to(device, dtype)
      )

    def autograd_conv_silu(x_in, w_in, seg):
      return F.silu(gated_delta._causal_conv(x_in, w_in, seg))  # pylint: disable=protected-access

    # fp64 on the bf16 values: only the backward's own rounding counts.
    expected_grads = conv_grads(autograd_conv_silu, "cpu", torch.float64)
    dev = self.accelerator_device
    grads = conv_grads(gated_delta.causal_conv_silu, dev, torch.bfloat16)
    autograd_grads = conv_grads(autograd_conv_silu, dev, torch.bfloat16)
    for name, grad_bf16, autograd_bf16, expected_grad in zip(
        ("x", "weight"), grads, autograd_grads, expected_grads, strict=True
    ):
      self.assertEqual(grad_bf16.dtype, torch.bfloat16)
      err = _relative_l2_error(grad_bf16, expected_grad)
      autograd_err = _relative_l2_error(autograd_bf16, expected_grad)
      self.assertLessEqual(
          err,
          1.25 * autograd_err,
          msg=f"d{name}: {err:.2e} vs autograd {autograd_err:.2e}",
      )
      self.assertLess(err, 1e-2, msg=f"d{name}: {err:.2e}")

  def test_causal_conv_silu_backward_avoids_scatter_adds(self):
    """Tests that the conv backward does not add slices into zero buffers.

    Autograd of the slice-based forward adds every tap's input gradient into
    a zero buffer of the padded input (``slice_backward``), which XLA emits as
    slow ``dynamic-update-slice`` adds. The traced backward of
    ``causal_conv_silu`` must have no such op, with or without documents.
    """
    scatter_ops = {
        torch.ops.aten.slice_backward.default,
        torch.ops.aten.slice_scatter.default,
        torch.ops.aten.index_put.default,
        torch.ops.aten.scatter_add.default,
    }
    x = torch.randn(2, 16, 8, requires_grad=True)
    weight = torch.randn(8, 1, 4, requires_grad=True)
    grad = torch.randn(2, 16, 8)
    segment_ids = segment_ids_from_positions(_packed_positions((5, 11)))

    def autograd_conv_silu(x_in, w_in, seg):
      return F.silu(gated_delta._causal_conv(x_in, w_in, seg))  # pylint: disable=protected-access

    def backward_ops(conv_silu, seg):
      def grads(x_in, w_in, grad_in):
        return torch.autograd.grad(
            conv_silu(x_in, w_in, seg), (x_in, w_in), grad_in
        )

      graph = make_fx(grads)(x, weight, grad).graph
      return {node.target for node in graph.nodes if node.op == "call_function"}

    for seg in (None, segment_ids.expand(2, -1)):
      with self.subTest(packed=seg is not None):
        self.assertEmpty(
            backward_ops(gated_delta.causal_conv_silu, seg) & scatter_ops
        )
        # Negative control: autograd of the forward does use them.
        self.assertNotEmpty(backward_ops(autograd_conv_silu, seg) & scatter_ops)

  def test_causal_conv_silu_compiles_fullgraph(self):
    """Tests causal_conv_silu under torch.compile(fullgraph=True).

    The per-layer compile mode compiles each transformer block with
    ``fullgraph=True``, so the custom autograd.Function must trace without
    graph breaks.
    """
    torch.manual_seed(0)
    self.addCleanup(torch._dynamo.reset)  # pylint: disable=protected-access
    x = torch.randn(2, 16, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(8, 1, 4, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(2, 16, 8, dtype=torch.float64)
    segment_ids = segment_ids_from_positions(_packed_positions((5, 11)))
    compiled = torch.compile(
        gated_delta.causal_conv_silu, backend="aot_eager", fullgraph=True
    )
    for seg in (None, segment_ids.expand(2, -1)):
      with self.subTest(packed=seg is not None):
        out = compiled(x, weight, seg)
        expected = gated_delta.causal_conv_silu(x, weight, seg)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(
            torch.autograd.grad(out, (x, weight), grad),
            torch.autograd.grad(expected, (x, weight), grad),
        )

  @parameterized.named_parameters(
      dict(
          testcase_name="_fp32",
          dtype=torch.float32,
          packed=False,
          reference=_indexed_inter_chunk_recurrence,
      ),
      dict(
          testcase_name="_fp32_packed",
          dtype=torch.float32,
          packed=True,
          reference=_indexed_inter_chunk_recurrence,
      ),
      dict(
          testcase_name="_bf16_packed",
          dtype=torch.bfloat16,
          packed=True,
          reference=_indexed_inter_chunk_recurrence,
      ),
      dict(
          testcase_name="_fp32_unbind_loop",
          dtype=torch.float32,
          packed=False,
          reference=_unbind_inter_chunk_recurrence,
      ),
      dict(
          testcase_name="_fp32_packed_unbind_loop",
          dtype=torch.float32,
          packed=True,
          reference=_unbind_inter_chunk_recurrence,
      ),
      dict(
          testcase_name="_bf16_packed_unbind_loop",
          dtype=torch.bfloat16,
          packed=True,
          reference=_unbind_inter_chunk_recurrence,
      ),
  )
  def test_chunked_gated_delta_matches_indexed_loop_bitwise(
      self, dtype: torch.dtype, packed: bool, reference
  ):
    """Tests that the inter-chunk loop's rewrites change no bit.

    The forward and all five input gradients of ``chunked_gated_delta`` must
    have the same bit patterns as with the earlier forms of the inter-chunk
    loop: indexing every chunk (``_indexed_inter_chunk_recurrence``) and taking
    ``qk`` per chunk inside the loop (``_unbind_inter_chunk_recurrence``). Runs
    on CPU with one thread, where the kernels themselves are deterministic.
    """
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    self.addCleanup(torch.set_num_threads, num_threads)
    torch.manual_seed(0)
    # Five 16-token chunks, the last one padded.
    bs, seq_len, n_heads, d_k, d_v = 2, 70, 2, 16, 16
    segment_ids = None
    if packed:
      # Documents start mid-chunk and at chunk starts (16 and 48).
      segment_ids = segment_ids_from_positions(
          _packed_positions((5, 11, 1, 31, 22))
      ).expand(bs, -1)
    leaves = _gated_delta_inputs(bs, seq_len, n_heads, d_k, d_v, dtype)
    grad = torch.randn(bs, seq_len, n_heads, d_v, dtype=dtype)

    def forward_and_grads():
      inputs = tuple(x.clone().requires_grad_() for x in leaves)
      out = gated_delta.chunked_gated_delta(
          *inputs, chunk_size=16, segment_ids=segment_ids
      )
      return (out.detach(), *torch.autograd.grad(out, inputs, grad))

    results = forward_and_grads()
    with mock.patch.object(
        gated_delta,
        "_inter_chunk_recurrence",
        wraps=reference,
    ) as indexed:
      indexed_results = forward_and_grads()
    indexed.assert_called_once()
    names = ("out", "q grad", "k grad", "v grad", "g grad", "beta grad")
    for name, actual, expected in zip(
        names, results, indexed_results, strict=True
    ):
      self.assertEqual(actual.dtype, dtype, msg=name)
      self.assertEqual(actual.shape, expected.shape, msg=name)
      # Compares bit patterns, which unlike values also tell -0.0 from 0.0.
      self.assertTrue(
          torch.equal(
              actual.contiguous().view(torch.uint8),
              expected.contiguous().view(torch.uint8),
          ),
          msg=f"{name} differs from {reference.__name__}'s",
      )

  def test_chunked_gated_delta_backward_avoids_select_backward(self):
    """Tests that the backward writes no full-size buffer per chunk.

    Autograd of an in-loop ``t[:, :, i]`` is a ``select_backward`` that writes
    a zero buffer the size of all of ``t`` for one chunk, so the backward of
    the inter-chunk loop would write O(seq_len^2 / chunk_size) bytes. The
    traced backward of ``chunked_gated_delta`` must have no such op, with or
    without documents, while that of the indexed loop has them.
    """
    select_ops = {
        torch.ops.aten.select_backward.default,
        torch.ops.aten.select_scatter.default,
    }
    n_heads, d_k, d_v, chunk_size = 2, 8, 8, 16

    def select_writes(recurrence, seq_len, packed):
      """Number and total bytes of the select ops in the traced backward."""
      leaves = tuple(
          x.requires_grad_()
          for x in _gated_delta_inputs(1, seq_len, n_heads, d_k, d_v)
      )
      grad = torch.randn(1, seq_len, n_heads, d_v)
      segment_ids = None
      if packed:
        segment_ids = segment_ids_from_positions(
            _packed_positions((20, seq_len - 20)).unsqueeze(0)
        )

      def grads(q, k, v, g, beta, grad_in):
        out = gated_delta.chunked_gated_delta(
            q, k, v, g, beta, chunk_size=chunk_size, segment_ids=segment_ids
        )
        return torch.autograd.grad(out, (q, k, v, g, beta), grad_in)

      with mock.patch.object(
          gated_delta, "_inter_chunk_recurrence", recurrence
      ):
        graph = make_fx(grads)(*leaves, grad).graph
      nodes = [
          node
          for node in graph.nodes
          if node.op == "call_function" and node.target in select_ops
      ]
      return len(nodes), sum(node.meta["val"].nbytes for node in nodes)

    unbind = gated_delta._inter_chunk_recurrence  # pylint: disable=protected-access
    seq_len = 64
    for packed in (False, True):
      with self.subTest(packed=packed):
        self.assertEqual(select_writes(unbind, seq_len, packed), (0, 0))
        self.assertEqual(select_writes(unbind, 2 * seq_len, packed), (0, 0))
        # Negative control: the indexed loop writes a buffer of all of `wq`,
        # `u` and `qk` for each chunk, so twice the tokens write at least four
        # times the bytes.
        num_selects, num_bytes = select_writes(
            _indexed_inter_chunk_recurrence, seq_len, packed
        )
        num_selects_2x, num_bytes_2x = select_writes(
            _indexed_inter_chunk_recurrence, 2 * seq_len, packed
        )
        self.assertGreaterEqual(num_selects, 3 * (seq_len // chunk_size))
        self.assertGreaterEqual(num_selects_2x, 2 * num_selects)
        self.assertGreaterEqual(num_bytes_2x, 4 * num_bytes)

  def test_chunked_gated_delta_backward_has_no_qk_gradient_stack(self):
    """Tests that the backward stacks no per-chunk ``(chunk, chunk)`` gradients.

    ``qk @ stack(v_new)`` after the loop makes ``qk``'s gradient one batched
    matmul. With ``qk`` taken per chunk inside the loop, its gradient is a
    ``stack`` of ``num_chunks`` per-chunk gradients, which XLA on TPU lays out
    with the chunk axis minor-most.
    """
    n_heads, d_k, d_v, chunk_size, seq_len = 2, 8, 4, 16, 64
    num_chunks = seq_len // chunk_size

    def qk_gradient_stacks(recurrence):
      """Stacks / concatenations of ``num_chunks`` ``(chunk, chunk)`` blocks."""
      leaves = tuple(
          x.requires_grad_()
          for x in _gated_delta_inputs(1, seq_len, n_heads, d_k, d_v)
      )
      grad = torch.randn(1, seq_len, n_heads, d_v)

      def grads(q, k, v, g, beta, grad_in):
        out = gated_delta.chunked_gated_delta(
            q, k, v, g, beta, chunk_size=chunk_size
        )
        return torch.autograd.grad(out, (q, k, v, g, beta), grad_in)

      with mock.patch.object(
          gated_delta, "_inter_chunk_recurrence", recurrence
      ):
        graph = make_fx(grads)(*leaves, grad).graph
      want = (num_chunks, chunk_size, chunk_size)
      return sum(
          1
          for node in graph.nodes
          if node.op == "call_function"
          and node.target
          in (torch.ops.aten.stack.default, torch.ops.aten.cat.default)
          and len(node.args[0]) == num_chunks
          and tuple(node.meta["val"].shape[-3:]) == want
      )

    batched = gated_delta._inter_chunk_recurrence  # pylint: disable=protected-access
    self.assertEqual(qk_gradient_stacks(batched), 0)
    # Negative control: the loop that takes `qk` per chunk stacks its gradient.
    self.assertGreaterEqual(
        qk_gradient_stacks(_unbind_inter_chunk_recurrence), 1
    )

  def test_gdn_unit_solve_triangular_fake_strides_match_the_kernel(self):
    """Tests that the solve op's fake strides are C-contiguous, as its output.

    A saved tensor reaches the compiled backward with the strides of its real
    value while the backward was compiled for those of its fake value. Unlike
    ``aten.linalg_solve_triangular``, whose meta kernel reports column-major
    matrices, the op's fake and real outputs are both C-contiguous.
    """
    solve_op = torch.ops.torchtitan_tpu.gdn_unit_solve_triangular.default
    self.assertEqual(
        gated_delta.GDN_SOLVE_OP, "torchtitan_tpu::gdn_unit_solve_triangular"
    )
    # The shapes at 16384 tokens: 256 chunks of 64, 16 heads, d_k + d_v = 256.
    with FakeTensorMode():
      a, rhs = torch.empty(1, 16, 256, 64, 64), torch.empty(1, 16, 256, 64, 256)
      fake = solve_op(a, rhs)
      aten_meta = torch.linalg.solve_triangular(
          a, rhs, upper=False, unitriangular=True
      )
    self.assertEqual(tuple(fake.shape), (1, 16, 256, 64, 256))
    self.assertTrue(fake.is_contiguous())
    self.assertFalse(aten_meta.is_contiguous())  # Column-major matrices.
    torch.manual_seed(0)
    a = (torch.randn(1, 2, 3, 64, 64) * 0.3).tril(-1)
    self.assertTrue(solve_op(a, torch.randn(1, 2, 3, 64, 32)).is_contiguous())

  @parameterized.named_parameters(
      dict(testcase_name="_fp32", dtype=torch.float32),
      dict(testcase_name="_fp64", dtype=torch.float64),
  )
  def test_gdn_unit_solve_triangular_matches_aten_bitwise(self, dtype):
    """Tests the solve op's forward and backward against aten's, bit for bit.

    Only the strictly lower triangle of ``a`` is read, batch dimensions
    broadcast as in aten, and the gradients of every subset of inputs equal
    those of autograd through ``torch.linalg.solve_triangular``.
    """
    solve_op = torch.ops.torchtitan_tpu.gdn_unit_solve_triangular.default
    torch.manual_seed(0)

    def aten_solve(a, rhs):
      return torch.linalg.solve_triangular(
          a, rhs, upper=False, unitriangular=True
      )

    a = (torch.randn(1, 4, 3, 64, 64, dtype=dtype) * 0.3).tril(-1)
    rhs = torch.randn(1, 4, 3, 64, 96, dtype=dtype)
    out = solve_op(a, rhs)
    self.assertTrue(torch.equal(out, aten_solve(a, rhs)))
    upper = torch.randn(a.shape, dtype=dtype).triu()
    self.assertTrue(torch.equal(solve_op(a + upper, rhs), out))
    a_b, rhs_b = a[:, :1], torch.randn(2, 4, 3, 64, 32, dtype=dtype)
    self.assertTrue(torch.equal(solve_op(a_b, rhs_b), aten_solve(a_b, rhs_b)))
    grad = torch.randn_like(out)
    for grad_a, grad_rhs in ((True, True), (True, False), (False, True)):
      with self.subTest(grad_a=grad_a, grad_rhs=grad_rhs):
        ins = (
            a.clone().requires_grad_(grad_a),
            rhs.clone().requires_grad_(grad_rhs),
        )
        ref = (
            a.clone().requires_grad_(grad_a),
            rhs.clone().requires_grad_(grad_rhs),
        )
        got = torch.autograd.grad(
            solve_op(*ins), [t for t in ins if t.requires_grad], grad
        )
        want = torch.autograd.grad(
            aten_solve(*ref), [t for t in ref if t.requires_grad], grad
        )
        for g_got, g_want in zip(got, want, strict=True):
          self.assertTrue(torch.equal(g_got, g_want))
    if dtype == torch.float64:
      small_a = torch.randn(1, 2, 2, 8, 8, dtype=dtype).requires_grad_()
      small_rhs = torch.randn(1, 2, 2, 8, 5, dtype=dtype).requires_grad_()
      self.assertTrue(
          torch.autograd.gradcheck(
              lambda a, rhs: solve_op(a.tril(-1), rhs), (small_a, small_rhs)
          )
      )
      torch.library.opcheck(
          solve_op, (a.clone().requires_grad_(), rhs.clone().requires_grad_())
      )

  def test_chunked_gated_delta_solve_op_matches_aten_solve_bitwise(self):
    """Tests that the UT transform's solve op changes no bit of the layer.

    ``chunked_gated_delta`` with ``torch.linalg.solve_triangular`` in place of
    the custom op gives the same output and input gradients, bit for bit.
    """
    torch.manual_seed(0)
    leaves = _gated_delta_inputs(2, 70, 2, 16, 16)
    grad = torch.randn(2, 70, 2, 16)

    def forward_and_grads():
      inputs = tuple(x.clone().requires_grad_() for x in leaves)
      out = gated_delta.chunked_gated_delta(*inputs, chunk_size=16)
      return (out.detach(), *torch.autograd.grad(out, inputs, grad))

    with_op = forward_and_grads()
    with mock.patch.object(
        gated_delta,
        "gdn_unit_solve_triangular",
        lambda a, rhs: torch.linalg.solve_triangular(
            a, rhs, upper=False, unitriangular=True
        ),
    ):
      with_aten = forward_and_grads()
    for actual, expected in zip(with_op, with_aten, strict=True):
      self.assertTrue(
          torch.equal(
              actual.contiguous().view(torch.uint8),
              expected.contiguous().view(torch.uint8),
          )
      )

  @parameterized.named_parameters(
      # Documents start mid-chunk and at chunk starts (16 and 48 with 16-token
      # chunks), one has a single token, some span several chunks, and the
      # last chunk is padded.
      dict(testcase_name="_chunk16", chunk_size=16),
      dict(testcase_name="_chunk64", chunk_size=64),
  )
  def test_chunked_gated_delta_resets_at_documents(self, chunk_size: int):
    """Tests chunked_gated_delta on packed documents against the recurrence."""
    torch.manual_seed(0)
    doc_lens = ((5, 11, 1, 31, 22), (70,))
    bs, seq_len, n_heads, d_k, d_v = len(doc_lens), 70, 2, 16, 16
    segment_ids = segment_ids_from_positions(
        torch.stack([_packed_positions(lens) for lens in doc_lens])
    )
    q = torch.randn(bs, seq_len, n_heads, d_k)
    k = torch.randn(bs, seq_len, n_heads, d_k)
    v = torch.randn(bs, seq_len, n_heads, d_v)
    g_raw = torch.randn(bs, seq_len, n_heads)
    beta_raw = torch.randn(bs, seq_len, n_heads)
    leaves_cpu = tuple(x.requires_grad_() for x in (q, k, v, g_raw, beta_raw))
    leaves_dev = tuple(
        x.detach().to(self.accelerator_device).requires_grad_()
        for x in leaves_cpu
    )

    def run(fn, q, k, v, g_raw, beta_raw, **kwargs):
      return fn(q, k, v, -F.softplus(g_raw), torch.sigmoid(beta_raw), **kwargs)

    expected = run(
        _recurrent_gated_delta_reference, *leaves_cpu, segment_ids=segment_ids
    )
    out_dev = run(
        gated_delta.chunked_gated_delta,
        *leaves_dev,
        chunk_size=chunk_size,
        segment_ids=segment_ids.to(self.accelerator_device),
    )
    tol = 5e-3 if self._is_tpu() else 1e-4
    test_utils.check_equivalence(
        out_dev.cpu(),
        expected,
        atol=tol,
        rtol=tol,
        check_name="Packed Chunked vs Recurrent GDN Forward",
    )
    grad_out = torch.randn_like(expected)
    expected.backward(grad_out)
    out_dev.backward(grad_out.to(self.accelerator_device))
    grad_tol = 2e-2 if self._is_tpu() else 5e-4
    for name, x_cpu, x_dev in zip(
        ("q", "k", "v", "g", "beta"), leaves_cpu, leaves_dev, strict=True
    ):
      self.assertLessEqual(
          _relative_l2_error(x_dev.grad, x_cpu.grad),
          grad_tol,
          msg=f"Packed GDN {name} grad mismatch",
      )

    # Negative control: without segment_ids, the state of each document leaks
    # into the first tokens of the next one.
    with torch.no_grad():
      leaky = run(
          gated_delta.chunked_gated_delta, *leaves_dev, chunk_size=chunk_size
      )
    starts = [5, 16, 17, 48]
    self.assertGreater(
        _relative_l2_error(leaky[0, starts], expected[0, starts]),
        0.1,
        msg="Negative control failed: no state leak without segment_ids",
    )

  def test_causal_conv_resets_at_documents(self):
    """Tests that causal conv1d taps into an earlier document read zeros."""
    torch.manual_seed(0)
    # Documents shorter than the 4-tap kernel, including a 1-token document.
    doc_lens = (3, 1, 6, 2, 8)
    seq_len, channels = sum(doc_lens), 8
    segment_ids = segment_ids_from_positions(_packed_positions(doc_lens))
    segment_ids = segment_ids.expand(2, -1).to(self.accelerator_device)
    x = torch.randn(
        2, seq_len, channels, device=self.accelerator_device
    ).requires_grad_()
    weight = torch.randn(
        channels, 1, 4, device=self.accelerator_device
    ).requires_grad_()

    out = gated_delta._causal_conv(x, weight, segment_ids)  # pylint: disable=protected-access
    starts = [sum(doc_lens[:i]) for i in range(len(doc_lens))]
    expected = torch.cat(
        [
            gated_delta._causal_conv(x[:, s : s + n], weight)  # pylint: disable=protected-access
            for s, n in zip(starts, doc_lens)
        ],
        dim=1,
    )
    tol = 5e-3 if self._is_tpu() else 1e-6
    test_utils.check_equivalence(
        out.detach().cpu(),
        expected.detach().cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Causal Conv Forward",
    )
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (x, weight), grad_out)
    expected_grads = torch.autograd.grad(
        expected, (x, weight), grad_out, retain_graph=True
    )
    for name, grad, expected_grad in zip(
        ("x", "weight"), grads, expected_grads, strict=True
    ):
      test_utils.check_equivalence(
          grad.cpu(),
          expected_grad.cpu(),
          atol=tol,
          rtol=tol,
          check_name=f"Packed Causal Conv {name} Grad",
      )
    # The slice-based backward of causal_conv_silu restarts at documents too.
    out_silu = gated_delta.causal_conv_silu(x, weight, segment_ids)
    expected_silu = F.silu(expected)
    test_utils.check_equivalence(
        out_silu.detach().cpu(),
        expected_silu.detach().cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Causal Conv SiLU Forward",
    )
    grads_silu = torch.autograd.grad(out_silu, (x, weight), grad_out)
    expected_silu_grads = torch.autograd.grad(
        expected_silu, (x, weight), grad_out
    )
    for name, grad, expected_grad in zip(
        ("x", "weight"), grads_silu, expected_silu_grads, strict=True
    ):
      test_utils.check_equivalence(
          grad.cpu(),
          expected_grad.cpu(),
          atol=tol,
          rtol=tol,
          check_name=f"Packed Causal Conv SiLU {name} Grad",
      )
    # Negative control: without segment_ids, the taps cross documents.
    with torch.no_grad():
      leaky = gated_delta._causal_conv(x, weight)  # pylint: disable=protected-access
    self.assertGreater(
        _relative_l2_error(leaky, expected),
        0.1,
        msg="Negative control failed: conv taps did not cross documents",
    )

  def test_causal_conv_adds_bf16_taps_in_fp32(self):
    """Tests that bf16 causal conv1d rounds an fp32 tap sum once, as F.conv1d."""
    torch.manual_seed(0)
    seq_len, channels, kernel_size = 64, 32, 4
    x = torch.randn(2, seq_len, channels).bfloat16()
    weight = torch.randn(channels, 1, kernel_size).bfloat16()
    # The products of bf16 values are exact in fp32; F.conv1d adds them in fp32.
    expected = F.conv1d(
        F.pad(x.float().transpose(1, 2), (kernel_size - 1, 0)),
        weight.float(),
        groups=channels,
    ).transpose(1, 2)
    out = gated_delta._causal_conv(  # pylint: disable=protected-access
        x.to(self.accelerator_device), weight.to(self.accelerator_device)
    ).cpu()
    self.assertEqual(out.dtype, torch.bfloat16)
    # Rounding once to bf16 is off by at most half an ulp, 2**-8 of the value.
    bound = expected.abs() * 2**-8 + 1e-5
    self.assertTrue(
        bool(((out.float() - expected).abs() <= bound).all()),
        msg="bf16 causal conv1d taps did not add in fp32",
    )
    # Negative control: adding the bf16 products in bf16 exceeds the bound.
    x_pad = F.pad(x, (0, 0, kernel_size - 1, 0))
    bf16_sum = sum(
        x_pad[:, k : k + seq_len] * weight[:, 0, k] for k in range(kernel_size)
    )
    self.assertFalse(
        bool(((bf16_sum.float() - expected).abs() <= bound).all()),
        msg="Negative control failed: the bf16 sum is within the bound",
    )

  def test_gated_delta_net_isolates_batch_sequences(self):
    """Tests that B=2 packed sequences in GatedDeltaNet equal two B=1 runs."""
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    gdn = config.layers[0].delta_net.build()
    gdn.init_states(buffer_device=torch.device("cpu"))
    gdn = gdn.to(self.accelerator_device)

    seq_len = 16
    x0 = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    x1 = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    x_b2 = torch.cat([x0, x1], dim=0)

    with torch.no_grad():
      with set_unflatten_seq_len(seq_len):
        out0 = gdn(x0)
        out1 = gdn(x1)
        out_b2 = gdn(x_b2)
      # Negative control: if treated as 1 sequence of length 2*seq_len, the
      # second half changes significantly because conv1d and state bleed across.
      with set_unflatten_seq_len(2 * seq_len):
        out_bleed = gdn(x_b2)

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        out_b2[:seq_len].cpu(),
        out0.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Batch Slice 0",
    )
    test_utils.check_equivalence(
        out_b2[seq_len:].cpu(),
        out1.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Batch Slice 1",
    )
    self.assertGreater(
        _relative_l2_error(out_bleed[seq_len:], out1),
        0.05,
        msg="Negative control failed: GDN bleed did not change slice 1",
    )

  def test_gated_delta_net_isolates_packed_documents(self):
    """Tests that documents packed into GatedDeltaNet rows equal single runs."""
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    gdn = config.layers[0].delta_net.build()
    gdn.init_states(buffer_device=torch.device("cpu"))
    gdn = gdn.to(self.accelerator_device)

    # Two rows of 80 tokens. With 64-token chunks, documents start mid-chunk
    # and at a chunk start (64), and some span two chunks.
    rows = ((30, 34, 16), (50, 30))
    seq_len = 80
    doc_lens = [n for lens in rows for n in lens]
    starts = [sum(doc_lens[:i]) for i in range(len(doc_lens))]
    x = torch.randn(
        len(rows) * seq_len, config.dim, device=self.accelerator_device
    )
    positions = torch.cat([_packed_positions(lens) for lens in rows]).to(
        self.accelerator_device
    )

    with torch.no_grad():
      with set_unflatten_seq_len(seq_len):
        out_packed = gdn(x, positions=positions)
        out_leaky = gdn(x)
      # Without set_unflatten_seq_len, the [T] token stream is one row.
      out_stream = gdn(x, positions=positions)
      expected = []
      for s, n in zip(starts, doc_lens):
        with set_unflatten_seq_len(n):
          expected.append(gdn(x[s : s + n]))
      expected = torch.cat(expected)

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        out_packed.cpu(),
        expected.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Packed Rows",
    )
    test_utils.check_equivalence(
        out_stream.cpu(),
        expected.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Packed Stream",
    )
    # Negative control: without positions, the convolution and the state leak
    # into the first tokens of each document that does not start a row.
    inner_starts = [30, 64, 130]
    self.assertGreater(
        _relative_l2_error(out_leaky[inner_starts], expected[inner_starts]),
        0.1,
        msg="Negative control failed: no GDN leak without positions",
    )

  def test_model_isolates_packed_documents(self):
    """Tests that documents packed into rows equal the documents run alone.

    Covers ``positions`` reaching every layer: the attention masks other
    documents, and the GatedDeltaNet restarts its convolution and state, in the
    forward pass and in every parameter gradient.
    """
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    vocab_size = 512
    config.vocab_size = vocab_size
    config.tok_embeddings.num_embeddings = vocab_size
    config.lm_head.out_features = vocab_size
    config.layers = config.layers[:4]  # 3 GatedDeltaNet + 1 attention layers.
    torch.manual_seed(0)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    # Moderate decays, as in the HF parity test: A_log's init reaches log(16),
    # with which a GatedDeltaNet forgets within a token or two (hiding state
    # leaks) and the fp32 chunk-cumulative decays grow to O(1000).
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
      for name, param in model.named_parameters():
        if "A_log" in name or "dt_bias" in name:
          param.copy_(0.2 * torch.randn(param.shape, generator=gen))
    model = model.to(self.accelerator_device)
    if config.enable_weight_tying:
      model.tok_embeddings.weight = model.lm_head.weight

    rows = ((30, 34, 16), (50, 30))
    seq_len = 80
    doc_lens = [n for lens in rows for n in lens]
    starts = [sum(doc_lens[:i]) for i in range(len(doc_lens))]
    gen = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, vocab_size, (len(rows) * seq_len,), generator=gen)
    loss_weights = torch.randn(len(rows) * seq_len, vocab_size, generator=gen)
    positions = torch.cat([_packed_positions(lens) for lens in rows])

    def forward_backward(tokens, positions, loss_weights, row_len):
      model.zero_grad()
      with set_unflatten_seq_len(row_len):
        logits = model(
            tokens.to(self.accelerator_device),
            positions=positions.to(self.accelerator_device),
        )
      (
          logits.float() * loss_weights.to(self.accelerator_device)
      ).sum().backward()
      grads = {
          name: param.grad.detach().float().cpu()
          for name, param in model.named_parameters()
          if param.grad is not None
      }
      return logits.detach().float().cpu(), grads

    logits_packed, grads_packed = forward_backward(
        tokens, positions, loss_weights, seq_len
    )
    logits_docs, grads_docs = [], {}
    for s, n in zip(starts, doc_lens):
      logits, grads = forward_backward(
          tokens[s : s + n], torch.arange(n), loss_weights[s : s + n], n
      )
      logits_docs.append(logits)
      for name, grad in grads.items():
        grads_docs[name] = grads_docs.get(name, 0) + grad
    logits_docs = torch.cat(logits_docs)

    tol = 5e-2 if self._is_tpu() else 1e-4
    test_utils.check_equivalence(
        logits_packed,
        logits_docs,
        atol=tol,
        rtol=tol,
        check_name="Packed Documents Logits",
    )
    grad_rel_bound = 2e-2 if self._is_tpu() else 5e-4
    self.assertEqual(set(grads_packed), set(grads_docs))
    for name, grad in grads_docs.items():
      self.assertLessEqual(
          _relative_l2_error(grads_packed[name], grad),
          grad_rel_bound,
          msg=f"Packed documents gradient mismatch for {name}",
      )

    # Negative control: positions that run on across documents leak attention
    # and GatedDeltaNet state into every document that does not start a row.
    later = torch.cat([
        torch.arange(s, s + n)
        for s, n in zip(starts, doc_lens)
        if s % seq_len != 0
    ])
    with torch.no_grad(), set_unflatten_seq_len(seq_len):
      logits_unpacked = model(
          tokens.to(self.accelerator_device),
          positions=torch.arange(seq_len)
          .repeat(len(rows))
          .to(self.accelerator_device),
      )
    leak = _relative_l2_error(logits_unpacked.cpu()[later], logits_docs[later])
    self.assertGreater(
        leak,
        0.01,
        msg="Negative control failed: unpacked positions did not leak",
    )

  def test_model_computes_segment_ids_once(self):
    """Tests that the model derives the document ids once for all layers.

    ``Qwen35Model.forward`` computes ``segment_ids_from_positions(positions)``
    for a 1-D token stream and passes the ids to every TPU layer, so the
    GatedDeltaNet layers no longer derive their own. The logits and every
    parameter gradient keep their bit patterns compared with the per-layer
    path.
    """
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    self.addCleanup(torch.set_num_threads, num_threads)
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    vocab_size = 512
    config.vocab_size = vocab_size
    config.tok_embeddings.num_embeddings = vocab_size
    config.lm_head.out_features = vocab_size
    config.layers = config.layers[:4]  # 3 GatedDeltaNet + 1 attention layers.
    torch.manual_seed(0)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    if config.enable_weight_tying:
      model.tok_embeddings.weight = model.lm_head.weight

    rows = ((30, 34, 16), (50, 30))
    seq_len = 80
    gen = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, vocab_size, (len(rows) * seq_len,), generator=gen)
    positions = torch.cat([_packed_positions(lens) for lens in rows])

    def forward_backward():
      model.zero_grad()
      with set_unflatten_seq_len(seq_len):
        logits = model(tokens, positions=positions)
      logits.float().square().sum().backward()
      grads = [
          param.grad.detach().clone()
          for param in model.parameters()
          if param.grad is not None
      ]
      return [logits.detach(), *grads]

    with mock.patch.object(
        qwen3_5_model,
        "segment_ids_from_positions",
        wraps=segment_ids_from_positions,
    ) as in_model, mock.patch.object(
        gated_delta,
        "segment_ids_from_positions",
        wraps=segment_ids_from_positions,
    ) as in_gdn:
      once = forward_backward()
    in_model.assert_called_once()
    in_gdn.assert_not_called()

    with mock.patch.object(
        tpu_model.TpuQwen35TransformerBlock, "accepts_segment_ids", False
    ), mock.patch.object(
        gated_delta,
        "segment_ids_from_positions",
        wraps=segment_ids_from_positions,
    ) as in_gdn_per_layer:
      per_layer = forward_backward()
    # Negative control: without the model's ids, every GatedDeltaNet layer
    # derives them from `positions` itself.
    self.assertEqual(in_gdn_per_layer.call_count, 3)
    self.assertEqual(len(once), len(per_layer))
    for i, (actual, expected) in enumerate(zip(once, per_layer, strict=True)):
      self.assertTrue(
          torch.equal(
              actual.contiguous().view(torch.uint8),
              expected.contiguous().view(torch.uint8),
          ),
          msg=f"output {i} differs from the per-layer path",
      )

  def test_gated_delta_net_uses_given_segment_ids(self):
    """Tests ``TpuInnerGatedDeltaNet`` with ``segment_ids`` from the model.

    Given ``segment_ids``, the layer uses them instead of deriving them from
    ``positions`` (same output), and it rejects ids that do not cover every
    token.
    """
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    gdn = config.layers[0].delta_net.build()
    gdn.init_states(buffer_device=torch.device("cpu"))
    seq_len = 32
    torch.manual_seed(0)
    x = torch.randn(seq_len, config.dim)
    positions = _packed_positions((10, 22))
    with torch.no_grad(), set_unflatten_seq_len(seq_len):
      from_positions = gdn(x, positions=positions)
      with mock.patch.object(
          gated_delta,
          "segment_ids_from_positions",
          wraps=segment_ids_from_positions,
      ) as derive:
        given = gdn(
            x,
            positions=positions,
            segment_ids=segment_ids_from_positions(positions),
        )
      derive.assert_not_called()
      self.assertTrue(torch.equal(given, from_positions))
      with self.assertRaisesRegex(ValueError, "one segment id per token"):
        gdn(
            x,
            positions=positions,
            segment_ids=segment_ids_from_positions(positions)[:-1],
        )

  def test_attention_isolates_packed_documents(self):
    """Tests that TpuQwen35Attention isolates packed documents via positions."""
    config = qwen3_5_tpu.model_registry("debugmodel", seq_len=128).model
    attn = config.layers[3].attention.build()
    attn.init_states(buffer_device=torch.device("cpu"))
    attn = attn.to(self.accelerator_device)

    len_a, len_b = 7, 9
    seq_len = len_a + len_b
    x = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    positions_packed = torch.cat([
        torch.arange(len_a, device=self.accelerator_device),
        torch.arange(len_b, device=self.accelerator_device),
    ])
    positions_unpacked = torch.arange(seq_len, device=self.accelerator_device)

    with torch.no_grad():
      with set_unflatten_seq_len(seq_len):
        out_packed = attn(x, None, positions_packed)
        out_unpacked = attn(x, None, positions_unpacked)
      with set_unflatten_seq_len(len_a):
        out_a = attn(
            x[:len_a],
            None,
            torch.arange(len_a, device=self.accelerator_device),
        )
      with set_unflatten_seq_len(len_b):
        out_b = attn(
            x[len_a:],
            None,
            torch.arange(len_b, device=self.accelerator_device),
        )

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        out_packed[:len_a].cpu(),
        out_a.cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Doc A",
    )
    test_utils.check_equivalence(
        out_packed[len_a:].cpu(),
        out_b.cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Doc B",
    )
    self.assertGreater(
        _relative_l2_error(out_unpacked[len_a:], out_b),
        0.1,
        msg="Negative control failed: unpacked positions did not change Doc B",
    )

  def test_splash_attention_matches_sdpa_on_packed_documents(self):
    """Tests Splash Attention patch on TpuQwen35Attention vs CPU SDPA."""
    if not self._is_tpu():
      self.skipTest("Splash Attention is a TPU-only Pallas kernel.")
    config = qwen3_5_tpu.model_registry("debugmodel").model
    batch, seq_len = 2, 256
    attn_cpu = config.layers[3].attention.build()
    attn_cpu.init_states(buffer_device=torch.device("cpu"))
    attn_cpu = attn_cpu.cpu()
    attn_tpu = copy.deepcopy(attn_cpu).to(self.accelerator_device)

    sa_config = SplashAttentionKernelConfig(
        use_splash_attention_kernel=True,
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_block_q_dkv=128,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=128,
        sa_block_q_dq=128,
        sa_block_kv_dq=128,
    )
    workarounds.use_splash_attention_patch(attn_tpu, sa_config)

    x_cpu, x_tpu = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    positions_cpu = torch.cat([
        torch.arange(100),
        torch.arange(seq_len - 100),
        torch.arange(seq_len),
    ])
    positions_tpu = positions_cpu.to(self.accelerator_device)

    with set_unflatten_seq_len(seq_len):
      out_cpu = attn_cpu(x_cpu, None, positions_cpu)
      out_tpu = attn_tpu(x_tpu, None, positions_tpu)
      with torch.no_grad():
        out_cpu_unpacked = attn_cpu(
            x_cpu, None, torch.arange(seq_len).repeat(batch)
        )

    self.assertGreater(_relative_l2_error(out_cpu_unpacked, out_cpu), 0.1)
    self.assertLessEqual(_relative_l2_error(out_tpu, out_cpu), 1e-2)

    grad_out = torch.randn_like(out_cpu)
    out_cpu.backward(grad_out)
    out_tpu.backward(grad_out.to(self.accelerator_device))
    self.assertLessEqual(_relative_l2_error(x_tpu.grad, x_cpu.grad), 2.5e-2)

  def test_flavors_and_registry(self):
    """Tests the 35B-A3B geometry and the TPU modules of every flavor."""
    moe_35b = qwen3_5_tpu.model_registry("35B-A3B").model
    self.assertLen(moe_35b.layers, 40)
    self.assertEqual(moe_35b.dim, 2048)
    self.assertEqual(moe_35b.vocab_size, 248320)
    attn_layers = [l for l in moe_35b.layers if l.attention is not None]
    self.assertLen(attn_layers, 10)
    self.assertEqual(attn_layers[0].attention.head_dim, 256)
    self.assertEqual(attn_layers[0].attention.n_heads, 16)
    self.assertEqual(attn_layers[0].attention.n_kv_heads, 2)
    self.assertEqual(moe_35b.layers[0].moe.num_experts, 256)
    self.assertEqual(moe_35b.layers[0].moe.router.top_k, 8)

    for flavor in ("debugmodel", "debugmodel_moe", "35B-A3B"):
      with self.subTest(flavor=flavor):
        spec = qwen3_5_tpu.model_registry(flavor)
        self.assertIs(spec.parallelize_fn, parallelize_qwen3_5)
        self.assertIsNone(spec.pipelining_fn)
        self.assertIsNone(spec.model.vision_encoder)
        for layer in spec.model.layers:
          self.assertIsInstance(
              layer, tpu_model.TpuQwen35TransformerBlock.Config
          )
          if layer.attention is not None:
            self.assertIsInstance(
                layer.attention, tpu_model.TpuQwen35Attention.Config
            )
            self.assertIsInstance(
                layer.attention.inner_attention,
                SlidingWindowScaledDotProductInnerAttention.Config,
            )
          else:
            self.assertIsInstance(
                layer.delta_net, gated_delta.TpuGatedDeltaNet.Config
            )
            self.assertIsInstance(
                layer.delta_net.inner_gated_delta_net,
                gated_delta.TpuInnerGatedDeltaNet.Config,
            )
            for conv in (
                layer.delta_net.conv_q,
                layer.delta_net.conv_k,
                layer.delta_net.conv_v,
            ):
              self.assertIn("weight", conv.param_init or {})
          if layer.moe is not None:
            self.assertIsNone(layer.moe.load_balance_coeff)

    with self.assertRaisesRegex(ValueError, "attn_backend='sdpa'"):
      qwen3_5_tpu.model_registry("debugmodel", attn_backend="flex")
    with self.assertRaisesRegex(ValueError, "exceeds max context length"):
      qwen3_5_tpu.model_registry("debugmodel", seq_len=262145)

  def _assert_uses_tpu_moe_kernels(self, model_config) -> None:
    """Asserts that `_use_tpu_moe_kernels` reached every MoE layer."""
    num_moe = sum(layer.moe is not None for layer in model_config.layers)
    experts = [
        c for _, c, _, _ in model_config.traverse(TpuGroupedExperts.Config)
    ]
    dispatchers = [
        c
        for _, c, _, _ in model_config.traverse(
            TpuAllToAllTokenDispatcher.Config
        )
    ]
    # A traversal that stops matching would silently drop the kernels.
    self.assertGreater(num_moe, 0)
    self.assertLen(experts, num_moe)
    self.assertLen(dispatchers, num_moe)
    for experts_config in experts:
      self.assertTrue(experts_config.use_pallas_grouped_mm)
      self.assertTrue(experts_config.use_pallas_paired_grouped_mm)
      self.assertTrue(experts_config.use_kernel_native_weight_layout)
      self.assertEqual(experts_config.pallas_grouped_mm_tile_m, 256)
      self.assertEqual(
          experts_config.pallas_grouped_mm_tgmm_tiling, (256, 0, 0)
      )
    for dispatcher in dispatchers:
      self.assertTrue(dispatcher.fuse_route_score_backward)

  def test_tpu_moe_kernels_debug_recipe(self):
    """Tests that the debug recipe selects the 35B recipe's expert kernels."""
    cfg = config_registry.qwen35_moe_debugmodel_tpu_moe_kernels()
    self._assert_uses_tpu_moe_kernels(cfg.model_spec.model)

  def test_35b_recipe(self):
    """Tests the 35B-A3B recipe, also as parsed from the command line."""
    cfg = config_registry.qwen35_moe_35b()
    self.assertEqual(cfg.model_spec.flavor, "35B-A3B")
    self.assertEqual(cfg.training.dtype, "float32")
    self.assertEqual(cfg.training.mixed_precision_param, "bfloat16")
    self.assertEqual(cfg.training.mixed_precision_reduce, "float32")
    self.assertTrue(cfg.parallelism.use_simple_fsdp)
    self.assertEqual(cfg.parallelism.fsdp_reshard_after_forward, "always")
    self.assertTrue(cfg.compile.compile_grad_norm)
    self.assertEqual(cfg.tpu_config.eager_mode, "DEFER_AND_FUSE")
    self._assert_uses_tpu_moe_kernels(cfg.model_spec.model)
    splash = cfg.splash_attention_kernel
    self.assertTrue(splash.use_splash_attention_kernel)
    self.assertEqual(
        (
            splash.sa_block_q,
            splash.sa_block_kv,
            splash.sa_block_q_dkv,
            splash.sa_block_kv_dkv,
            splash.sa_block_q_dq,
            splash.sa_block_kv_dq,
        ),
        (1024,) * 6,
    )
    self.assertIsInstance(cfg.loss, PallasCrossEntropyLoss.Config)
    self.assertIn(config_registry.FUSED_LOSS_OVERRIDE, cfg.override.imports)
    self.assertEqual(
        cfg.libtpu_init_flags, {"xla_tpu_scoped_vmem_limit_kib": "65472"}
    )
    with torch.device("meta"):
      model = cfg.model_spec.model.build()
    self.assertLen(model.layers, 40)
    self.assertIsInstance(
        model.layers["0"].moe.routed_experts.inner_experts, TpuGroupedExperts
    )

    parsed = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3_5",
        "--config=qwen35_moe_35b",
        "--training.num_tokens_per_microbatch_per_dp_rank=8192",
    ])
    self.assertEqual(parsed.model_spec.flavor, "35B-A3B")
    self.assertEqual(
        parsed.training.num_tokens_per_microbatch_per_dp_rank, 8192
    )
    self.assertIsInstance(parsed.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(
        parsed.libtpu_init_flags, {"xla_tpu_scoped_vmem_limit_kib": "65472"}
    )


if __name__ == "__main__":
  absltest.main()
