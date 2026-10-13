"""Two-rank FULL_SHARD proof for the upstream two-stage decoder interface."""
from datetime import timedelta
from functools import partial
import json
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from lrvla.training_checkpoint import load_safetensors_streamed
from lrvla.training_distributed_checkpoint import (
    export_distributed_hf, resume_distributed_checkpoint, save_distributed_checkpoint,
)
from lrvla.training_fsdp import BCForward, fsdp_shard_report, wrap_expert_fsdp
from lrvla.training_models import configure_trainable, flow_matching_bc, sdpa_attention
from lrvla.training_runtime import add_vendor_path, enable_action_checkpointing
from lrvla.training_smoke import TinyConfig


class FrozenPrefixDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8)
        self.k, self.v = nn.Linear(8, 4), nn.Linear(8, 4)
        self.o = nn.Linear(8, 8)

    def forward(self, hidden, attention=None, start=None, end=None,
                compute_kqv=False, output_atten=False):
        hidden = hidden.to(self.q.weight.dtype)
        if compute_kqv:
            b, s, _ = hidden.shape
            return self.q(hidden).view(b, s, 4, 2), self.k(hidden).view(b, s, 2, 2), self.v(hidden).view(b, s, 2, 2)
        assert output_atten
        return hidden + torch.tanh(self.o(attention[:, start:end].to(self.o.weight.dtype)))


class CoupledFlow(nn.Module):
    def __init__(self):
        super().__init__()
        add_vendor_path(Path(__file__).resolve().parents[1])
        from transformers import Qwen2Config
        from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2DecoderLayer
        from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import QwenvlWithExpertV2Model
        action_config = Qwen2Config(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2,
                                   use_sliding_window=False, attention_dropout=0)
        action_config._attn_implementation = "eager"
        coupled = QwenvlWithExpertV2Model.__new__(QwenvlWithExpertV2Model)
        nn.Module.__init__(coupled)
        coupled.config = SimpleNamespace(attention_implementation="eager", final_norm_adanorm=False,
                                         qwen_expert_config=action_config,
                                         freeze_vision_encoder=True, train_expert_only=True)
        coupled.qwenvl = nn.Module()
        coupled.qwenvl.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2,
                                                                           num_attention_heads=4))
        coupled.qwenvl.visual = nn.Linear(8, 8)
        coupled.qwenvl.model = nn.Module()
        language = coupled.qwenvl.model.language_model = nn.Module()
        language.embed_tokens = nn.Embedding(16, 8)
        language.layers = nn.ModuleList([FrozenPrefixDecoder(), FrozenPrefixDecoder()])
        language.norm = nn.LayerNorm(8)
        coupled.qwen_expert = nn.Module()
        expert = coupled.qwen_expert.model = nn.Module()
        expert.layers = nn.ModuleList([Qwen2DecoderLayer(action_config, index) for index in range(2)])
        expert.norm = nn.LayerNorm(8)
        coupled.apply_mrope = lambda q, k, positions: (q, k)
        coupled.attention_interface = sdpa_attention
        self.qwenvl_with_expert = coupled
        self.state_proj, self.action_in_proj, self.action_out_proj = nn.Linear(2, 8), nn.Linear(2, 8), nn.Linear(8, 2)
        self.action_time_mlp_in, self.action_time_mlp_out = nn.Linear(16, 8), nn.Linear(8, 8)
        self.config = SimpleNamespace(n_action_steps=2, loss_type="fm", adanorm_time=False)

    def sample_time(self, batch_size, device):
        return torch.full((batch_size,), 0.4, device=device)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw):
        vlm = self.qwenvl_with_expert.qwenvl
        prefix = torch.cat([vlm.visual(images.to(vlm.visual.weight.dtype)),
                            vlm.model.language_model.embed_tokens(lang_tokens)], dim=1)
        pad = torch.cat([img_masks, lang_masks], dim=1)
        positions = torch.zeros(3, *pad.shape, device=prefix.device, dtype=torch.long)
        return prefix, pad, torch.zeros_like(pad), positions, None, None

    def embed_suffix(self, state, actions, time):
        embedded = self.action_in_proj(actions)
        time_emb = time[:, None, None].expand_as(embedded)
        embedded = embedded + self.action_time_mlp_out(torch.tanh(
            self.action_time_mlp_in(torch.cat([embedded, time_emb], dim=-1))))
        suffix = torch.cat([self.state_proj(state)[:, None], embedded], dim=1)
        pad = torch.ones(suffix.shape[:2], dtype=torch.bool, device=state.device)
        ar = torch.tensor([[True, True, False]], device=state.device).expand_as(pad)
        return time_emb, suffix, pad, ar

    @staticmethod
    def _build_full_position_ids(prefix_pos, prefix_pad, suffix_pad):
        return torch.zeros(3, prefix_pad.shape[0], prefix_pad.shape[1] + suffix_pad.shape[1],
                           dtype=torch.long, device=prefix_pad.device)

    @staticmethod
    def _block_suffix_to_future_video_if_enabled_(mask, **kwargs):
        return mask

    @staticmethod
    def _fp32_linear(module, x):
        return torch.nn.functional.linear(x.float(), module.weight.float(), module.bias.float())


class CoupledPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CoupledFlow()
        self.config = TinyConfig(model_type="synthetic_smoke", action_dim=2)


def _policy(device):
    torch.manual_seed(10)
    policy = CoupledPolicy().to(device=device, dtype=torch.bfloat16)
    selection = configure_trainable(policy, {"mode": "expert", "expert_last_n_layers": -1})
    enable_action_checkpointing(policy)
    return policy, selection


def _wrap(policy, device, layer_sharding):
    if device.type == "cuda":
        return wrap_expert_fsdp(policy, device=device, layer_sharding=layer_sharding)
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2DecoderLayer
    options = {}
    if layer_sharding:
        options["auto_wrap_policy"] = partial(transformer_auto_wrap_policy,
                                               transformer_layer_cls={Qwen2DecoderLayer})
    return FSDP(BCForward(policy), sharding_strategy=ShardingStrategy.FULL_SHARD,
                ignored_states=[p for p in policy.parameters() if not p.requires_grad],
                use_orig_params=True, device_id=device, limit_all_gathers=True, **options)


def _full_parameters(model, policy, *, gradients=False):
    with FSDP.summon_full_params(model, recurse=True, writeback=False, with_grads=gradients):
        return {name.replace("_fsdp_wrapped_module.", ""): (p.grad if gradients else p).detach().cpu().clone()
                for name, p in policy.named_parameters() if not gradients or p.grad is not None}


def _worker(rank, world_size, directory, device_name):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    torch.set_num_threads(1)
    directory = Path(directory)
    device = torch.device("cuda", 0) if device_name == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=90))
    try:
        reference, selection = _policy(device)
        torch.manual_seed(100 + rank)
        observations = {
            "actions": torch.randn(2, 2, 2, device=device), "state": torch.randn(2, 2, device=device),
            "images": torch.randn(2, 3, 8, device=device),
            "img_masks": torch.ones(2, 3, device=device, dtype=torch.bool),
            "lang_tokens": torch.tensor([[1, 2], [3, 4]], device=device),
            "lang_masks": torch.ones(2, 2, device=device, dtype=torch.bool),
            "image_grid_thw": torch.ones(2, 3, device=device, dtype=torch.long),
            "joint_mask": torch.ones(2, 2, 2, device=device, dtype=torch.bool),
        }
        reference_optimizer = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=1e-3,
                                               foreach=False)
        torch.manual_seed(300 + rank)
        flow_matching_bc(reference, observations, reduction="sum").backward()
        expected_gradients = {}
        for name, parameter in reference.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world_size)
                expected_gradients[name] = parameter.grad.cpu().clone()
        reference_optimizer.step()
        expected_parameters = {name: p.detach().cpu().clone() for name, p in reference.named_parameters()}
        reports = []
        for layer_sharding in (False, True):
            policy, selection = _policy(device)
            model = _wrap(policy, device, layer_sharding)
            assert model.sharding_strategy == ShardingStrategy.FULL_SHARD
            if layer_sharding:
                assert sum(isinstance(module, FSDP) for module in model.modules()) == 3
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, foreach=False)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            torch.manual_seed(300 + rank)
            model(observations, reduction="sum").backward()
            gradients = _full_parameters(model, policy, gradients=True)
            for name, value in expected_gradients.items():
                torch.testing.assert_close(gradients[name], value, rtol=1e-5, atol=1e-6)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            actual_parameters = _full_parameters(model, policy)
            for name, value in expected_parameters.items():
                torch.testing.assert_close(actual_parameters[name], value, rtol=1e-5, atol=1e-6)
            if layer_sharding:
                metadata = {"base_identity": "tiny-fixed-base", "manifest_content_sha256": "tiny-clean",
                            "selection": selection, "data_sampling": {"world_size": world_size}}
                checkpoint = directory / "checkpoint"
                save_distributed_checkpoint(model, optimizer, scheduler, checkpoint, {"global_step": 1}, metadata,
                                            rank=rank, world_size=world_size, device=device)
                # CPU FSDP is useful for sharding correctness, but offloading
                # its already-CPU flat storage can leave invalid full-state
                # views in libtorch. Exercise production CPU-offload exports
                # with CUDA FSDP, where offload creates independent CPU copies.
                if device.type == "cuda":
                    export = export_distributed_hf(model, policy, directory / "hf_ckpt", dtype=torch.bfloat16,
                                                    processor=None, rank=rank)
                    if rank == 0:
                        # Constructing the reference reload resets Torch's RNG;
                        # preserve the training noise stream across this check.
                        with torch.random.fork_rng(devices=[device.index]):
                            reloaded, _ = _policy(torch.device("cpu"))
                            load_safetensors_streamed(reloaded, export, dtype=torch.bfloat16, device="cpu")
                            for name, parameter in reloaded.named_parameters():
                                torch.testing.assert_close(parameter, actual_parameters[name].to(torch.bfloat16), rtol=0, atol=0)
                model(observations, reduction="sum").backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                advanced_parameters = _full_parameters(model, policy)
                restored_policy, _ = _policy(device)
                restored = _wrap(restored_policy, device, True)
                restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],
                                                       lr=1e-3, foreach=False)
                restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1)
                state = resume_distributed_checkpoint(restored, restored_optimizer, restored_scheduler, checkpoint,
                                                      expected_metadata=metadata, rank=rank,
                                                      world_size=world_size, device=device)
                assert state == {"global_step": 1}
                restored(observations, reduction="sum").backward()
                restored_optimizer.step()
                restored_scheduler.step()
                restored_parameters = _full_parameters(restored, restored_policy)
                for name, parameter in restored_parameters.items():
                    torch.testing.assert_close(parameter, advanced_parameters[name], rtol=1e-5, atol=1e-6)
            reports.append({"layer_sharding": layer_sharding, "full_shard": True,
                            "upstream_decoder_twice_with_checkpoint": True, "reference_gradients_and_adam": True})
        if rank == 0:
            (directory / "proof.json").write_text(json.dumps({"world_size": world_size, "device": device_name,
                                                            "cases": reports, "dcp_resume": True,
                                                            "hf_export": device.type == "cuda"}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_two_rank_full_shard_upstream_layers_and_exact_checkpoint(tmp_path, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("The CUDA FSDP export proof requires a CUDA device.")
    try:
        torch.multiprocessing.spawn(_worker, args=(2, str(tmp_path), device_name), nprocs=2, join=True)
    except torch.multiprocessing.ProcessRaisedException as error:
        if "Cannot resolve 127.0.0.1" in str(error):
            pytest.skip("The execution sandbox blocks Gloo's local network interface.")
        raise
    proof = json.loads((tmp_path / "proof.json").read_text())
    assert proof["world_size"] == 2 and proof["dcp_resume"]
    assert proof["hf_export"] == (device_name == "cuda")


def test_layer_shard_report_excludes_frozen_vlm_and_counts_root_projections():
    policy, _ = _policy(torch.device("cpu"))
    whole = fsdp_shard_report(policy, layer_sharding=False)
    layers = fsdp_shard_report(policy, layer_sharding=True)
    assert whole["root_trainable_parameters"] == whole["total_trainable_parameters"]
    assert layers["wrapped_decoder_layers"] == 2
    assert (sum(layers["decoder_layer_trainable_parameters"]) + layers["root_trainable_parameters"]
            == whole["total_trainable_parameters"])
    assert 0 < layers["largest_shard_parameters"] < whole["largest_shard_parameters"]
