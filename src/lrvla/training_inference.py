"""Low host-RAM policy server with official preprocessing/WebSocket interfaces."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from .training_runtime import add_vendor_path, build_policy
from .training_checkpoint import run_relative_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
add_vendor_path(PROJECT_ROOT)
from deploy.lingbot_vla_v2_policy import (  # noqa: E402
    LingbotVLAv2Server as OfficialServer,
    LingBotVlaV2InferencePolicy,
    str2bool,
)


class LingbotVLAv2Server(OfficialServer):
    """Compatible policy object; frozen weights stream directly to the GPU.

    Local BF16 SDPA results are a resource-limited baseline. The official release
    reports FP32 benchmark inference, whose storage alone can exceed laptop VRAM.
    """

    def __init__(self, path_to_pi_model="", robot_norm_path=None, adaptive_ensemble_alpha=0.1,
                 action_ensemble_horizon=8, use_length=1, chunk_ret=False, use_bf16=True,
                 use_fp32=False, use_compile=False):
        if use_bf16 == use_fp32:
            raise ValueError("Select exactly one of use_bf16 or use_fp32.")
        if use_compile:
            raise ValueError("The local streamed reference server requires --use_compile false; use the native server for compiled kernels.")
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.use_length, self.chunk_ret = use_length, chunk_ret
        self.robot_norm_path, self.task_description = robot_norm_path, None
        self.use_bf16, self.use_fp32, self.use_compile = use_bf16, use_fp32, False
        self.global_step, self.last_action_chunk, self.last_normalized_action_chunk = 0, None, None
        self.action_key = "action"
        self.vla = self.load_vla(path_to_pi_model).eval()

    def load_vla(self, path_to_pi_model):
        checkpoint = Path(path_to_pi_model).expanduser().resolve()
        run = checkpoint.parent.parent.parent
        cli_path = run / "lingbotvla_cli.yaml"
        if not cli_path.is_file():
            raise FileNotFoundError(f"Expected official run layout: {cli_path}")
        cli = yaml.safe_load(cli_path.read_text())
        qwen_override = os.environ.get("QWEN3VL_PATH")
        qwen = Path(qwen_override).expanduser().resolve() if qwen_override else run_relative_path(run, cli["model"]["tokenizer_path"])
        dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        model = build_policy(PROJECT_ROOT, cli, checkpoint, qwen, policy_class=LingBotVlaV2InferencePolicy,
                             weight_dtype=dtype, device="cuda")
        model.config.use_cache = True
        model.config.action_fp32 = True
        model.feature_transform = None
        model.model._use_compile_predict_velocity = False
        model.model._compiled_predict_velocity = None
        self.vla, self.config, self.model_name = model, model.config, "qwen3vl"
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(qwen, padding_side="right", local_files_only=True)
        self.language_tokenizer = self.processor.tokenizer
        data = dict(cli["data"])
        for key in ("norm_stats_file", "robot_config_root", "train_path"):
            if key in data:
                data[key] = str(run_relative_path(run, data[key]))
        self.data_config = SimpleNamespace(**data)
        self.sample_actions_fn = model.model.sample_actions
        self.robot_config_root = Path(data.get("robot_config_root", run / "configs" / "robot_configs"))
        if self.robot_norm_path is None:
            self.robot_norm_path = data["norm_stats_file"]
        print(json.dumps({"checkpoint": str(checkpoint), "precision": str(dtype), "attention": "pytorch_sdpa",
                          "moe": "torch_reference_fused_storage", "compile": False,
                          "official_fp32_benchmark_equivalence_verified": False}), flush=True)
        return model

    def reset(self, robo_name, path_to_pi_model=None):
        if path_to_pi_model is not None:
            self.vla = self.load_vla(path_to_pi_model).eval()
        self.global_step, self.last_action_chunk, self.last_normalized_action_chunk = 0, None, None
        robot_path = self.robot_config_root / f"{robo_name}.yaml"
        self.robot_config = yaml.safe_load(robot_path.read_text())
        from lingbotvla.data.vla_data.utils import FeatureTransform
        self.vla.feature_transform = FeatureTransform(str(robot_path), self.data_config, self.config,
                                                       self.processor, chunk_size=self.config.chunk_size,
                                                       norm_stats_path=self.robot_norm_path)
        self.action_key = self.vla.feature_transform.org_features["actions"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--use_length", type=int, default=50)
    parser.add_argument("--chunk_ret", type=str2bool, default=True)
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    parser.add_argument("--use_fp32", type=str2bool, default=False)
    parser.add_argument("--use_compile", type=str2bool, default=False)
    args = parser.parse_args(argv)
    model = LingbotVLAv2Server(args.model_path, use_length=args.use_length, chunk_ret=args.chunk_ret,
                             use_bf16=args.use_bf16, use_fp32=args.use_fp32, use_compile=args.use_compile)
    from deploy.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(model, port=args.port).serve_forever()
