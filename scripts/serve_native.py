#!/usr/bin/env python3
"""Official native policy/kernels with explicit, attested denoising settings."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/lingbot-vla-v2"))
# Do not call training_runtime.add_vendor_path: native serving must retain the
# official fused-MoE implementation, rather than the training reference backend.
from lrvla.inference_profile import apply_inference_profile  # noqa: E402


def main():
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, str2bool
    from deploy.websocket_policy_server import WebsocketPolicyServer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=9330)
    parser.add_argument("--use_length", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--use_bf16", type=str2bool, default=False)
    parser.add_argument("--use_fp32", type=str2bool, default=True)
    parser.add_argument("--use_compile", type=str2bool, default=False)
    args = parser.parse_args()
    if args.use_bf16 == args.use_fp32 or args.num_steps <= 0 or args.use_length <= 0:
        parser.error("Select one precision and positive use_length/num_steps.")
    policy = LingbotVLAv2Server(args.model_path, use_length=args.use_length, chunk_ret=True,
                               use_bf16=args.use_bf16, use_fp32=args.use_fp32, use_compile=args.use_compile)
    apply_inference_profile(policy, args.num_steps)
    WebsocketPolicyServer(policy, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
