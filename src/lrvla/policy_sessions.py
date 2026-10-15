"""Serialize shared-model calls while isolating each scene's Torch RNG stream."""
from __future__ import annotations

import hashlib
import json

POLICY_RNG = "scene_seeded_torch_v1"


def scene_seed(task, setting, seed):
    payload = json.dumps([POLICY_RNG, task, setting, seed], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


class PolicySession:
    """One websocket's random stream; calls stay synchronous on one event loop.

    The underlying policy is shared, including its feature transform. Only one
    fixed robot is allowed and full chunks must be recomputed on every request.
    This wrapper must never be called concurrently from different threads.
    """

    def __init__(self, policy):
        if not policy.chunk_ret:
            raise ValueError("Shared inference requires chunk_ret=True.")
        import torch
        self.torch, self.policy = torch, policy
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.cpu_state = self.cuda_state = None

    def infer(self, observation):
        observation = dict(observation)
        if "path_to_pi_model" in observation:
            raise ValueError("A shared service cannot change checkpoint.")
        reset = observation.get("reset", False)
        if reset and (observation.get("robo_name") != "robotwin" or type(observation.get("rng_seed")) is not int
                      or not 0 <= observation["rng_seed"] < 2**63):
            raise ValueError("Reset requires robotwin and a nonnegative scene rng_seed < 2**63.")
        if not reset and self.cpu_state is None:
            raise ValueError("Reset this connection with a scene seed before inference.")
        torch = self.torch
        with torch.random.fork_rng(devices=[] if self.device is None else [self.device]):
            if reset:
                torch.manual_seed(observation.pop("rng_seed"))
            else:
                torch.set_rng_state(self.cpu_state)
                if self.device is not None:
                    torch.cuda.set_rng_state(self.cuda_state, self.device)
            result = self.policy.infer(observation)
            self.cpu_state = torch.get_rng_state()
            if self.device is not None:
                self.cuda_state = torch.cuda.get_rng_state(self.device)
        return result


def policy_server(policy, port, *, scene_rng=False):
    from deploy.websocket_policy_server import WebsocketPolicyServer
    if not scene_rng:
        return WebsocketPolicyServer(policy, port=port)

    class SharedServer(WebsocketPolicyServer):
        async def _handler(self, websocket):
            # A lightweight delegate owns connection state, never model weights.
            delegate = WebsocketPolicyServer(PolicySession(policy), metadata={"policy_rng": POLICY_RNG,
                                             "chunk_ret": True, "robot": "robotwin"})
            await delegate._handler(websocket)

    return SharedServer(policy, port=port)
