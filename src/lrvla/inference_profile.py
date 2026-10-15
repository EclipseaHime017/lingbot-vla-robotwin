"""Apply and attest inference settings to the object used by sample_actions."""
import json


def apply_inference_profile(policy, num_steps):
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    config = policy.vla.model.config
    if not 1 <= policy.use_length <= config.chunk_size:
        raise ValueError("use_length must be within the model action chunk.")
    config.num_steps = num_steps
    if policy.vla.model.config.num_steps != num_steps:
        raise RuntimeError("Denoising step configuration did not reach the sampling model.")
    print(json.dumps({"effective_inference": {"num_steps": config.num_steps, "use_length": policy.use_length,
                                              "compile": policy.use_compile, "action_smoothing": "none"}}), flush=True)
