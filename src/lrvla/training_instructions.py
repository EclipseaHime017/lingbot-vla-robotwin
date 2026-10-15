"""Optional, episode-local resampling of released clean seen instructions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .training_identity import canonical_sha256


def load_seen_pool(path, manifest):
    pool = json.loads(Path(path).read_text())
    if pool.get("schema") != "clean_episode_seen_v1" or pool.get("clean_only") is not True:
        raise ValueError("Expected a clean episode-local seen instruction sidecar.")
    for entry in manifest["datasets"]:
        task = pool["tasks"].get(entry["task"], {})
        if task.get("source_sha256") != entry["source"]["sha256"]:
            raise ValueError(f"Seen instruction source mismatch: {entry['task']}")
        for episode in entry["episode_ids"]:
            texts = task.get("episodes", {}).get(str(episode))
            if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
                raise ValueError(f"Missing seen instructions: {entry['task']}/{episode}")
    return pool, canonical_sha256(pool)


def sample_seen(pool, task, episode, frame, *, seed, epoch):
    texts = pool["tasks"][task]["episodes"][str(episode)]
    digest = hashlib.sha256(f"{seed}:{epoch}:{task}:{episode}:{frame}".encode()).digest()
    return texts[int.from_bytes(digest[:8], "little") % len(texts)]


def set_dataset_epoch(dataset, epoch):
    for child in getattr(dataset, "datasets", [dataset]):
        child.epoch = epoch


def sampling_identity(config, seen_hash=None):
    return {"image_augment": {"enabled": bool(config.get("image_augment", False)),
                              "implementation": "upstream_shared_photometric_v1",
                              "brightness": [0.7, 1.3], "contrast": [0.6, 1.4], "saturation": [0.5, 1.5]},
            "seen_instruction_resampling": {"enabled": bool(config.get("resample_seen_instructions", False)),
                                            "algorithm": "episode_frame_epoch_sha256_v1", "content_sha256": seen_hash}}
