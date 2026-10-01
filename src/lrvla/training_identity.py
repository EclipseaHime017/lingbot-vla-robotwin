"""Location-independent identities for frozen weights and audited clean inputs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_content_sha256(manifest):
    """Keep data/source/split/order identity while allowing directory relocation."""
    content = deepcopy(manifest)
    content.pop("training_list", None)
    for entry in content.get("datasets", []):
        entry.pop("path", None)
    return canonical_sha256(content)


def base_checkpoint_identity(checkpoint):
    """Hash the actual frozen shard bytes, without trusting a local directory name."""
    files = sorted(Path(checkpoint).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No frozen safetensors shards in {checkpoint}")
    inventory = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        inventory.append({"file": path.name, "size": path.stat().st_size, "sha256": digest.hexdigest()})
    return {"sha256": canonical_sha256(inventory), "shards": inventory}
