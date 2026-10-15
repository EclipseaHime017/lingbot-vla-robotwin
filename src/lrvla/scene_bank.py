"""Opt-in expert selection cache with instruction/RNG and initial-RGB checks.

Replay is experimental until a rendered record/replay comparison passes on the
target simulator. It never caches policy actions or policy success outcomes.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity(robotwin):
    root = Path(robotwin)
    files = {}
    for directory in ("envs", "task_config", "description"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".yml", ".yaml", ".urdf", ".srdf"}:
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assets = []
    for path in sorted((root / "assets").rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(root))
            if path.suffix in {".json", ".yml", ".yaml", ".urdf", ".srdf"}:
                files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                stat = path.stat()
                assets.append((relative, stat.st_size, stat.st_mtime_ns))
    for name in ("evaluation.py", "evaluation_runtime.py", "scene_metadata.py", "scene_bank.py"):
        files["adapter/" + name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    if not files:
        raise ValueError("Cannot identify simulator sources for expert cache.")
    return {"schema": "expert-scene-bank-v1", "sources_sha256": digest(files),
            "asset_inventory_sha256": digest(assets), "asset_identity": "config_sha256_and_payload_size_mtime"}


def nested_tuples(value):
    return tuple(nested_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


def numpy_state():
    import numpy as np
    name, state, pos, gauss, cached = np.random.get_state()
    return [name, state.tolist(), pos, gauss, cached]


def observation_identity(observation):
    import numpy as np
    values = [observation["joint_action"]["vector"]]
    values.extend(observation["observation"][key]["rgb"] for key in ("head_camera", "left_camera", "right_camera"))
    result = hashlib.sha256()
    for value in values:
        array = np.asarray(value)
        result.update(str((array.dtype.str, array.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


class SceneBank:
    def __init__(self, directory, identity, mode, *, task, setting, trials, start_seed):
        if mode not in {"record", "replay"}:
            raise ValueError("Scene cache mode must be record or replay.")
        self.mode = mode
        self.identity = {**identity, "task": task, "setting": setting, "trials": trials, "start_seed": start_seed}
        folder = Path(directory) / digest(self.identity)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "scenes.json"
        self.lock = (folder / "scenes.lock").open("a")
        try:
            fcntl.flock(self.lock, (fcntl.LOCK_EX if mode == "record" else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            self.initialize(trials, start_seed)
        except BaseException:
            self.lock.close()
            raise

    def initialize(self, trials, start_seed):
        if self.mode == "record":
            if self.path.exists():
                raise ValueError("Recording refuses to overwrite an existing scene bank; use replay or a new directory.")
            self.rows = []
        else:
            payload = json.loads(self.path.read_text())
            if payload["identity"] != self.identity:
                raise ValueError("Expert scene bank identity differs.")
            self.rows = payload["scenes"]
            seed = start_seed
            if len(self.rows) != trials:
                raise ValueError("Expert scene bank is incomplete.")
            for index, row in enumerate(self.rows):
                attempts = row["attempts"]
                if (row["index"] != index or row["start_seed"] != seed or not 1 <= len(attempts) <= 5
                        or [a["seed"] for a in attempts] != list(range(seed, seed + len(attempts)))
                        or any(a["success"] for a in attempts[:-1])
                        or (len(attempts) < 5 and not attempts[-1]["success"])
                        or row["policy_seed"] != attempts[-1]["seed"] or not row.get("ready")):
                    raise ValueError("Invalid or incomplete expert seed chain.")
                seed = row["policy_seed"] + 1

    @classmethod
    def from_environment(cls, task, setting, trials, start_seed):
        from importlib.metadata import PackageNotFoundError, version
        versions = {}
        for package in ("numpy", "torch", "sapien", "mplib", "curobo"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        identity = {**json.loads(os.environ["LRVLA_EXPERT_CACHE_IDENTITY"]), "simulator_packages": versions}
        return cls(os.environ["LRVLA_EXPERT_CACHE"], identity,
                   os.environ["LRVLA_EXPERT_CACHE_MODE"], task=task, setting=setting,
                   trials=trials, start_seed=start_seed)

    def save(self):
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"identity": self.identity, "scenes": self.rows}) + "\n")
        temporary.replace(self.path)

    def lookup(self, index, start_seed):
        if self.mode == "record":
            return None
        row = deepcopy(self.rows[index])
        if row["start_seed"] != start_seed:
            raise ValueError("Expert replay seed chain changed.")
        print(f"Expert cache hit: scene={index}, policy_seed={row['policy_seed']}", flush=True)
        return row

    def expert_done(self, index, start_seed, policy_seed, episode_info, attempts):
        row = {"index": index, "start_seed": start_seed, "policy_seed": policy_seed,
               "episode_info": episode_info, "attempts": attempts, "python_before_setup": random.getstate()}
        if index != len(self.rows):
            raise ValueError("Expert cache must be recorded in episode order.")
        self.rows.append(row)
        self.save()
        return row

    def before_setup(self, row):
        if self.mode == "replay":
            random.setstate(nested_tuples(row["python_before_setup"]))

    def before_instruction(self, row):
        state = numpy_state()
        if self.mode == "record":
            row.update(python_before_instruction=random.getstate(), numpy_before_instruction=state)
        else:
            if row.get("policy_initialization_unstable"):
                raise ValueError("A previously unstable cached scene initialized successfully.")
            if state != row["numpy_before_instruction"]:
                raise ValueError("Cached scene NumPy state differs after policy setup.")
            random.setstate(nested_tuples(row["python_before_instruction"]))

    def instruction(self, row, instruction):
        if self.mode == "record":
            row["instruction"] = str(instruction)
            self.save()
        elif str(instruction) != row["instruction"]:
            raise ValueError("Cached instruction did not replay exactly.")

    def first_observation(self, row, observation):
        signature = observation_identity(observation)
        if self.mode == "record":
            row.update(initial_observation_sha256=signature, ready=True)
            self.save()
        elif signature != row.get("initial_observation_sha256"):
            raise ValueError("Cached initial RGB/joint state differs; rebuild and validate the scene bank.")

    def unstable(self, row):
        if self.mode == "record":
            row.update(policy_initialization_unstable=True, ready=True)
            self.save()
        elif not row.get("policy_initialization_unstable"):
            raise ValueError("A previously stable cached scene became unstable.")

    def close(self):
        self.lock.close()
