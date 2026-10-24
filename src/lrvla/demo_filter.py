"""Statistics-only screening of clean demonstrations for implausible or abrupt arm motion.

A small share of RoboTwin 2.0 demonstrations comes from implausible inverse-kinematics
solutions: arms that swing through flipped configurations, take long detours or jump
between frames. Each episode is described by its length, largest single-frame arm-joint
step, joint-to-gripper path ratio per moving arm and joint-range excursion, and compared
only with episodes of the same task that move the same arm(s). Evaluation results are
never read; the rules and thresholds are the same for every task.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

FILTER_VERSION = "same_arm_robust_v1"
THRESHOLDS = {"length_z": 4.0, "ik_ratio_z": 5.0, "excursion": 1.0, "jump_rad": 0.3}
ARMS = list(range(0, 6)) + list(range(7, 13))
MOVING_PATH_M = 0.05
MIN_GROUP = 8


def episode_statistics(joints, left_pose, right_pose):
    """joints: (T, 14) joint14 record; *_pose: (T, >=3) end-effector positions."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 14 or len(joints) < 2 or not np.isfinite(joints).all():
        raise ValueError("Expected a finite (T >= 2, 14) joint record.")
    steps = np.abs(np.diff(joints[:, ARMS], axis=0))
    joint_path = [float(steps[:, arm * 6:(arm + 1) * 6].sum()) for arm in range(2)]
    ee_path = [float(np.linalg.norm(np.diff(np.asarray(pose, dtype=np.float64)[:, :3], axis=0), axis=1).sum())
               for pose in (left_pose, right_pose)]
    return {"length": len(joints), "max_step": float(steps.max()), "joint_path": joint_path, "ee_path": ee_path,
            "q_min": joints[:, ARMS].min(axis=0).tolist(), "q_max": joints[:, ARMS].max(axis=0).tolist()}


def scan_archive(archive, task, episodes=50):
    """Statistics of every episode in one official clean archive."""
    import h5py
    rows = []
    with zipfile.ZipFile(archive) as bundle:
        prefix = next(name.split("/")[0] for name in bundle.namelist() if "/data/episode" in name)
        for episode in range(episodes):
            with bundle.open(f"{prefix}/data/episode{episode}.hdf5") as stream:
                with h5py.File(io.BytesIO(stream.read()), "r") as record:
                    rows.append({"task": task, "episode": episode, **episode_statistics(
                        record["joint_action/vector"], record["endpose/left_endpose"], record["endpose/right_endpose"])})
    return rows


def robust_z(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    scale = max(np.median(np.abs(values - median)) * 1.4826, 0.02 * abs(median), 1e-9)
    return (values - median) / scale


def flag_episodes(rows, thresholds=THRESHOLDS):
    """Add `reasons` to each row: long, ik_ratio, excursion or jump."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], tuple(path > MOVING_PATH_M for path in row["ee_path"]))].append(row)
    flagged = []
    for (task, moving), members in groups.items():
        count = len(members)
        length_z, ratio_z, excursion = np.zeros(count), np.zeros(count), np.zeros(count)
        # Groups too small for robust statistics are only screened for jumps.
        if count >= MIN_GROUP:
            length_z = robust_z([row["length"] for row in members])
            for arm in range(2):
                if moving[arm]:
                    ratio_z = np.maximum(ratio_z, robust_z([row["joint_path"][arm] / row["ee_path"][arm] for row in members]))
            low = np.quantile([row["q_min"] for row in members], 0.1, axis=0)
            high = np.quantile([row["q_max"] for row in members], 0.9, axis=0)
            span = np.maximum(high - low, 0.3)
            excursion = np.array([max(((low - np.array(row["q_min"])) / span).max(),
                                      ((np.array(row["q_max"]) - high) / span).max()) for row in members])
        for index, row in enumerate(members):
            reasons = [name for name, hit in (("long", length_z[index] > thresholds["length_z"]),
                                               ("ik_ratio", ratio_z[index] > thresholds["ik_ratio_z"]),
                                               ("excursion", excursion[index] > thresholds["excursion"]),
                                               ("jump", row["max_step"] > thresholds["jump_rad"])) if hit]
            flagged.append({**row, "group_size": count, "length_z": float(length_z[index]),
                            "ratio_z": float(ratio_z[index]), "excursion": float(excursion[index]), "reasons": reasons})
    return sorted(flagged, key=lambda row: (row["task"], row["episode"]))


def exclusion_record(rows, archives, thresholds=THRESHOLDS):
    """Reviewable, versioned list of excluded episodes."""
    flagged = [row for row in flag_episodes(rows, thresholds) if row["reasons"]]
    excluded = defaultdict(list)
    for row in flagged:
        excluded[row["task"]].append(row["episode"])
    return {"filter_version": FILTER_VERSION, "thresholds": dict(thresholds), "episodes_scanned": len(rows),
            "episodes_excluded": len(flagged), "archives": archives, "excluded": dict(sorted(excluded.items())),
            "details": [{key: row[key] for key in ("task", "episode", "reasons", "length", "length_z", "ratio_z",
                                                    "excursion", "max_step")} for row in flagged]}


def filtered_manifest(manifest, record):
    """Copy a clean manifest, keeping only episodes that passed the screening."""
    excluded = {task: set(ids) for task, ids in record["excluded"].items()}
    tasks = {entry["task"] for entry in manifest["datasets"]}
    if set(excluded) - tasks:
        raise ValueError("Exclusion list names tasks outside the manifest.")
    result = json.loads(json.dumps(manifest))
    for entry in result["datasets"]:
        ids = entry.get("episode_ids", list(range(int(entry["episodes"]))))
        removed = excluded.get(entry["task"], set())
        if not removed <= set(ids) or len(removed) >= len(ids):
            raise ValueError(f"Invalid exclusions for {entry['task']}.")
        entry["episode_ids"] = [episode for episode in ids if episode not in removed]
    result["total_episodes"] = sum(len(entry["episode_ids"]) for entry in result["datasets"])
    result["demo_filter"] = {"filter_version": record["filter_version"], "thresholds": record["thresholds"],
                             "episodes_excluded": sum(map(len, excluded.values()))}
    return result


def compute_norm_for_manifest(manifest_path, output):
    """bounds/mean-std statistics over exactly the manifest's episode_ids, no holdout."""
    import pyarrow.parquet as pq
    from .data import file_sha256, load_manifest
    from .training_identity import manifest_content_sha256
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    states, actions = [], []
    for entry in manifest["datasets"]:
        keep_ids = set(entry["episode_ids"])
        for filename in entry["parquet_files"]:
            table = pq.read_table(Path(entry["path"]) / filename, columns=["observation.state", "action", "episode_index"])
            keep = np.isin(np.asarray(table["episode_index"].to_pylist()), list(keep_ids))
            if keep.any():
                states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[keep])
                actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32)[keep])
    states, actions = np.concatenate(states), np.concatenate(actions)
    result = {"norm_stats": {}, "count": len(states)}
    for prefix, values in (("observation.state", states), ("action", actions)):
        if not np.isfinite(values).all() or values.shape[1] != 14:
            raise ValueError("Normalization input must contain finite joint14 vectors")
        for name, cols in (("arm.position", ARMS), ("effector.position", [6, 13])):
            batch = values[:, cols].astype(np.float64)
            result["norm_stats"][f"{prefix}.{name}"] = {
                "mean": batch.mean(axis=0).tolist(), "std": batch.std(axis=0).tolist(),
                **{f"q{int(q * 100):02d}": np.quantile(batch, q, axis=0).tolist() for q in (.01, .02, .98, .99)},
                "min": batch.min(axis=0).tolist(), "max": batch.max(axis=0).tolist(),
            }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    provenance = {"manifest": str(manifest_path.resolve()), "manifest_sha256": file_sha256(manifest_path),
                  "manifest_content_sha256": manifest_content_sha256(manifest), "clean_only": True,
                  "episode_ids": {entry["task"]: entry["episode_ids"] for entry in manifest["datasets"]},
                  "val_episode_ids": [], "demo_filter": manifest.get("demo_filter"),
                  "quantiles": "exact numpy linear; single-frame absolute actions", "count": len(states),
                  "norm_stats_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return result
