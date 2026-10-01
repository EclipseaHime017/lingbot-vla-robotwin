"""Audited official clean-only RoboTwin joint14 data preparation.

Frame alignment follows pinned RoboTwin pi0 preprocessing: image/state[t]
predict the next recorded joint command q[t+1]. No simulation data is generated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = "TianxingChen/RoboTwin2.0"
SOURCE_REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
SETTING = "aloha-agilex_clean_50"
CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
TASKS = tuple(sorted((
    "lift_pot", "hanging_mug", "stack_bowls_three", "scan_object", "handover_block", "click_bell",
    "put_object_cabinet", "open_microwave", "stack_blocks_three", "place_shoe", "adjust_bottle",
    "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size", "click_alarmclock", "dump_bin_bigbin",
    "grab_roller", "handover_mic", "move_can_pot", "move_pillbottle_pad", "move_playingcard_away",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup", "place_fan",
    "place_mouse_pad", "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
    "move_stapler_pad", "open_laptop", "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket", "press_stapler",
    "rotate_qrcode", "shake_bottle_horizontally", "shake_bottle", "stack_blocks_two", "stack_bowls_two",
    "stamp_seal", "turn_switch", "put_bottles_dustbin",
)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_tasks(value: str | None) -> list[str]:
    tasks = list(TASKS) if not value or value == "all" else value.split(",")
    if len(set(tasks)) != len(tasks) or not set(tasks).issubset(TASKS):
        raise ValueError("Tasks must be unique members of the official 50-task list")
    return tasks


def extract_clean_archive(archive: Path, task: str, raw_root: Path, metadata_file: Path | None = None) -> Path:
    if archive.name != f"{SETTING}.zip" or task not in TASKS:
        raise ValueError("Only official aloha-agilex clean archives may be extracted")
    digest = file_sha256(archive)
    metadata = json.loads((metadata_file or ROOT / "configs/raw_clean_hf_metadata.json").read_text())
    relative = f"dataset/{task}/{SETTING}.zip"
    entry = next((entry for entry in metadata["files"] if entry["rfilename"] == relative), None)
    if not entry or metadata["revision"] != SOURCE_REVISION or digest != entry["lfs"]["sha256"]:
        raise ValueError(f"Archive provenance/hash mismatch: {archive}")
    task_root = (raw_root / task).resolve()
    task_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (task_root / member.filename).resolve()
            if not target.is_relative_to(task_root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive member")
        bundle.extractall(task_root)
    candidates = list(task_root.rglob("data/episode0.hdf5"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one episode0.hdf5 after extraction, found {len(candidates)}")
    root = candidates[0].parents[1]
    source = {"repo_id": SOURCE_REPO, "revision": SOURCE_REVISION, "archive": relative, "sha256": digest,
              "embodiment": "aloha-agilex", "setting": "clean", "episodes": 50}
    (root / "source_provenance.json").write_text(json.dumps(source, indent=2) + "\n")
    return root


def joint14_arrays(raw_file: Path):
    import h5py
    import numpy as np
    with h5py.File(raw_file, "r") as stream:
        left = np.asarray(stream["joint_action/left_arm"], dtype=np.float32)
        right = np.asarray(stream["joint_action/right_arm"], dtype=np.float32)
        lg = np.asarray(stream["joint_action/left_gripper"], dtype=np.float32).reshape(-1, 1)
        rg = np.asarray(stream["joint_action/right_gripper"], dtype=np.float32).reshape(-1, 1)
    if left.shape[1:] != (6,) or right.shape[1:] != (6,):
        raise ValueError("RoboTwin V2 requires two six-joint Aloha-AgileX arms")
    joints = np.concatenate([left, lg, right, rg], axis=1)
    if len(joints) < 2 or not np.isfinite(joints).all():
        raise ValueError("Raw episode must contain at least two finite joint14 frames")
    return joints[:-1], joints[1:]


def convert_raw_task(raw_path: Path, output: Path, task: str, seed: int = 42) -> Path:
    import cv2
    import h5py
    import numpy as np
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    source = json.loads((raw_path / "source_provenance.json").read_text())
    if source.get("repo_id") != SOURCE_REPO or source.get("revision") != SOURCE_REVISION or source.get("setting") != "clean" or source.get("embodiment") != "aloha-agilex":
        raise ValueError("Raw data lacks official Aloha-AgileX clean provenance")
    files = list((raw_path / "data").glob("episode*.hdf5"))
    ids = sorted(int(re.fullmatch(r"episode(\d+)\.hdf5", file.name).group(1)) for file in files)
    if ids != list(range(50)):
        raise ValueError(f"Expected exactly official episodes 0..49; found {ids}")
    if output.exists():
        audit_dataset(output, task)
        return output
    names = [f"{arm}_{joint}" for arm in ("left", "right") for joint in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]
    features = {key: {"dtype": "float32", "shape": (14,), "names": names} for key in ("observation.state", "action")}
    features.update({f"observation.images.{camera}": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]} for camera in CAMERAS})
    dataset = LeRobotDataset.create(repo_id=f"local/{output.name}", root=output, fps=50, robot_type="aloha", features=features,
                                   use_videos=True, image_writer_threads=4, image_writer_processes=0, video_backend="pyav")
    rng = random.Random(seed)
    for episode in range(50):
        raw_file = raw_path / "data" / f"episode{episode}.hdf5"
        states, actions = joint14_arrays(raw_file)
        instructions = json.loads((raw_path / "instructions" / f"episode{episode}.json").read_text())["seen"]
        if not instructions or not all(isinstance(text, str) and text for text in instructions):
            raise ValueError("Official seen instructions must be nonempty")
        instruction = rng.choice(instructions)
        with h5py.File(raw_file, "r") as stream:
            for frame in range(len(states)):
                item = {"observation.state": states[frame], "action": actions[frame], "task": instruction}
                for camera, original in CAMERAS.items():
                    blob = stream[f"observation/{original}/rgb"][frame]
                    encoded = np.frombuffer(blob, dtype=np.uint8)
                    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Invalid encoded image: task={task}, episode={episode}, frame={frame}, camera={camera}")
                    # Pinned RoboTwin pkl2hdf5 passes simulator RGB arrays directly
                    # to cv2.imencode without a color conversion. imdecode thus
                    # restores the original RGB channel ordering; do not swap it.
                    item[f"observation.images.{camera}"] = cv2.resize(image, (640, 480))
                dataset.add_frame(item)
        dataset.save_episode()
        print(f"{task}: converted clean episode {episode + 1}/50", flush=True)
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()
    (output / "source_provenance.json").write_text(json.dumps({**source, "conversion": "joint14 image/state[t] -> q[t+1]", "image_color": "RGB", "instructions": "seen", "seed": seed}, indent=2) + "\n")
    audit_dataset(output, task)
    return output


def audit_dataset(path: Path, task: str, episodes: int = 50) -> dict:
    path = Path(path).resolve()
    if task not in TASKS or "randomized" in str(path).lower() or "aug_" in str(path).lower():
        raise ValueError("Only the designated clean tasks are accepted")
    info = json.loads((path / "meta/info.json").read_text())
    source = json.loads((path / "source_provenance.json").read_text())
    if source.get("setting") != "clean" or source.get("embodiment") != "aloha-agilex" or source.get("repo_id") != SOURCE_REPO or source.get("revision") != SOURCE_REVISION or source.get("archive") != f"dataset/{task}/{SETTING}.zip":
        raise ValueError("Dataset provenance does not identify the official clean Aloha archive")
    official = json.loads((ROOT / "configs/raw_clean_hf_metadata.json").read_text())
    expected = next(item for item in official["files"] if item["rfilename"] == source["archive"])
    if source.get("sha256") != expected["lfs"]["sha256"]:
        raise ValueError("Dataset source archive hash differs from the official frozen release")
    if info.get("total_episodes") != episodes or episodes != 50:
        raise ValueError(f"Require exactly 50 official demonstrations per task; found {info.get('total_episodes')}")
    if info.get("codebase_version") not in ("v2.1", "v3.0"):
        raise ValueError("Expected LeRobot v2.1 or v3.0 metadata")
    for key in ("observation.state", "action"):
        if info["features"].get(key, {}).get("shape") != [14]:
            raise ValueError(f"{key} must be joint14, not the incompatible EEF16 representation")
    for camera in CAMERAS:
        if f"observation.images.{camera}" not in info["features"]:
            raise ValueError(f"Required camera missing: {camera}")
    parquet_files = sorted((path / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError("Dataset contains no local parquet frames")
    if info["codebase_version"].startswith("v2"):
        if len(parquet_files) != 50:
            raise ValueError("LeRobot v2 clean dataset must contain exactly 50 episode parquet files")
        for episode in range(50):
            chunk = episode // info.get("chunks_size", 1000)
            data_file = info["data_path"].format(episode_chunk=chunk, episode_index=episode)
            if not (path / data_file).is_file():
                raise ValueError(f"Missing clean episode data: {data_file}")
            for camera in CAMERAS:
                if info["features"][f"observation.images.{camera}"]["dtype"] == "video":
                    video = info["video_path"].format(episode_chunk=chunk, episode_index=episode, video_key=f"observation.images.{camera}")
                    if not (path / video).is_file():
                        raise ValueError(f"Missing clean episode video: {video}")
    else:
        import pyarrow.parquet as pq
        records = []
        for file in sorted((path / "meta/episodes").rglob("*.parquet")):
            records.extend(pq.read_table(file).to_pylist())
        if sorted(row["episode_index"] for row in records) != list(range(50)):
            raise ValueError("LeRobot v3 episode metadata must contain exactly clean IDs 0..49")
        if sum(row["length"] for row in records) != info["total_frames"]:
            raise ValueError("LeRobot v3 episode lengths differ from total_frames")
        for row in records:
            data_file = info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
            if not (path / data_file).is_file():
                raise ValueError(f"Missing clean episode data: {data_file}")
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                if info["features"][key]["dtype"] == "video":
                    video = info["video_path"].format(video_key=key, chunk_index=row[f"videos/{key}/chunk_index"], file_index=row[f"videos/{key}/file_index"])
                    if not (path / video).is_file():
                        raise ValueError(f"Missing clean episode video: {video}")
        if sum(pq.ParquetFile(file).metadata.num_rows for file in parquet_files) != info["total_frames"]:
            raise ValueError("Local parquet row counts differ from declared clean frames")
    metadata_digest = file_sha256(path / "meta/info.json")
    return {"task": task, "path": str(path), "episodes": 50, "episode_ids": list(range(50)), "frames": info["total_frames"],
            "format": info["codebase_version"], "source": source, "metadata_sha256": metadata_digest,
            "parquet_files": [str(file.relative_to(path)) for file in parquet_files]}


def prepare_manifest(data_root: Path, tasks: list[str], output: Path, allow_subset: bool = False) -> dict:
    if set(tasks) != set(TASKS) and not allow_subset:
        raise ValueError("Full training requires all 50 tasks; use --allow-subset only for pipeline smoke")
    entries = []
    for task in tasks:
        candidates = [data_root / f"{task}-{SETTING}-50", data_root / task]
        path = next((path for path in candidates if (path / "meta/info.json").exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing local clean LeRobot dataset for {task}")
        entries.append(audit_dataset(path, task))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    training_list = output.with_suffix(".txt")
    training_list.write_text("".join(f"robotwin {entry['path']}\n" for entry in entries))
    manifest = {"schema_version": 1, "clean_only": True, "complete_50_tasks": len(entries) == 50,
                "status": "full_clean_data" if len(entries) == 50 else "incomplete_smoke_subset",
                "training_list": str(training_list), "datasets": entries, "total_episodes": 50 * len(entries)}
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(path: Path | str) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("clean_only"):
        raise ValueError("Expected an audited clean-only manifest")
    tasks = [entry["task"] for entry in manifest["datasets"]]
    if len(set(tasks)) != len(tasks) or not set(tasks).issubset(TASKS):
        raise ValueError("Manifest tasks are invalid or duplicated")
    for entry in manifest["datasets"]:
        current = audit_dataset(Path(entry["path"]), entry["task"])
        if current["metadata_sha256"] != entry["metadata_sha256"] or current["source"] != entry["source"]:
            raise ValueError("Dataset changed after manifest creation")
    if manifest.get("complete_50_tasks") != (set(tasks) == set(TASKS)):
        raise ValueError("Manifest full-task completeness claim is incorrect")
    return manifest


def compute_clean_norm(manifest_path: Path, output: Path, val_episodes: int = 5) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from .training_identity import manifest_content_sha256
    if not 0 <= val_episodes < 50:
        raise ValueError("val_episodes must be between 0 and 49")
    manifest = load_manifest(manifest_path)
    state_batches, action_batches = [], []
    for entry in manifest["datasets"]:
        path = Path(entry["path"])
        for filename in entry["parquet_files"]:
            table = pq.read_table(path / filename, columns=["observation.state", "action", "episode_index"])
            ids = np.asarray(table["episode_index"].to_pylist())
            keep = ids < 50 - val_episodes
            if keep.any():
                state_batches.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[keep])
                action_batches.append(np.asarray(table["action"].to_pylist(), dtype=np.float32)[keep])
    states, actions = np.concatenate(state_batches), np.concatenate(action_batches)
    result = {"norm_stats": {}, "count": len(states)}
    for prefix, values in (("observation.state", states), ("action", actions)):
        if not np.isfinite(values).all() or values.shape[1] != 14:
            raise ValueError("Normalization input must contain finite joint14 vectors")
        for name, cols in (("arm.position", list(range(6)) + list(range(7, 13))), ("effector.position", [6, 13])):
            batch = values[:, cols].astype(np.float64)
            result["norm_stats"][f"{prefix}.{name}"] = {
                "mean": batch.mean(axis=0).tolist(), "std": batch.std(axis=0).tolist(),
                **{f"q{int(q * 100):02d}": np.quantile(batch, q, axis=0).tolist() for q in (.01, .02, .98, .99)},
                "min": batch.min(axis=0).tolist(), "max": batch.max(axis=0).tolist(),
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    provenance = {"manifest": str(manifest_path.resolve()), "manifest_sha256": file_sha256(manifest_path),
                  "manifest_content_sha256": manifest_content_sha256(manifest),
                  "clean_only": True, "train_episode_ids": list(range(50 - val_episodes)), "val_episode_ids": list(range(50 - val_episodes, 50)),
                  "quantiles": "exact numpy linear; single-frame absolute actions", "count": len(states),
                  "norm_stats_sha256": file_sha256(output)}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return result
