"""Statistics-only demonstration screening and the filtered manifest it produces."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lrvla import demo_filter
from lrvla.data import TASKS

ROOT = Path(__file__).resolve().parents[1]


def row(task, episode, *, length=100, max_step=0.1, left=(10.0, 1.0), right=(0.0, 0.0), q_min=0.0, q_max=1.0):
    return {"task": task, "episode": episode, "length": length, "max_step": max_step,
            "joint_path": [left[0], right[0]], "ee_path": [left[1], right[1]],
            "q_min": [q_min] * 12, "q_max": [q_max] * 12}


def test_episode_statistics_measure_jumps_paths_and_ranges():
    joints = np.zeros((5, 14))
    joints[3, 0] = 0.5                                  # single-frame left waist jump
    joints[:, 6] = [0, 1, 1, 0, 0]                      # gripper motion is not an arm step
    left = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.2, 0, 0], [0.2, 0, 0]])
    stats = demo_filter.episode_statistics(joints, left, np.zeros((5, 7)))
    assert stats["length"] == 5 and stats["max_step"] == pytest.approx(0.5)
    assert stats["joint_path"] == [pytest.approx(1.0), 0.0] and stats["ee_path"] == [pytest.approx(0.2), 0.0]
    assert stats["q_max"][0] == 0.5 and len(stats["q_min"]) == 12
    for bad in (np.zeros((1, 14)), np.zeros((5, 16)), np.full((5, 14), np.nan)):
        with pytest.raises(ValueError):
            demo_filter.episode_statistics(bad, left, left)


def test_rules_compare_episodes_only_with_the_same_task_and_arm():
    rows = [row("adjust_bottle", index, length=100 + index % 3) for index in range(20)]
    rows += [row("adjust_bottle", 20, length=200),                       # long detour
             row("adjust_bottle", 21, max_step=0.5),                     # abrupt jump
             row("adjust_bottle", 22, left=(40.0, 1.0)),                 # joints swing, gripper barely moves
             row("adjust_bottle", 23, q_max=3.0)]                        # far outside the usual joint range
    # Right-arm episodes are longer by design; they must not look like left-arm outliers.
    rows += [row("adjust_bottle", 30 + index, length=180 + index % 2, left=(0.0, 0.0), right=(10.0, 1.0)) for index in range(10)]
    # Too few episodes for robust statistics: only jumps are screened.
    rows += [row("click_bell", 0, length=500), row("click_bell", 1, max_step=0.4), row("click_bell", 2)]
    reasons = {(item["task"], item["episode"]): item["reasons"] for item in demo_filter.flag_episodes(rows)}
    flagged = {key: value for key, value in reasons.items() if value}
    assert flagged == {("adjust_bottle", 20): ["long"], ("adjust_bottle", 21): ["jump"],
                       ("adjust_bottle", 22): ["ik_ratio"], ("adjust_bottle", 23): ["excursion"],
                       ("click_bell", 1): ["jump"]}
    record = demo_filter.exclusion_record(rows, {"adjust_bottle": "a" * 64})
    assert record["excluded"] == {"adjust_bottle": [20, 21, 22, 23], "click_bell": [1]}
    assert record["episodes_excluded"] == 5 and record["thresholds"] == demo_filter.THRESHOLDS


def test_filtered_manifest_keeps_the_remaining_episode_ids():
    manifest = {"schema_version": 1, "clean_only": True, "total_episodes": 100,
                "datasets": [{"task": "adjust_bottle", "episodes": 50, "episode_ids": list(range(50))},
                             {"task": "click_bell", "episodes": 50, "episode_ids": list(range(50))}]}
    record = {"filter_version": demo_filter.FILTER_VERSION, "thresholds": demo_filter.THRESHOLDS,
              "excluded": {"click_bell": [3, 49]}}
    filtered = demo_filter.filtered_manifest(manifest, record)
    assert filtered["datasets"][0]["episode_ids"] == list(range(50))
    assert filtered["datasets"][1]["episode_ids"] == [i for i in range(50) if i not in (3, 49)]
    assert filtered["total_episodes"] == 98 and filtered["demo_filter"]["episodes_excluded"] == 2
    assert manifest["datasets"][1]["episode_ids"] == list(range(50))  # input untouched
    for excluded in ({"stamp_seal": [1]}, {"click_bell": [50]}, {"click_bell": list(range(50))}):
        with pytest.raises(ValueError):
            demo_filter.filtered_manifest(manifest, {**record, "excluded": excluded})


def test_normalization_covers_only_kept_episodes(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    data = tmp_path / "adjust_bottle"
    (data / "data").mkdir(parents=True)
    frames = [(episode, float(episode == 7) * 100.0) for episode in range(10) for _ in range(4)]
    table = pa.table({"observation.state": [[value] * 14 for _, value in frames],
                      "action": [[value + 1] * 14 for _, value in frames],
                      "episode_index": [episode for episode, _ in frames]})
    pq.write_table(table, data / "data/file.parquet")
    manifest = {"schema_version": 1, "clean_only": True,
                "datasets": [{"task": "adjust_bottle", "path": str(data), "parquet_files": ["data/file.parquet"],
                              "episodes": 50, "episode_ids": [i for i in range(10) if i != 7]}]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    import lrvla.data
    monkeypatch.setattr(lrvla.data, "load_manifest", lambda path: json.loads(Path(path).read_text()))
    output = tmp_path / "norm.json"
    result = demo_filter.compute_norm_for_manifest(manifest_path, output)
    assert result["count"] == 36
    arm = result["norm_stats"]["observation.state.arm.position"]
    assert arm["max"] == [0.0] * 12 and result["norm_stats"]["action.arm.position"]["min"] == [1.0] * 12
    provenance = json.loads((tmp_path / "norm.provenance.json").read_text())
    # The checks training.prepare_runtime_files applies with val_episodes: 0.
    assert provenance["clean_only"] is True and provenance["val_episode_ids"] == []
    assert provenance["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert provenance["norm_stats_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance["episode_ids"] == {"adjust_bottle": [0, 1, 2, 3, 4, 5, 6, 8, 9]}


def test_committed_exclusion_list_and_training_config_agree():
    record = json.loads((ROOT / "configs/demo_filter.json").read_text())
    assert record["filter_version"] == demo_filter.FILTER_VERSION and record["thresholds"] == demo_filter.THRESHOLDS
    assert record["episodes_scanned"] == 2500 and set(record["archives"]) == set(TASKS)
    details = {(item["task"], item["episode"]) for item in record["details"]}
    assert details == {(task, episode) for task, ids in record["excluded"].items() for episode in ids}
    assert record["episodes_excluded"] == len(details) and all(item["reasons"] for item in record["details"])
    assert max(map(len, record["excluded"].values())) < 10
    import yaml
    gentle = yaml.safe_load((ROOT / "configs/train_expert_full_filter.yaml").read_text())
    strong = yaml.safe_load((ROOT / "configs/train_expert_full_filter_lr3e5.yaml").read_text())
    round1 = yaml.safe_load((ROOT / "configs/train_expert_full_warm.yaml").read_text())
    for config in (gentle, strong):
        assert config["manifest"] == "data/clean_manifest_filtered.json" and config["val_episodes"] == 0
        assert config["image_augment"] is False and "image_crop_scale" not in config and "state_noise_std" not in config
        assert config["init_from"] == round1["init_from"] and config["resample_seen_instructions"] is True
    # The strong variant differs from the gentle one only in its update budget, which matches round 1.
    changed = {key for key in gentle.keys() | strong.keys() if gentle.get(key) != strong.get(key)}
    assert changed == {"output_dir", "seed", "lr", "warmup_steps", "max_steps"}
    assert [strong[key] for key in ("lr", "warmup_steps", "max_steps")] == [round1[key] for key in ("lr", "warmup_steps", "max_steps")]
