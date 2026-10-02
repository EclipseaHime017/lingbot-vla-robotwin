from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrvla.training_data import (
    LocalCleanDataset,
    MODEL_KEYS,
    _AbsoluteEpisodeQueryMixin,
    build_clean_datasets,
    full_action_chunk_indices,
    validate_action_chunk_tail,
)


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return [row[key] for row in self.rows] if isinstance(key, str) else self.rows[key]

    def __len__(self):
        return len(self.rows)


def episode_rows():
    episodes, rows, start = {}, [], 0
    for ep, length in enumerate([6, 2, 5]):
        episodes[ep] = {"dataset_from_index": start, "dataset_to_index": start + length, "length": length}
        rows.extend({"index": start + frame, "episode_index": ep, "frame_index": frame}
                    for frame in range(length))
        start += length
    return episodes, rows


def test_subset_queries_use_absolute_rows_with_the_actual_lerobot_query_method():
    # The actual upstream method clamps against metadata's absolute bounds.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as UpstreamDataset

    class Backend(_AbsoluteEpisodeQueryMixin, UpstreamDataset):
        pass

    episodes, rows = episode_rows()
    dataset = Backend.__new__(Backend)
    dataset.meta = SimpleNamespace(episodes=episodes)
    dataset.hf_dataset = Rows([row for row in rows if row["episode_index"] == 2])
    dataset.delta_indices = {"action": list(range(4))}
    first, first_padding = dataset._get_query_indices(0, 2)
    last, last_padding = dataset._get_query_indices(4, 2)
    assert first["action"] == [8, 9, 10, 11]
    assert not first_padding["action_is_pad"].any()
    assert last["action"] == [12, 12, 12, 12]
    assert last_padding["action_is_pad"].tolist() == [False, True, True, True]


def test_v2_query_path_is_preserved():
    class V2Backend:
        def _get_query_indices(self, idx, ep_idx):
            return idx, ep_idx

    class Backend(_AbsoluteEpisodeQueryMixin, V2Backend):
        _absolute_episode_bounds = False

    # No hf_dataset lookup or absolute conversion on the unaudited v2 path.
    assert Backend()._get_query_indices(3, 9) == (3, 9)


def test_full_windows_include_the_last_complete_start_and_exclude_short_episodes():
    episodes, rows = episode_rows()
    dataset = SimpleNamespace(meta=SimpleNamespace(episodes=episodes), hf_dataset=Rows(rows[:8]))
    assert full_action_chunk_indices(dataset, [0, 1], 4) == [0, 1, 2]
    subset = SimpleNamespace(meta=dataset.meta, hf_dataset=Rows(rows[8:]))
    assert full_action_chunk_indices(subset, [2], 4) == [0, 1]
    # A requested split cannot accidentally admit another episode's rows.
    with pytest.raises(ValueError, match="outside the requested split"):
        full_action_chunk_indices(dataset, [0], 4)


def test_v2_length_metadata_and_invalid_tail_configuration():
    dataset = SimpleNamespace(meta=SimpleNamespace(episodes={3: {"length": 5}}),
                              hf_dataset=Rows([{"episode_index": 3, "index": 100 + i, "frame_index": i}
                                               for i in range(5)]))
    assert full_action_chunk_indices(dataset, [3], 4) == [0, 1]
    with pytest.raises(ValueError, match="chunk_size"):
        full_action_chunk_indices(dataset, [3], 0)
    with pytest.raises(ValueError, match="action_chunk_tail"):
        validate_action_chunk_tail("truncate")


@pytest.fixture
def local_backend(tmp_path, monkeypatch):
    from lrvla.training_runtime import add_vendor_path
    add_vendor_path(Path(__file__).resolve().parents[1])
    from lingbotvla.data.vla_data import base_dataset, utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as UpstreamDataset

    episodes, rows = episode_rows()
    class Backend(UpstreamDataset):
        def __init__(self, *, episodes, **kwargs):
            self.meta = SimpleNamespace(episodes=metadata)
            self.hf_dataset = Rows([row for row in rows if row["episode_index"] in episodes])
            self.delta_indices = {"action": list(range(4))}

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            row = self.hf_dataset[idx]
            query, padding = self._get_query_indices(idx, row["episode_index"])
            return {"observation.state": torch.zeros(14),
                    "action": torch.tensor(query["action"])[:, None].expand(-1, 14).float(),
                    **padding}

    class Transform:
        def __init__(self, *args, **kwargs):
            self.org_features = {"actions": ["action"]}

        def apply(self, raw):
            item = {key: torch.empty(0) for key in MODEL_KEYS}
            item.update(actions=raw["action"], joint_mask=torch.ones(4, 14, dtype=torch.bool),
                        action_is_pad=raw["action_is_pad"])
            return item

    metadata = episodes
    monkeypatch.setattr(base_dataset, "LeRobotDataset", Backend)
    monkeypatch.setattr(base_dataset, "LeRobotDatasetMetadata", lambda **kwargs: SimpleNamespace(fps=50))
    monkeypatch.setattr(utils, "FeatureTransform", Transform)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    return {"task": "test_task", "path": str(tmp_path), "episodes": 3}, {"img_size": 32,
            "norm_stats_file": str(tmp_path / "norm.json")}, SimpleNamespace(chunk_size=4)


def test_drop_filters_training_but_preserves_validation_and_its_real_actions(local_backend):
    entry, data, model = local_backend
    train, validation, splits = build_clean_datasets({"clean_only": True, "datasets": [entry]}, data,
                                                   model, None, "robot.yaml", val_episodes=1,
                                                   action_chunk_tail="drop")
    assert splits == {"test_task": {"train": [0, 1], "validation": [2]}}
    assert len(train) == 3 and len(validation) == 5
    assert train[2]["actions"][:, 0].tolist() == [2, 3, 4, 5]
    assert train[2]["joint_mask"].all()
    assert validation[0]["actions"][:, 0].tolist() == [8, 9, 10, 11]
    assert validation[0]["joint_mask"].all()
    assert validation[4]["actions"][:, 0].tolist() == [12, 12, 12, 12]
    assert validation[4]["joint_mask"][:, 0].tolist() == [True, False, False, False]
    # A non-prefix subset still uses local row indices exactly once after filtering.
    nonprefix = LocalCleanDataset(entry, [2], data, model, None, "robot.yaml", action_chunk_tail="drop")
    assert len(nonprefix) == 2
    assert nonprefix[1]["actions"][:, 0].tolist() == [9, 10, 11, 12]


def test_mask_retains_all_samples_and_empty_complete_training_split_fails(local_backend):
    entry, data, model = local_backend
    train, validation, _ = build_clean_datasets({"clean_only": True, "datasets": [entry]}, data,
                                              model, None, "robot.yaml", val_episodes=1)
    assert len(train) == 8 and len(validation) == 5
    assert train[5]["joint_mask"][:, 0].tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="No complete action chunks remain"):
        LocalCleanDataset(entry, [1], data, model, None, "robot.yaml", action_chunk_tail="drop")


def test_backend_class_can_be_resolved_after_pickle_module_import(local_backend, monkeypatch):
    import pickle
    import lrvla.training_data as training_data

    entry, data, model = local_backend
    dataset = LocalCleanDataset(entry, [2], data, model, None, "robot.yaml").dataset
    payload = pickle.dumps(dataset)
    # Emulate class lookup after importing a fresh module in a spawned worker.
    monkeypatch.delattr(training_data, "EpisodeAwareLeRobotDataset")
    restored = pickle.loads(payload)
    assert restored[0]["action_is_pad"].tolist() == [False] * 4
    assert restored[4]["action_is_pad"].tolist() == [False, True, True, True]
