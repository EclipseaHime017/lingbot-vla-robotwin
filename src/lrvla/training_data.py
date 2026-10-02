"""Local clean-only LeRobot inputs, episode holdout, and resumable shuffling."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

MODEL_KEYS = {"images", "img_masks", "lang_tokens", "lang_masks", "state", "actions", "joint_mask", "image_grid_thw"}


def validate_action_chunk_tail(value):
    if value not in {"mask", "drop"}:
        raise ValueError("action_chunk_tail must be 'mask' or 'drop'.")
    return value


class _AbsoluteEpisodeQueryMixin:
    """Translate local rows for the audited LeRobot v3 absolute episode bounds."""

    def _get_query_indices(self, idx, ep_idx):
        if not getattr(self, "_absolute_episode_bounds", True):
            return super()._get_query_indices(idx, ep_idx)
        absolute_idx = int(self.hf_dataset[idx]["index"])
        return super()._get_query_indices(absolute_idx, ep_idx)


def _episode_aware_backend_type():
    from lingbotvla.data.vla_data.base_dataset import LEROBOT_DATASET_API, LeRobotDataset
    backend = globals().get("EpisodeAwareLeRobotDataset")
    if backend is None or backend.__bases__[1] is not LeRobotDataset:
        backend = type("EpisodeAwareLeRobotDataset", (_AbsoluteEpisodeQueryMixin, LeRobotDataset),
                       {"__module__": __name__, "_absolute_episode_bounds": LEROBOT_DATASET_API == "v3"})
        globals()["EpisodeAwareLeRobotDataset"] = backend
    return backend


def __getattr__(name):
    # Spawned DataLoader workers must resolve the same module-level class while
    # importing the heavy vendor dependencies only when an actual dataset is used.
    if name == "EpisodeAwareLeRobotDataset":
        return _episode_aware_backend_type()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def full_action_chunk_indices(dataset, episodes, chunk_size):
    """Select local row indices from metadata, without loading images or actions."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    selected_episodes = set(map(int, episodes))
    # Episode metadata can itself be an HF Dataset with a wide statistics row.
    # Read each row once rather than once per training frame.
    boundaries = {}
    for episode_id in selected_episodes:
        episode = dataset.meta.episodes[episode_id]
        boundaries[episode_id] = (int(episode["dataset_to_index"]) if "dataset_to_index" in episode else None,
                                  int(episode["length"]))
    indices = []
    episode_ids = dataset.hf_dataset["episode_index"]
    absolute_indices = dataset.hf_dataset["index"]
    frame_indices = dataset.hf_dataset["frame_index"]
    for relative_idx, (episode_id, absolute_idx, frame_idx) in enumerate(
            zip(episode_ids, absolute_indices, frame_indices)):
        episode_id, absolute_idx, frame_idx = int(episode_id), int(absolute_idx), int(frame_idx)
        if episode_id not in selected_episodes:
            raise ValueError("Loaded dataset contains an episode outside the requested split.")
        absolute_end, length = boundaries[episode_id]
        # v3 stores absolute bounds; v2 metadata provides the episode length.
        if absolute_end is not None:
            complete = absolute_idx + chunk_size <= absolute_end
        else:
            complete = frame_idx + chunk_size <= length
        if complete:
            indices.append(relative_idx)
    return indices


def model_collate(items):
    if not items:
        raise ValueError("Empty batch")
    result = {}
    for key in MODEL_KEYS:
        if any(key not in item for item in items):
            raise ValueError(f"Model input key missing: {key}")
        values = [item[key] for item in items]
        if key in {"lang_tokens", "lang_masks"}:
            result[key] = pad_sequence(values, batch_first=True, padding_value=0)
        else:
            result[key] = torch.stack(values)
    return result


def episode_split(ids, val_episodes):
    ids = sorted(set(map(int, ids)))
    if val_episodes < 0 or val_episodes >= len(ids):
        raise ValueError("val_episodes must leave at least one training episode.")
    return (ids[:-val_episodes], ids[-val_episodes:]) if val_episodes else (ids, [])


class LocalCleanDataset(Dataset):
    def __init__(self, entry, episodes, data_config, model_config, processor, robot_config,
                 training=True, action_chunk_tail="mask"):
        action_chunk_tail = validate_action_chunk_tail(action_chunk_tail)
        from torchvision.transforms.v2 import Resize
        from lingbotvla.data.vla_data.base_dataset import LeRobotDatasetMetadata
        from lingbotvla.data.vla_data.utils import FeatureTransform
        backend_type = _episode_aware_backend_type()
        root = Path(entry["path"]).resolve()
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Local LeRobot metadata missing: {root}")
        meta = LeRobotDatasetMetadata(repo_id=root.name, root=root)
        cfg = SimpleNamespace(**data_config)
        self.transform = FeatureTransform(str(robot_config), cfg, model_config, processor,
                                          chunk_size=model_config.chunk_size,
                                          norm_stats_path=data_config["norm_stats_file"],
                                          image_augment=training and bool(data_config.get("image_augment", False)),
                                          use_depth_align=False, use_future_image=False)
        delta = {key: [t / meta.fps for t in range(model_config.chunk_size)]
                 for key in self.transform.org_features["actions"]}
        self.dataset = backend_type(
            repo_id=root.name, root=root, episodes=episodes, delta_timestamps=delta,
            image_transforms=Resize((cfg.img_size, cfg.img_size)),
            video_backend=data_config.get("video_backend", "pyav"), load_image=True)
        # Validation always retains all frames so changing training tail sampling
        # does not change the holdout loss population.
        self.action_chunk_tail = action_chunk_tail if training else "mask"
        self.sample_indices = None
        if self.action_chunk_tail == "drop":
            self.sample_indices = full_action_chunk_indices(self.dataset, episodes, model_config.chunk_size)
            if not self.sample_indices:
                raise ValueError(f"No complete action chunks remain for {entry['task']} with "
                                 f"chunk_size={model_config.chunk_size}; use action_chunk_tail='mask'.")

    def __len__(self):
        indices = getattr(self, "sample_indices", None)
        return len(indices) if indices is not None else len(self.dataset)

    def __getitem__(self, idx):
        indices = getattr(self, "sample_indices", None)
        if indices is not None:
            idx = indices[idx]
        raw = self.dataset[idx]
        # Joint-space data has [14] states and [T,14] absolute action chunks.
        if raw["observation.state"].shape[-1] != 14 or raw["action"].shape[-1] != 14:
            raise ValueError("Competition baseline requires joint-space 14D state/actions.")
        item = self.transform.apply(raw)
        # Upstream joint_mask only covers padded dimensions; temporal padding
        # must be removed explicitly so repeated episode-end targets add no loss.
        temporal_pad = item["action_is_pad"].bool().reshape(-1)
        if temporal_pad.shape[0] != item["joint_mask"].shape[0]:
            raise ValueError("action_is_pad length does not match the action chunk horizon.")
        item["joint_mask"] = item["joint_mask"].bool() & ~temporal_pad[:, None]
        return {key: item[key] for key in MODEL_KEYS}


def build_clean_datasets(manifest, data_config, model_config, processor, robot_config,
                         val_episodes=5, action_chunk_tail="mask"):
    action_chunk_tail = validate_action_chunk_tail(action_chunk_tail)
    if manifest.get("clean_only") is not True:
        raise ValueError("A verified clean_only manifest is required.")
    train, validation, splits = [], [], {}
    for entry in manifest["datasets"]:
        ids = entry.get("episode_ids", list(range(int(entry["episodes"]))))
        train_ids, val_ids = episode_split(ids, val_episodes)
        splits[entry["task"]] = {"train": train_ids, "validation": val_ids}
        train.append(LocalCleanDataset(entry, train_ids, data_config, model_config, processor, robot_config,
                                       action_chunk_tail=action_chunk_tail))
        if val_ids:
            validation.append(LocalCleanDataset(entry, val_ids, data_config, model_config, processor, robot_config, False))
    if not train:
        raise ValueError("No local clean datasets in the manifest.")
    return ConcatDataset(train), ConcatDataset(validation) if validation else None, splits


class EpochBatchSampler(Sampler):
    """Restart an epoch at its consumed-batch offset, independent of prefetch."""
    def __init__(self, length, batch_size, seed, epoch=0, offset=0):
        if length <= 0 or batch_size <= 0:
            raise ValueError("Dataset length and batch size must be positive.")
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.epoch, self.offset = epoch, offset

    def __iter__(self):
        order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        batches = [order[i:i + self.batch_size] for i in range(0, self.length, self.batch_size)]
        yield from batches[self.offset:]

    def __len__(self):
        return max(0, (self.length + self.batch_size - 1) // self.batch_size - self.offset)


def epoch_dataloader(dataset, batch_size, seed, epoch=0, offset=0, num_workers=0, collate_fn=model_collate):
    sampler = EpochBatchSampler(len(dataset), batch_size, seed, epoch, offset)
    # DataLoader draws its worker/base seed even with num_workers=0. Isolate that
    # draw so constructing a resumed iterator does not change saved FM noise RNG.
    worker_rng = torch.Generator().manual_seed(seed + 100000 + epoch)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=worker_rng)
