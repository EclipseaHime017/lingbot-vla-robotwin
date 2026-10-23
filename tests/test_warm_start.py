"""Isolated warm-start entry: augmentations, FP32 warm start and untouched original trainer."""
import json
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch
import yaml

from lrvla import training, training_data, training_distributed, training_instructions, warm_start
from lrvla.warm_start import (
    AugmentedCleanDataset, AugmentingTransform, installed, load_trainable_from_export, random_area_crop,
    sample_generator,
)


class FakeTransform:
    org_features = {"actions": ["action"]}

    def __init__(self):
        self.seen = []

    def apply(self, raw, policy_eval=False):
        self.seen.append((raw, policy_eval))
        return {"state": torch.zeros(6), "state_joint_mask": torch.tensor([True, True, True, False, False, False])}


def tiny_export(directory, values, dtype="torch.float32"):
    export = directory / "hf_ckpt"
    export.mkdir(parents=True)
    save_file({name: value.contiguous() for name, value in values.items()}, str(export / "model-00001.safetensors"))
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001.safetensors" for name in values}}))
    (export / "export.json").write_text(json.dumps({"dtype": dtype, "trainable_source": "fp32_master",
                                                    "merged_lora": False}))
    return export


def test_random_area_crop_keeps_geometry_and_replays_per_sample():
    image = torch.rand(3, 48, 64)
    first = random_area_crop(image, 0.6, sample_generator(1, 0, "task", 5, "crop"))
    assert first.shape == image.shape and first.dtype == image.dtype
    assert torch.equal(first, random_area_crop(image, 0.6, sample_generator(1, 0, "task", 5, "crop")))
    assert not torch.equal(first, random_area_crop(image, 0.6, sample_generator(1, 1, "task", 5, "crop")))
    assert random_area_crop(image, 1.0, sample_generator(1, 0, "task", 5, "crop")) is image
    as_uint8 = (image * 255).to(torch.uint8)
    cropped = random_area_crop(as_uint8, 0.6, sample_generator(1, 0, "task", 5, "crop"))
    assert cropped.dtype == torch.uint8 and cropped.shape == as_uint8.shape


def test_transform_crops_views_and_perturbs_only_real_joints():
    owner = SimpleNamespace(seed=3, epoch=0, task="adjust_bottle")
    inner = FakeTransform()
    wrapped = AugmentingTransform(inner, owner, 0.7, 0.05)
    image = torch.rand(3, 32, 32)
    raw = {"index": torch.tensor(11), "observation.images.cam_high": image, "observation.state": torch.zeros(14)}
    item = wrapped.apply(raw)
    passed = inner.seen[-1][0]
    assert passed["observation.images.cam_high"].shape == image.shape
    assert not torch.equal(passed["observation.images.cam_high"], image)
    assert raw["observation.images.cam_high"] is image
    assert item["state"][:3].abs().sum() > 0 and torch.equal(item["state"][3:], torch.zeros(3))
    assert torch.equal(wrapped.apply(raw)["state"], item["state"])
    owner.epoch = 1
    assert not torch.equal(wrapped.apply(raw)["state"], item["state"])
    wrapped.apply(raw, policy_eval=True)
    assert inner.seen[-1] == (raw, True)
    assert wrapped.org_features == inner.org_features


def test_dataset_hook_augments_training_but_not_validation(monkeypatch):
    def light_init(self, entry, episodes, data_config, model_config, processor, robot_config, training=True, **kwargs):
        self.transform, self.task, self.seed, self.epoch = FakeTransform(), entry["task"], 1, 0

    monkeypatch.setattr(training_data.LocalCleanDataset, "__init__", light_init)
    config = {"image_crop_scale": 0.9, "state_noise_std": 0.01}
    assert isinstance(AugmentedCleanDataset({"task": "click_bell"}, [0], config, None, None, None).transform,
                      AugmentingTransform)
    assert isinstance(AugmentedCleanDataset({"task": "click_bell"}, [1], config, None, None, None, False).transform,
                      FakeTransform)
    assert isinstance(AugmentedCleanDataset({"task": "click_bell"}, [0], {}, None, None, None).transform, FakeTransform)


def test_hooks_are_process_local_and_original_trainer_is_restored(monkeypatch, tmp_path):
    untouched = (training_data.LocalCleanDataset, training.sampling_identity, training_distributed.sampling_identity,
                 training.restore_trainable_base, training_distributed.restore_trainable_base)
    monkeypatch.setattr(training, "prepare_runtime_files", lambda *args: ({"img_size": 256}, "robot.yaml", "digest"))
    with installed() as state:
        assert training_data.LocalCleanDataset is AugmentedCleanDataset
        data, _, _ = training.prepare_runtime_files({"image_crop_scale": 0.9, "state_noise_std": 0.02}, {}, None, None,
                                                    tmp_path)
        assert data == {"img_size": 256, "image_crop_scale": 0.9, "state_noise_std": 0.02}
        assert state["output"] == tmp_path
        identity = training_distributed.sampling_identity({"image_crop_scale": 0.9})
        assert identity["image_crop"]["min_area"] == 0.9 and "state_noise" not in identity
        assert training.sampling_identity({}) == training_instructions.sampling_identity({})
        with pytest.raises(ValueError, match="image_crop_scale"):
            training.prepare_runtime_files({"image_crop_scale": 0.3}, {}, None, None, tmp_path)
    assert (training_data.LocalCleanDataset, training.sampling_identity, training_distributed.sampling_identity,
            training.restore_trainable_base, training_distributed.restore_trainable_base) == untouched


def test_warm_start_replaces_only_trainable_masters_and_records_source(monkeypatch, tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model[0].requires_grad_(False)
    frozen = model[0].weight.detach().clone()
    export = tiny_export(tmp_path / "earlier", {"1.weight": torch.full((2, 2), 3.0), "1.bias": torch.full((2,), 4.0),
                                                "0.weight": torch.zeros(2, 2), "0.bias": torch.zeros(2)})
    monkeypatch.setattr(training, "restore_trainable_base", lambda model, base: None)
    monkeypatch.setattr(training, "prepare_runtime_files", lambda *args: ({}, "robot.yaml", "digest"))
    with installed(export):
        training.prepare_runtime_files({}, {}, None, None, tmp_path)
        training.restore_trainable_base(model, "base")
    assert torch.equal(model[1].weight, torch.full((2, 2), 3.0)) and torch.equal(model[1].bias, torch.full((2,), 4.0))
    assert torch.equal(model[0].weight, frozen)
    record = json.loads((tmp_path / "warm_start.json").read_text())
    assert record["init_from"] == str(export) and record["tensors"] == 2


def test_warm_start_rejects_rounded_or_mismatched_exports(tmp_path):
    model = torch.nn.Linear(2, 2)
    rounded = tiny_export(tmp_path / "bf16", {"weight": torch.ones(2, 2), "bias": torch.ones(2)}, dtype="torch.bfloat16")
    with pytest.raises(ValueError, match="FP32 export"):
        load_trainable_from_export(model, rounded)
    mismatched = tiny_export(tmp_path / "shape", {"weight": torch.ones(3, 2), "bias": torch.ones(2)})
    with pytest.raises(ValueError, match="matching weight"):
        load_trainable_from_export(model, mismatched)


def test_entry_validates_augmentations_before_any_training(monkeypatch, tmp_path):
    config = tmp_path / "warm.yaml"
    config.write_text(yaml.safe_dump({"state_noise_std": 0.5}))
    calls = []
    monkeypatch.setattr(training, "main", lambda argv: calls.append(argv))
    with pytest.raises(ValueError, match="state_noise_std"):
        warm_start.main(["--config", str(config)])
    assert not calls
    config.write_text(yaml.safe_dump({"state_noise_std": 0.01}))
    warm_start.main(["--config", str(config), "--max-steps", "5"])
    assert calls == [["--config", str(config), "--max-steps", "5"]]


ROUND1 = {"image_crop_scale": 0.88, "state_noise_std": 0.01}
PIXELS = {"color_temperature": 0.1, "image_blur_prob": 0.2, "image_noise_std": 0.02, "image_erase_prob": 0.5,
          "image_erase_max_area": 0.12}


def two_view_sample(index=11):
    return {"index": torch.tensor(index), "observation.images.cam_high": torch.rand(3, 32, 32),
            "observation.images.cam_left_wrist": torch.rand(3, 32, 32), "observation.state": torch.zeros(14)}


def test_round_one_configuration_reproduces_exactly_with_new_code():
    owner = SimpleNamespace(seed=3, epoch=2, task="stack_bowls_two")
    raw = two_view_sample()
    inner = FakeTransform()
    item = AugmentingTransform(inner, owner, 0.88, 0.01).apply(raw)
    crop = sample_generator(3, 2, "stack_bowls_two", 11, "crop")
    expected = {key: random_area_crop(raw[key], 0.88, crop)
                for key in sorted(raw) if key.startswith("observation.images.")}
    passed = inner.seen[-1][0]
    assert all(torch.equal(passed[key], value) for key, value in expected.items())
    noise = torch.randn(6, generator=sample_generator(3, 2, "stack_bowls_two", 11, "state")) * 0.01
    assert torch.equal(item["state"], noise * torch.tensor([1.0, 1, 1, 0, 0, 0]))
    with_pixels = AugmentingTransform(FakeTransform(), owner, 0.88, 0.01, PIXELS).apply(raw)
    assert torch.equal(with_pixels["state"], item["state"])


def test_pixel_degradations_share_temperature_and_bound_each_effect():
    owner = SimpleNamespace(seed=5, epoch=0, task="dump_bin_bigbin")
    raw = {"index": torch.tensor(3), "observation.images.cam_high": torch.full((3, 40, 40), 0.5),
           "observation.images.cam_left_wrist": torch.full((3, 40, 40), 0.5)}
    views = ["observation.images.cam_high", "observation.images.cam_left_wrist"]
    warm = warm_start.pixel_degradations(raw, views, {"color_temperature": 0.2}, owner, 3)
    assert torch.equal(warm[views[0]], warm[views[1]]) and torch.allclose(warm[views[0]][1], torch.full((40, 40), 0.5))
    assert not torch.allclose(warm[views[0]][0], warm[views[0]][2])
    erased = warm_start.pixel_degradations(raw, views, {"image_erase_prob": 1.0, "image_erase_max_area": 0.12}, owner, 3)
    for key in views:
        changed = float((erased[key] != 0.5).any(dim=0).float().mean())
        assert 0.015 <= changed <= 0.13 and erased[key].min() >= 0 and erased[key].max() <= 1
    textured = {**raw, views[0]: torch.rand(3, 40, 40)}
    blurred = warm_start.pixel_degradations(textured, views[:1], {"image_blur_prob": 1.0}, owner, 3)[views[0]]
    assert blurred.std() < textured[views[0]].std()
    noisy = warm_start.pixel_degradations(raw, views, {"image_noise_std": 0.05}, owner, 3)
    assert all(0 <= float(noisy[key].min()) and float(noisy[key].max()) <= 1 for key in views)
    as_uint8 = {**raw, views[0]: torch.full((3, 40, 40), 128, dtype=torch.uint8)}
    restored = warm_start.pixel_degradations(as_uint8, views[:1], PIXELS, owner, 3)[views[0]]
    assert restored.dtype == torch.uint8 and restored.shape == (3, 40, 40)
    again = warm_start.pixel_degradations(raw, views, PIXELS, owner, 3)
    assert all(torch.equal(again[key], warm_start.pixel_degradations(raw, views, PIXELS, owner, 3)[key]) for key in views)


def test_pixel_settings_validate_and_enter_identity_only_when_enabled(monkeypatch):
    assert "pixel_degradation" not in warm_start.augmentation_identity(ROUND1)
    identity = warm_start.augmentation_identity({**ROUND1, **PIXELS})["pixel_degradation"]
    assert identity["image_erase_prob"] == 0.5 and identity["image_erase_max_area"] == 0.12
    for bad in ({"image_erase_prob": 1.5}, {"image_noise_std": 0.5}, {"color_temperature": 0.5},
                {"image_erase_prob": 0.5, "image_erase_max_area": 0.4}):
        with pytest.raises(ValueError):
            warm_start.augmentation_settings(bad)

    def light_init(self, entry, episodes, data_config, model_config, processor, robot_config, training=True, **kwargs):
        self.transform, self.task, self.seed, self.epoch = FakeTransform(), entry["task"], 1, 0

    monkeypatch.setattr(training_data.LocalCleanDataset, "__init__", light_init)
    dataset = AugmentedCleanDataset({"task": "click_bell"}, [0], warm_start.augmentation_settings(PIXELS),
                                    None, None, None)
    assert isinstance(dataset.transform, AugmentingTransform) and dataset.transform.pixels["image_erase_prob"] == 0.5
