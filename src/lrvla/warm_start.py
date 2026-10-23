"""Isolated warm-start BC: whitelisted augmentations on top of the unchanged trainer.

The original entry point (scripts/train.py) is untouched, so earlier runs remain
reproducible. scripts/train_warm.py installs these hooks for its own process only:

- init_from: after the FP32 trainable masters are restored from the original base,
  replace them with an earlier run's FP32 export (frozen tensors stay untouched);
- image_crop_scale: small random crop/shift/zoom of each camera view (rule 2.1.1);
- state_noise_std: additive proprioception noise on the real joints (rule 2.1);
- color_temperature, image_blur_prob, image_noise_std, image_erase_prob and
  image_erase_max_area: colour-temperature jitter, Gaussian blur, Gaussian pixel
  noise and random erasing (all rule 2.1.1 pixel degradations).

Every option is off unless configured and draws from its own random stream, so
enabling one never changes the draws of another and earlier configurations
reproduce exactly.

The augmentations apply to training datasets only; validation and inference never
see them. Both enter the run's data_sampling identity, and warm_start.json in the
run directory records the warm-start source.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

from . import training, training_data, training_distributed
from .training_checkpoint import base_inventory

DEFAULT_CONFIG = "configs/train_expert_full_warm.yaml"


def sample_generator(seed, epoch, task, index, stream):
    """Replayable per-sample randomness, independent of worker count or prefetch."""
    digest = hashlib.sha256(f"{seed}:{epoch}:{task}:{index}:{stream}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


def random_area_crop(image, min_area, generator):
    """Keep a random [min_area, 1] fraction of the view and resize it back."""
    if min_area >= 1:
        return image
    height, width = image.shape[-2:]
    side = (min_area + (1 - min_area) * float(torch.rand((), generator=generator))) ** 0.5
    crop_h, crop_w = max(1, round(height * side)), max(1, round(width * side))
    top = int(torch.randint(0, height - crop_h + 1, (), generator=generator))
    left = int(torch.randint(0, width - crop_w + 1, (), generator=generator))
    crop = image[..., top:top + crop_h, left:left + crop_w]
    resized = torch.nn.functional.interpolate(crop.reshape(-1, *crop.shape[-3:]).float(), size=(height, width),
                                              mode="bilinear", align_corners=False, antialias=True)
    resized = resized.reshape(image.shape)
    return resized.to(image.dtype) if image.is_floating_point() else resized.round().clamp(0, 255).to(image.dtype)


PIXEL_LIMITS = {"color_temperature": (0, 0.3), "image_blur_prob": (0, 1), "image_noise_std": (0, 0.1),
                "image_erase_prob": (0, 1)}
ERASE_MIN_AREA = 0.02


def unit_image(image):
    """Float [0, 1] view of an image plus a function restoring its dtype and scale."""
    scale = 255.0 if not image.is_floating_point() or float(image.max()) > 2.0 else 1.0
    def restore(unit):
        value = unit.clamp(0, 1) * scale
        return value.to(image.dtype) if image.is_floating_point() else value.round().to(image.dtype)
    return image.float() / scale, restore


def gaussian_blur(image, sigma, size=5):
    axis = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-axis ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = image.shape[-3]
    flat = image.reshape(-1, channels, *image.shape[-2:])
    padded = torch.nn.functional.pad(flat, (size // 2,) * 4, mode="reflect")
    horizontal = torch.nn.functional.conv2d(padded, kernel.view(1, 1, 1, size).repeat(channels, 1, 1, 1), groups=channels)
    blurred = torch.nn.functional.conv2d(horizontal, kernel.view(1, 1, size, 1).repeat(channels, 1, 1, 1), groups=channels)
    return blurred.reshape(image.shape)


def random_erase(image, max_area, generator):
    """Cover one rectangle of [ERASE_MIN_AREA, max_area] of the view with uniform noise."""
    height, width = image.shape[-2:]
    area = (ERASE_MIN_AREA + (max_area - ERASE_MIN_AREA) * float(torch.rand((), generator=generator))) * height * width
    ratio = float(torch.exp(torch.empty(()).uniform_(-1.2, 1.2, generator=generator)))
    erase_h = max(1, min(height, round((area * ratio) ** 0.5)))
    erase_w = max(1, min(width, round((area / ratio) ** 0.5)))
    top = int(torch.randint(0, height - erase_h + 1, (), generator=generator))
    left = int(torch.randint(0, width - erase_w + 1, (), generator=generator))
    erased = image.clone()
    erased[..., top:top + erase_h, left:left + erase_w] = torch.rand(
        (*image.shape[:-2], erase_h, erase_w), generator=generator)
    return erased


def pixel_degradations(raw, views, pixels, owner, index):
    """Colour temperature (shared by all views), then per-view blur, noise and erasing."""
    units, restores = {}, {}
    for key in views:
        units[key], restores[key] = unit_image(raw[key])
    if pixels.get("color_temperature"):
        shift = (2 * float(torch.rand((), generator=sample_generator(owner.seed, owner.epoch, owner.task, index,
                                                                     "temperature"))) - 1) * pixels["color_temperature"]
        gains = torch.tensor([1 + shift, 1.0, 1 - shift]).view(3, 1, 1)
        for key in views:
            units[key] = units[key] * gains
    if pixels.get("image_blur_prob"):
        generator = sample_generator(owner.seed, owner.epoch, owner.task, index, "blur")
        for key in views:
            sigma = 0.1 + 0.9 * float(torch.rand((), generator=generator))
            if float(torch.rand((), generator=generator)) < pixels["image_blur_prob"]:
                units[key] = gaussian_blur(units[key], sigma)
    if pixels.get("image_noise_std"):
        generator = sample_generator(owner.seed, owner.epoch, owner.task, index, "pixel_noise")
        for key in views:
            std = pixels["image_noise_std"] * float(torch.rand((), generator=generator))
            units[key] = units[key] + std * torch.randn(units[key].shape, generator=generator)
    if pixels.get("image_erase_prob"):
        generator = sample_generator(owner.seed, owner.epoch, owner.task, index, "erase")
        for key in views:
            if float(torch.rand((), generator=generator)) < pixels["image_erase_prob"]:
                units[key] = random_erase(units[key].clamp(0, 1), pixels["image_erase_max_area"], generator)
    return {key: restores[key](units[key]) for key in views}


def augmentation_settings(config):
    """Validated augmentation values; absent or zero settings stay disabled."""
    values = {}
    for key, (low, high) in PIXEL_LIMITS.items():
        if float(config.get(key) or 0):
            value = float(config[key])
            if not low < value <= high:
                raise ValueError(f"{key} must be in ({low}, {high}].")
            values[key] = value
    if "image_erase_prob" in values:
        area = float(config.get("image_erase_max_area", 0.12))
        if not ERASE_MIN_AREA < area <= 0.25:
            raise ValueError(f"image_erase_max_area must be in ({ERASE_MIN_AREA}, 0.25].")
        values["image_erase_max_area"] = area
    if config.get("image_crop_scale") is not None:
        crop = float(config["image_crop_scale"])
        if not 0.5 <= crop <= 1:
            raise ValueError("image_crop_scale is the minimum kept view area and must be in [0.5, 1].")
        values["image_crop_scale"] = crop
    if float(config.get("state_noise_std") or 0):
        noise = float(config["state_noise_std"])
        if not 0 < noise <= 0.1:
            raise ValueError("state_noise_std is in normalized state units and must be in (0, 0.1].")
        values["state_noise_std"] = noise
    return values


def augmentation_identity(config):
    values = augmentation_settings(config)
    identity = {}
    if "image_crop_scale" in values:
        identity["image_crop"] = {"min_area": values["image_crop_scale"], "implementation": "per_view_area_crop_bilinear_v1"}
    if "state_noise_std" in values:
        identity["state_noise"] = {"std": values["state_noise_std"], "implementation": "normalized_valid_dims_gaussian_v1"}
    pixels = {key: values[key] for key in (*PIXEL_LIMITS, "image_erase_max_area") if key in values}
    if pixels:
        identity["pixel_degradation"] = {**pixels, "erase_min_area": ERASE_MIN_AREA if "image_erase_prob" in pixels else None,
                                         "implementation": "temperature_blur_noise_erase_v1"}
    return identity


class AugmentingTransform:
    """Wrap a training dataset's FeatureTransform; only apply() changes."""

    def __init__(self, inner, owner, crop_scale, state_noise, pixels=None):
        self.inner, self.owner, self.crop_scale, self.state_noise = inner, owner, crop_scale, state_noise
        self.pixels = dict(pixels or {})

    def __getattr__(self, name):
        if name.startswith("__") or name in {"inner", "owner", "crop_scale", "state_noise", "pixels"}:
            raise AttributeError(name)
        return getattr(self.inner, name)

    def apply(self, raw, policy_eval=False):
        if policy_eval:
            return self.inner.apply(raw, policy_eval=True)
        owner, index = self.owner, int(raw["index"])
        if self.crop_scale is not None and self.crop_scale < 1:
            raw = dict(raw)
            generator = sample_generator(owner.seed, owner.epoch, owner.task, index, "crop")
            for key in sorted(raw):
                value = raw[key]
                if key.startswith("observation.images.") and torch.is_tensor(value) and value.ndim >= 3:
                    raw[key] = random_area_crop(value, self.crop_scale, generator)
        if getattr(self, "pixels", None):
            views = [key for key in sorted(raw)
                     if key.startswith("observation.images.") and torch.is_tensor(raw[key]) and raw[key].ndim >= 3]
            raw = {**raw, **pixel_degradations(raw, views, self.pixels, owner, index)}
        item = self.inner.apply(raw)
        if self.state_noise:
            # Padded state dimensions stay exactly zero.
            generator = sample_generator(owner.seed, owner.epoch, owner.task, index, "state")
            noise = torch.randn(item["state"].shape, generator=generator) * self.state_noise
            item["state"] = item["state"] + (noise * item["state_joint_mask"].float()).to(item["state"].dtype)
        return item


class AugmentedCleanDataset(training_data.LocalCleanDataset):
    """LocalCleanDataset whose training instances use AugmentingTransform."""

    def __init__(self, entry, episodes, data_config, model_config, processor, robot_config, training=True, **kwargs):
        super().__init__(entry, episodes, data_config, model_config, processor, robot_config, training, **kwargs)
        crop, noise = data_config.get("image_crop_scale"), float(data_config.get("state_noise_std") or 0)
        pixels = {key: data_config[key] for key in (*PIXEL_LIMITS, "image_erase_max_area") if data_config.get(key)}
        if training and (crop is not None or noise or pixels):
            self.transform = AugmentingTransform(self.transform, self, crop, noise, pixels)


def load_trainable_from_export(model, export):
    """Replace FP32 trainable masters with an earlier run's FP32 export."""
    export = Path(export)
    record = json.loads((export / "export.json").read_text())
    if record.get("dtype") != "torch.float32" or record.get("trainable_source", "fp32_master") != "fp32_master":
        raise ValueError("Warm start requires an FP32 export of FP32 master weights.")
    if record.get("merged_lora"):
        raise ValueError("A merged-LoRA export cannot initialize separate LoRA matrices.")
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    inventory = base_inventory(export)
    by_file = {}
    for name, parameter in parameters.items():
        if name not in inventory or inventory[name][1] != tuple(parameter.shape) or inventory[name][2] != "F32":
            raise ValueError(f"Warm-start export lacks an FP32 tensor matching {name}.")
        by_file.setdefault(inventory[name][0], []).append(name)
    from safetensors import safe_open
    with torch.no_grad():
        for path, names in by_file.items():
            with safe_open(path, framework="pt", device="cpu") as reader:
                for name in names:
                    parameters[name].copy_(reader.get_tensor(name))
    fingerprint = hashlib.sha256()
    for name in ("export.json", "model.safetensors.index.json"):
        fingerprint.update((export / name).read_bytes())
    return {"init_from": str(export), "export_fingerprint_sha256": fingerprint.hexdigest(), "tensors": len(parameters)}


@contextmanager
def installed(init_from=None):
    """Hook the trainer modules for this process only; originals are always restored."""
    targets = [(training_data, "LocalCleanDataset"), (training, "prepare_runtime_files"),
               (training, "sampling_identity"), (training_distributed, "sampling_identity"),
               (training, "restore_trainable_base"), (training_distributed, "restore_trainable_base")]
    originals = {target: getattr(*target) for target in targets}
    prepare = originals[(training, "prepare_runtime_files")]
    identity = originals[(training, "sampling_identity")]
    restore = originals[(training, "restore_trainable_base")]
    state = {"output": None, "initialization": None}

    def prepare_runtime_files(config, architecture, manifest_path, norm_path, output):
        data, robot_config, digest = prepare(config, architecture, manifest_path, norm_path, output)
        data.update(augmentation_settings(config))
        state["output"] = Path(output)
        return data, robot_config, digest

    def sampling_identity(config, seen_hash=None):
        return {**identity(config, seen_hash), **augmentation_identity(config)}

    def restore_trainable_base(model, checkpoint):
        restore(model, checkpoint)
        if init_from:
            state["initialization"] = load_trainable_from_export(model, init_from)
            if state["output"] is not None:  # the single process, or distributed rank 0
                (state["output"] / "warm_start.json").write_text(json.dumps(state["initialization"], indent=2) + "\n")

    replacements = {(training_data, "LocalCleanDataset"): AugmentedCleanDataset,
                    (training, "prepare_runtime_files"): prepare_runtime_files,
                    (training, "sampling_identity"): sampling_identity,
                    (training_distributed, "sampling_identity"): sampling_identity,
                    (training, "restore_trainable_base"): restore_trainable_base,
                    (training_distributed, "restore_trainable_base"): restore_trainable_base}
    try:
        for (module, name), value in replacements.items():
            setattr(module, name, value)
        yield state
    finally:
        for (module, name), value in originals.items():
            setattr(module, name, value)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    known, _ = parser.parse_known_args(argv)
    if "--config" not in argv:
        argv = ["--config", known.config, *argv]
    config = yaml.safe_load(training.resolve_path(known.config).read_text())
    augmentation_settings(config)  # reject invalid values before loading any weights
    init_from = training.resolve_path(config["init_from"]) if config.get("init_from") else None
    with installed(init_from):
        return training.main(argv)
