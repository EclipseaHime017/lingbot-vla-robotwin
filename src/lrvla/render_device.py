"""Bind SAPIEN 3 compatibility scenes to an explicitly assigned physical GPU."""
from __future__ import annotations

import json
import os
import re


def normalized_pci(value):
    match = re.fullmatch(r"(?:pci:)?([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-7])", value.lower().strip())
    if not match:
        raise ValueError(f"Invalid PCI GPU address: {value}")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def bind_render_device(pci=None):
    """Install before task imports or any renderer construction in this process.

    The legacy SapienRenderer wrapper discards device kwargs, and CUDA visibility
    alone does not select a Vulkan adapter. Bind both the first renderer and
    subsequent default scenes; explicit physics-only systems remain unchanged.
    """
    expected = normalized_pci(pci or os.environ["LRVLA_RENDER_PCI"])
    import sapien
    from sapien.wrapper.renderer import SapienRenderer
    from sapien.wrapper.scene import Scene
    previous = getattr(Scene, "_lrvla_bound_pci", None)
    if previous is not None:
        if previous != expected:
            raise RuntimeError("A simulator process cannot switch rendering GPUs.")
        return
    bound = sapien.Device("pci:" + expected)
    if not bound.can_render() or normalized_pci(bound.pci_string) != expected:
        raise RuntimeError(f"Assigned GPU cannot render: {expected}")
    original_scene_init = Scene.__init__
    native_renderer = sapien.pysapien.render.SapienRenderer
    announced = False

    def scene_init(self, systems=None):
        nonlocal announced
        default_systems = systems is None
        if default_systems:
            systems = [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(bound)]
        original_scene_init(self, systems)
        if default_systems:
            actual = self.render_system.device
            if normalized_pci(actual.pci_string) != expected:
                raise RuntimeError("SAPIEN scene uses a GPU outside this worker's assignment.")
            if not announced:
                print("LRVLA_GPU_BINDING=" + json.dumps({
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "render_pci": normalized_pci(actual.pci_string), "render_cuda_id": actual.cuda_id,
                    "name": actual.name}), flush=True)
                announced = True

    def renderer_init(self, **kwargs):
        native_renderer.__init__(self, bound)

    SapienRenderer.__init__ = renderer_init
    Scene.__init__ = scene_init
    Scene._lrvla_bound_pci = expected
