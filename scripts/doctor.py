#!/usr/bin/env python3
"""Probe actual CUDA kernels, native physics and Vulkan rendering in isolation.

Runs with the standard library so a broken environment can still be diagnosed.
Native crashes and timeouts are captured instead of hanging the parent process.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lrvla.vendor_sources import upstream_revision
from lrvla.wsl_tools import linux_executable


PREFIX = 'DOCTOR_JSON='
EXPECTED_ROBOTWIN_REVISION = '13c3c47ff4312dd62484bcd51be034af55c062d1'
EXPECTED_LINGBOT_REVISION = 'be969b8fd117fb70550c5d4bf4bc328211b5b1b6'


def revision_status(repo: Path, expected: str) -> dict:
    try:
        revision = upstream_revision(repo)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return {'ok': False, 'status': 'unverified_source', 'expected_revision': expected,
                'error': f'{type(error).__name__}: {error}'}
    return {'ok': revision == expected,
            'status': 'passed' if revision == expected else 'revision_mismatch',
            'revision': revision, 'expected_revision': expected}


def isolated_probe(python: Path, code: str, *, timeout: float = 30,
                   cwd: Path | None = None, env: dict | None = None) -> dict:
    started = time.monotonic()
    if not python.is_file():
        return {'ok': False, 'status': 'missing_environment', 'python': str(python)}
    try:
        python = linux_executable('python', preferred=python)
    except (OSError, ValueError, RuntimeError) as error:
        return {'ok': False, 'status': 'invalid_executable', 'python': str(python),
                'error': f'{type(error).__name__}: {error}'}
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env.update({'PYTHONNOUSERSITE': '1', 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '2'})
    # A failed native driver should not leave a large core dump in the project.
    code = 'import resource; resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n' + code
    proc = subprocess.Popen([str(python), '-c', code], cwd=cwd, env=child_env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
    result = {'ok': proc.returncode == 0 and not timed_out,
              'status': 'timeout' if timed_out else ('passed' if proc.returncode == 0 else 'failed'),
              'returncode': proc.returncode, 'seconds': round(time.monotonic() - started, 3),
              'stdout': stdout[-8000:], 'stderr': stderr[-8000:]}
    for line in stdout.splitlines():
        if line.startswith(PREFIX):
            try:
                result['details'] = json.loads(line[len(PREFIX):])
            except json.JSONDecodeError:
                result['ok'] = False
                result['status'] = 'invalid_probe_output'
    if result['ok'] and 'details' not in result:
        result['ok'] = False
        result['status'] = 'missing_probe_output'
    return result


def imports_code(modules: dict[str, str]) -> str:
    return f'''
import importlib, importlib.metadata, json
result = {{}}
for name, distribution in {modules!r}.items():
    try:
        importlib.import_module(name)
        try: version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError: version = 'source checkout'
        result[name] = {{'ok': True, 'version': version}}
    except Exception as exc:
        result[name] = {{'ok': False, 'error': f'{{type(exc).__name__}}: {{exc}}'}}
print({PREFIX!r} + json.dumps(result))
raise SystemExit(0 if all(item['ok'] for item in result.values()) else 1)
'''


CUDA_CODE = f'''
import json, math, torch
result = {{'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
           'available': torch.cuda.is_available()}}
if result['available']:
    props = torch.cuda.get_device_properties(0)
    x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
    value = (x @ x).float().sum().item()
    assert math.isfinite(value), value
    torch.cuda.synchronize()
    result.update(name=props.name, vram_mib=props.total_memory // 1048576,
                  capability=list(torch.cuda.get_device_capability()), bf16_matmul_sum=value,
                  compiled_architectures=torch.cuda.get_arch_list())
print({PREFIX!r} + json.dumps(result))
raise SystemExit(0 if result['available'] else 1)
'''

PHYSICS_CODE = f'''
import json, numpy as np, sapien, sapien.physx
# Explicit systems avoid SAPIEN's compatibility Engine creating a renderer.
scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
scene.set_timestep(1/120)
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
actor = builder.build(name='doctor_physics_box')
actor.set_pose(sapien.Pose([0, 0, 1]))
for _ in range(10): scene.step()
position = actor.get_pose().p
assert np.isfinite(position).all() and position[2] < 1, position
print({PREFIX!r} + json.dumps({{'steps': 10, 'position': position.tolist(), 'backend': 'PhysX CPU'}}))
'''

RENDER_CODE = f'''
import json, numpy as np, sapien.core as sapien
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
scene.add_ground(0)
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[1, 0, 0])
actor = builder.build(name='doctor_render_box')
actor.set_pose(sapien.Pose([0, 0, 0.1]))
camera = scene.add_camera('doctor_camera', 64, 64, 1.0, 0.01, 10)
camera.set_pose(sapien.Pose([0.6, 0, 0.3], [0, 0, 0, 1]))
scene.step()
scene.update_render()
camera.take_picture()
rgba = camera.get_picture('Color')
assert rgba.shape == (64, 64, 4) and np.isfinite(rgba).all(), rgba.shape
assert np.ptp(rgba[:, :, :3]) > 0, 'Renderer returned a constant image'
print({PREFIX!r} + json.dumps({{'shape': list(rgba.shape), 'rgb_min': float(rgba[:,:,:3].min()),
                                      'rgb_max': float(rgba[:,:,:3].max())}}))
'''


CUROBO_CODE = f'''
import json, torch
from curobo.wrap.reacher.motion_gen import MotionGen
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
# Bundled Franka forward kinematics checks the compiled CUDA extension;
# this creates no competition task episode or training observation.
tensor_args = TensorDeviceType()
kinematics = load_yaml(join_path(get_robot_path(), 'franka.yml'))['robot_cfg']['kinematics']
config = RobotConfig.from_basic(kinematics['urdf_path'], kinematics['base_link'],
                               kinematics['ee_link'], tensor_args)
model = CudaRobotModel(config.kinematics)
q = torch.zeros((1, model.get_dof()), **tensor_args.as_torch_dict())
state = model.get_state(q)
torch.cuda.synchronize()
assert torch.isfinite(state.ee_position).all() and torch.isfinite(state.ee_quaternion).all()
assert state.ee_position.shape == (1, 3) and state.ee_quaternion.shape == (1, 4)
quaternion_norm = torch.linalg.vector_norm(state.ee_quaternion).item()
assert abs(quaternion_norm - 1) < 1e-3, quaternion_norm
print({PREFIX!r} + json.dumps({{'robot': 'bundled Franka diagnostic', 'dof': model.get_dof(),
                               'ee_position': state.ee_position.tolist(),
                               'quaternion_norm': quaternion_norm,
                               'device': str(state.ee_position.device)}}))
'''


def vulkan_icds(search_paths: list[Path] | None = None) -> list[dict]:
    if search_paths is None:
        search_paths = [Path('/usr/share/vulkan/icd.d'), Path('/etc/vulkan/icd.d')]
        # User-specific Vulkan manifests are files, never credential locations.
        search_paths.append(Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share'))) / 'vulkan/icd.d')
    manifests = []
    for directory in search_paths:
        for path in sorted(directory.glob('*.json')):
            try:
                data = json.loads(path.read_text())
                library = data.get('ICD', {}).get('library_path', '')
                manifests.append({'path': str(path), 'library': library,
                                  'nvidia': 'nvidia' in (path.name + library).lower()})
            except (OSError, json.JSONDecodeError):
                manifests.append({'path': str(path), 'error': 'Unreadable Vulkan manifest'})
    return manifests


def summarize(report: dict) -> dict:
    sim = report.get('sim', {})
    train = report.get('train', {})
    physics_ok = sim.get('physics', {}).get('ok', False)
    rendering_ok = sim.get('render', {}).get('ok', False)
    sim_cuda_ok = sim.get('cuda', {}).get('ok', False)
    train_cuda_ok = train.get('cuda', {}).get('ok', False)
    sim_ready = all(sim.get(name, {}).get('ok', False)
                    for name in ('imports', 'cuda', 'physics', 'render', 'curobo', 'robotwin_import'))
    sim_ready = sim_ready and all(report.get('assets', {}).get(name, False)
                                 for name in ('aloha_agilex', 'objects', 'textures'))
    sim_ready = sim_ready and report.get('upstream', {}).get('robotwin', {}).get('ok', False)
    messages = []
    if sim and report.get('host', {}).get('wsl') and not rendering_ok:
        messages.append('WSL CUDA compute and Vulkan rendering are separate. RoboTwin officially '
                        'does not support WSL rendering; use native Linux for closed-loop RGB evaluation.')
    if physics_ok and not rendering_ok:
        messages.append('CPU physics passed; camera rendering did not. This is not a completed RoboTwin evaluation.')
    if sim and not sim.get('curobo', {}).get('ok', False):
        messages.append('CuRobo native import is not ready; inspect the compiler/build log before task evaluation.')
    return {'training_cuda': train_cuda_ok, 'simulation_cuda': sim_cuda_ok,
            'cpu_physics': physics_ok, 'camera_rendering': rendering_ok,
            'closed_loop_ready': sim_ready, 'messages': messages}


def collect(root: Path, *, scope: str, timeout: float, render: bool) -> dict:
    conda_root = Path(os.environ.get('CONDA_ROOT') or Path.home() / 'miniconda3').expanduser()
    report = {'generated_at': dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
              'host': {'system': platform.system(), 'release': platform.release(),
                       'wsl': 'microsoft' in platform.release().lower()},
              'project_root': str(root), 'conda_root': str(conda_root), 'vulkan_icds': vulkan_icds(),
              'stack_note': 'Local sim uses torch 2.8.0/cu128 and torchvision 0.23.0 for Blackwell; '
                            'upstream pinned RoboTwin requirements use torch 2.4.1.',
              'upstream': {}}
    shared_env = {'XDG_CACHE_HOME': str(root / '.cache'),
                  'PYTHONPATH': str(root / 'vendor/lingbot-vla-v2')}
    tool_path = os.environ.get('PATH', '')
    if scope in ('all', 'train'):
        report['upstream']['lingbot_vla'] = revision_status(root / 'vendor/lingbot-vla-v2', EXPECTED_LINGBOT_REVISION)
        python = conda_root / 'envs/lingbot-vla/bin/python'
        train_env = {**shared_env, 'PATH': str(python.parent) + os.pathsep + tool_path}
        report['train'] = {
            'imports': isolated_probe(python, imports_code({'torch': 'torch', 'transformers': 'transformers',
                'lerobot': 'lerobot', 'lingbotvla': 'lingbotvla', 'draccus': 'draccus'}),
                timeout=timeout, cwd=root, env=train_env),
            'cuda': isolated_probe(python, CUDA_CODE, timeout=timeout, cwd=root, env=train_env)}
    if scope in ('all', 'sim'):
        report['upstream']['robotwin'] = revision_status(root / 'vendor/RoboTwin', EXPECTED_ROBOTWIN_REVISION)
        python = conda_root / 'envs/robotwin-sim/bin/python'
        sim_env = {**shared_env, 'PYTHONPATH': str(root / 'vendor/RoboTwin'),
                   'PATH': str(python.parent) + os.pathsep + tool_path,
                   'LD_LIBRARY_PATH': str(python.parent.parent / 'lib') + os.pathsep +
                       os.environ.get('LD_LIBRARY_PATH', '')}
        report['sim'] = {
            'imports': isolated_probe(python, imports_code({'numpy': 'numpy', 'scipy': 'scipy',
                'sapien': 'sapien', 'mplib': 'mplib', 'gymnasium': 'gymnasium', 'transforms3d': 'transforms3d',
                'open3d': 'open3d', 'toppra': 'toppra', 'cv2': 'opencv-python',
                'websockets.sync.client': 'websockets', 'h5py': 'h5py'}),
                timeout=timeout, cwd=root, env=sim_env),
            'cuda': isolated_probe(python, CUDA_CODE, timeout=timeout, cwd=root, env=sim_env),
            'physics': isolated_probe(python, PHYSICS_CODE, timeout=timeout, cwd=root, env=sim_env),
            'render': isolated_probe(python, RENDER_CODE, timeout=timeout, cwd=root, env=sim_env)
                if render else {'ok': False, 'status': 'skipped'},
            'curobo': isolated_probe(python, CUROBO_CODE,
                timeout=timeout, cwd=root / 'vendor/RoboTwin', env=sim_env),
            'robotwin_import': isolated_probe(python, imports_code({'envs._base_task': 'RoboTwin'}),
                timeout=timeout, cwd=root / 'vendor/RoboTwin', env=sim_env)}
    assets = root / 'vendor/RoboTwin/assets'
    report['assets'] = {'aloha_agilex': (assets / 'embodiments/aloha-agilex/config.yml').is_file(),
                        'objects': (assets / 'objects').is_dir(),
                        'textures': (assets / 'background_texture').is_dir()}
    report['summary'] = summarize(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scope', choices=('all', 'sim', 'train'), default='all')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-render', action='store_true')
    parser.add_argument('--strict', action='store_true', help='Exit nonzero unless the selected workflow is ready')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    report = collect(args.project_root.resolve(), scope=args.scope, timeout=args.timeout,
                     render=not args.skip_render)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    train_ok = (report.get('train', {}).get('imports', {}).get('ok', False)
                and report['summary']['training_cuda']
                and report['upstream'].get('lingbot_vla', {}).get('ok', False))
    sim_ok = report['summary']['closed_loop_ready']
    ready = train_ok if args.scope == 'train' else sim_ok if args.scope == 'sim' else train_ok and sim_ok
    return 1 if args.strict and not ready else 0


if __name__ == '__main__':
    raise SystemExit(main())
