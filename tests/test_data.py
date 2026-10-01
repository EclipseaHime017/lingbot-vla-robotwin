import importlib.util
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import TASKS, SOURCE_REPO, SOURCE_REVISION, SETTING, audit_dataset, joint14_arrays, parse_tasks, prepare_manifest


class CleanDataTests(unittest.TestCase):
    def test_resumable_download_preserves_offset_and_checks_digest(self):
        spec = importlib.util.spec_from_file_location("download_assets", ROOT / "scripts/download_assets.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = bytes(range(256)) * 100
        seen_ranges = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                header = self.headers.get("Range")
                seen_ranges.append(header)
                offset = int(header.split("=")[1].split("-")[0]) if header else 0
                self.send_response(206 if header else 200)
                if header:
                    self.send_header("Content-Range", f"bytes {offset}-{len(payload)-1}/{len(payload)}")
                self.end_headers()
                self.wfile.write(payload[offset:])
            def log_message(self, *args):
                pass
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("Sandbox denies localhost sockets; run outside the network sandbox")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / "model.safetensors"
                path.with_name(path.name + ".part").write_bytes(payload[:1234])
                expected = hashlib.sha256(payload).hexdigest()
                record = module.download_file(f"http://127.0.0.1:{server.server_port}/file", path, len(payload), expected)
                self.assertEqual(seen_ranges, ["bytes=1234-"])
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(record["sha256"], expected)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_task_guard(self):
        self.assertEqual(len(TASKS), 50)
        with self.assertRaises(ValueError):
            parse_tasks("adjust_bottle,adjust_bottle")
        with self.assertRaises(ValueError):
            parse_tasks("not_a_task")

    def fixture(self, root, episodes=50, dimension=14):
        path = Path(root) / f"adjust_bottle-{SETTING}-50"
        (path / "meta").mkdir(parents=True)
        info = {"codebase_version": "v2.1", "total_episodes": episodes, "total_frames": 500,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {"observation.state": {"shape": [dimension]}, "action": {"shape": [dimension]}}}
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            info["features"][f"observation.images.{camera}"] = {"dtype": "video"}
        (path / "meta/info.json").write_text(json.dumps(info))
        metadata = json.loads((ROOT / "configs/raw_clean_hf_metadata.json").read_text())
        expected = next(item for item in metadata["files"] if item["rfilename"] == f"dataset/adjust_bottle/{SETTING}.zip")
        source = {"repo_id": SOURCE_REPO, "revision": SOURCE_REVISION, "setting": "clean", "embodiment": "aloha-agilex",
                  "archive": expected["rfilename"], "sha256": expected["lfs"]["sha256"]}
        (path / "source_provenance.json").write_text(json.dumps(source))
        for episode in range(episodes):
            p = path / f"data/chunk-000/episode_{episode:06d}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fixture")
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                p = path / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fixture")
        return path

    def test_rejects_wrong_episode_count_and_eef(self):
        for episodes, dimension in ((49, 14), (50, 16)):
            with tempfile.TemporaryDirectory() as root:
                path = self.fixture(root, episodes, dimension)
                with self.assertRaises(ValueError):
                    audit_dataset(path, "adjust_bottle")

    def test_missing_video_and_unverified_source_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture(root)
            source = json.loads((path / "source_provenance.json").read_text())
            source["sha256"] = "0" * 64
            (path / "source_provenance.json").write_text(json.dumps(source))
            with self.assertRaises(ValueError):
                audit_dataset(path, "adjust_bottle")
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture(root)
            next((path / "videos").rglob("*.mp4")).unlink()
            with self.assertRaises(ValueError):
                audit_dataset(path, "adjust_bottle")

    def test_subset_must_be_labeled_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root)
            with self.assertRaises(ValueError):
                prepare_manifest(Path(root), ["adjust_bottle"], Path(root) / "manifest.json")
            manifest = prepare_manifest(Path(root), ["adjust_bottle"], Path(root) / "manifest.json", allow_subset=True)
            self.assertFalse(manifest["complete_50_tasks"])
            self.assertEqual(manifest["status"], "incomplete_smoke_subset")

    @unittest.skipUnless(importlib.util.find_spec("h5py") and importlib.util.find_spec("numpy"), "requires data environment")
    def test_alignment_is_next_joint_command(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode.hdf5"
            with h5py.File(path, "w") as file:
                file["joint_action/left_arm"] = np.arange(18).reshape(3, 6)
                file["joint_action/right_arm"] = np.arange(18, 36).reshape(3, 6)
                file["joint_action/left_gripper"] = [0.0, 0.5, 1.0]
                file["joint_action/right_gripper"] = [1.0, 0.5, 0.0]
            states, actions = joint14_arrays(path)
            self.assertEqual(states.shape, (2, 14))
            np.testing.assert_array_equal(states[1], actions[0])
            self.assertEqual(states[0, 0], 0)
            self.assertEqual(actions[0, 0], 6)


if __name__ == "__main__":
    unittest.main()
