"""The two-stage reproduction chain of the submitted checkpoint stays connected."""
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_stage_one_is_the_five_epoch_recipe_and_feeds_stage_two():
    base, stage1, stage2 = load("train_expert_full.yaml"), load("train_expert_full_5epochs.yaml"), load("train_expert_full_filter.yaml")
    changed = {key for key in base.keys() | stage1.keys() if base.get(key) != stage1.get(key)}
    assert changed == {"output_dir", "max_steps", "save_every"}
    assert stage1["max_steps"] == 38620 and stage1["val_episodes"] == 5 and stage1["lr"] == 5e-5 and stage1["seed"] == 42
    assert stage1["image_augment"] is False and stage1["resample_seen_instructions"] is False
    assert stage2["init_from"] == f"{stage1['output_dir']}/checkpoints/global_step_{stage1['max_steps']}/hf_ckpt"
    assert stage2["max_steps"] == 6000 and stage2["output_dir"] == "runs/expert-filter"


def test_reproduction_script_runs_both_stages_in_order():
    script = (ROOT / "scripts/reproduce_submission.sh").read_text()
    subprocess.run(["bash", "-n", str(ROOT / "scripts/reproduce_submission.sh")], check=True)
    order = ["prepare_data.py --tasks all\n", "--config configs/train_expert_full_5epochs.yaml", "filter_demos.py scan",
             "filter_demos.py manifest", "--config configs/train_expert_full_filter.yaml"]
    positions = [script.index(item) for item in order]
    assert positions == sorted(positions)
    assert "global_step_6000/hf_ckpt" in script
