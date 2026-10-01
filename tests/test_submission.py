import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_submission", ROOT / "scripts/package_submission.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def measured():
    data = json.loads(module.TEMPLATE.read_text())
    data["team_id"] = "test-team"
    data["results"]["clean"]["adjust_bottle"] = {"attempts": 2, "successes": 1}
    return data


def test_official_schema_accepts_measured_partial_counts():
    module.validate_results(measured())


def test_empty_template_cannot_be_packaged_as_measured_result():
    data = measured()
    data["results"]["clean"]["adjust_bottle"] = {"attempts": 0, "successes": 0}
    with pytest.raises(ValueError, match="no measured"):
        module.validate_results(data)


@pytest.mark.parametrize("row", [{"attempts": 1, "successes": 2}, {"attempts": True, "successes": 0}, {"attempts": 101, "successes": 0}])
def test_invalid_success_counts_rejected(row):
    data = measured()
    data["results"]["clean"]["adjust_bottle"] = row
    with pytest.raises(ValueError, match="integer"):
        module.validate_results(data)


def test_missing_task_rejected():
    data = measured()
    del data["results"]["randomized"]["lift_pot"]
    with pytest.raises(ValueError, match="50 task"):
        module.validate_results(data)


def test_code_package_preserves_official_docs_and_excludes_local_notes(tmp_path):
    required = {"README.md", "train.sh", "eval.sh", "pyproject.toml", "requirements-extra.txt"}
    published = {"scripts/train.py", "src/lrvla/training.py", "configs/train_lora.yaml",
                 "docs/official-templates/初赛评测结果模板_JSON版.json", "docs/competition-research.md",
                 "models/Qwen3-VL-4B-Instruct/config.json"}
    local = {"docs/commit-conventions.md", "docs/AGENTS.md", "docs/local/setup-notes.md"}
    for name in required | published | local:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    packaged = {path.relative_to(tmp_path).as_posix() for path in module.iter_code_files(tmp_path)}
    assert packaged == required | published


def evaluation_files(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"evaluated weights")
    results = tmp_path / "official_results.json"
    results.write_text(json.dumps(measured()))
    (tmp_path / "evaluation_plan.json").write_text(json.dumps({
        "fixed_checkpoint": True, "precision": "bf16",
        "checkpoint_identity": module.base_checkpoint_identity(checkpoint),
    }))
    (tmp_path / "results.json").write_text(json.dumps({"precision": "bf16", "settings": {
        "demo_clean": {"tasks": [{"task": "adjust_bottle", "trials": 2, "successes": 1}]},
    }}))
    return results, checkpoint


def test_evaluation_binding_survives_checkpoint_relocation(tmp_path):
    results, checkpoint = evaluation_files(tmp_path)
    moved = checkpoint.rename(tmp_path / "moved-model")
    module.validate_evaluation_binding(results, moved)


def test_evaluation_cannot_be_packaged_with_changed_weights(tmp_path):
    results, checkpoint = evaluation_files(tmp_path)
    (checkpoint / "model.safetensors").write_bytes(b"other model")
    with pytest.raises(ValueError, match="different checkpoint"):
        module.validate_evaluation_binding(results, checkpoint)


def test_evaluation_cannot_be_packaged_with_unrelated_counts(tmp_path):
    results, checkpoint = evaluation_files(tmp_path)
    changed = measured()
    changed["results"]["clean"]["adjust_bottle"]["successes"] = 2
    results.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="counts differ"):
        module.validate_evaluation_binding(results, checkpoint)
