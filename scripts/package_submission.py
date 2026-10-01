#!/usr/bin/env python3
"""Validate Tianchi's supplied results schema and package a trained run locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.training_identity import base_checkpoint_identity
TEMPLATE = ROOT / "docs/official-templates/初赛评测结果模板_JSON版.json"


def validate_results(payload: dict, *, require_observations: bool = True) -> None:
    template = json.loads(TEMPLATE.read_text())
    if set(payload) != set(template) or payload.get("schema_version") != 1:
        raise ValueError("Expected the supplied schema_version=1/team_id/results structure")
    team = payload.get("team_id")
    if not isinstance(team, str) or not team.strip() or team == template["team_id"]:
        raise ValueError("Provide your actual Tianchi team_id")
    if set(payload["results"]) != set(template["results"]):
        raise ValueError("Both clean and randomized settings are required")
    total = 0
    for setting, tasks in template["results"].items():
        actual = payload["results"][setting]
        if set(actual) != set(tasks):
            raise ValueError(f"{setting} must contain the exact official 50 task keys")
        for task, row in actual.items():
            if set(row) != {"attempts", "successes"}:
                raise ValueError(f"Invalid result fields: {setting}/{task}")
            attempts, successes = row["attempts"], row["successes"]
            if type(attempts) is not int or type(successes) is not int or not 0 <= successes <= attempts <= 100:
                raise ValueError(f"Require integer 0 <= successes <= attempts <= 100: {setting}/{task}")
            total += attempts
    if require_observations and total == 0:
        raise ValueError("The empty official template contains no measured evaluation results")


def iter_code_files(root: Path):
    for name in ("README.md", "train.sh", "eval.sh", "pyproject.toml", "requirements-extra.txt"):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required code material missing: {path}")
        yield path
    for directory in ("scripts", "src", "configs", "docs/official-templates"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts) and path.suffix != ".pyc":
                yield path
    research = root / "docs/competition-research.md"
    if research.is_file():
        yield research
    qwen = root / "models/Qwen3-VL-4B-Instruct"
    for path in sorted(qwen.glob("*")):
        if path.is_file() and path.suffix in {".json", ".txt", ".model"}:
            yield path


def validate_evaluation_binding(results_path: Path, checkpoint: Path) -> dict:
    """Associate the official counts with the evaluated bytes, even after moving."""
    directory = results_path.parent
    plan_path, detail_path = directory / "evaluation_plan.json", directory / "results.json"
    if not plan_path.is_file() or not detail_path.is_file():
        raise ValueError("Results need evaluation_plan.json and detailed results.json from the same evaluation")
    plan, detail = json.loads(plan_path.read_text()), json.loads(detail_path.read_text())
    if not plan.get("fixed_checkpoint") or not plan.get("checkpoint_identity"):
        raise ValueError("Evaluation must attest one fixed checkpoint's actual shard bytes")
    if plan.get("precision") != detail.get("precision"):
        raise ValueError("Evaluation precision differs between the plan and results")
    actual = json.loads(results_path.read_text())["results"]
    expected = {setting: {task: {"attempts": 0, "successes": 0} for task in rows}
                for setting, rows in actual.items()}
    for setting, label in (("demo_clean", "clean"), ("demo_randomized", "randomized")):
        for row in detail.get("settings", {}).get(setting, {}).get("tasks", []):
            expected[label][row["task"]] = {"attempts": row["trials"], "successes": row["successes"]}
    if actual != expected:
        raise ValueError("Official counts differ from the detailed measured evaluation")
    if plan["checkpoint_identity"] != base_checkpoint_identity(checkpoint):
        raise ValueError("Measured results belong to different checkpoint weights")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Measured results matching the supplied official JSON")
    parser.add_argument("--run", type=Path, required=True, help="Trained run containing metadata/configs/checkpoints")
    parser.add_argument("--report", type=Path, required=True, help="Completed reproduction report using the supplied template")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-vendor", action="store_true", help="Reclone pinned upstream sources using setup scripts")
    args = parser.parse_args()
    validate_results(json.loads(args.results.read_text()))
    run = args.run.resolve()
    if not (run / "lingbotvla_cli.yaml").is_file():
        parser.error("--run must contain lingbotvla_cli.yaml")
    metadata = json.loads((run / "run_metadata.json").read_text())
    if metadata.get("competition_all_50_tasks") is not True or metadata.get("training_method") != "clean_only_behavior_cloning":
        parser.error("Submission requires one jointly trained 50-task clean-only BC model; subset smoke runs are incomplete")
    checkpoints = list(run.glob("checkpoints/global_step_*/hf_ckpt/export.json"))
    if len(checkpoints) != 1:
        parser.error("Select a run containing exactly one merged exported checkpoint")
    validate_evaluation_binding(args.results, checkpoints[0].parent)
    if not args.report.is_file() or not args.report.read_text().strip():
        parser.error("--report must be a completed reproduction report")
    if args.output.exists():
        parser.error("Output already exists; choose a new archive name")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.write(args.results, "01_评测结果/results.json")
            for name in ("evaluation_plan.json", "results.json"):
                archive.write(args.results.parent / name, "01_评测结果/provenance/" + name)
            archive.write(args.report, "04_复现与调优说明/reproduction_report.md")
            for path in iter_code_files(ROOT):
                archive.write(path, "02_代码材料/" + path.relative_to(ROOT).as_posix())
            if not args.no_vendor:
                from lrvla.vendor_sources import snapshot_metadata
                revisions = {"lingbot-vla-v2": "be969b8fd117fb70550c5d4bf4bc328211b5b1b6",
                             "RoboTwin": "13c3c47ff4312dd62484bcd51be034af55c062d1"}
                for repo in ("lingbot-vla-v2", "RoboTwin"):
                    directory = ROOT / "vendor" / repo
                    snapshot = snapshot_metadata(directory, revisions[repo])
                    for relative in snapshot["files"]:
                        path = directory / relative
                        if path.is_file():
                            archive.write(path, "02_代码材料/" + path.relative_to(ROOT).as_posix())
                    archive.writestr("02_代码材料/vendor/" + repo + "/.lrvla_snapshot.json",
                                     json.dumps(snapshot, indent=2))
            for path in sorted(run.rglob("*")):
                if path.is_file() and path.suffix in {".safetensors", ".json", ".yaml", ".yml", ".txt"}:
                    archive.write(path, "03_模型材料/run/" + path.relative_to(run).as_posix())
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Local archive: {args.output.resolve()} ({args.output.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
