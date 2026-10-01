"""Recover official object descriptors after expert failure, without robot actions.

Only the pinned task's metadata dictionary is inspected. Read-only object/color
expressions refer to scene attributes already created by setup_demo. Arm-specific
descriptors are omitted so the original generator selects its official templates
without arm placeholders; every official task supplies both instruction pools.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
import re

READ_ONLY_NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Attribute, ast.Subscript,
                   ast.Slice, ast.List, ast.Tuple, ast.JoinedStr, ast.FormattedValue)


@lru_cache(maxsize=50)
def object_expressions(task: str, root: str) -> tuple[tuple[str, ast.AST], ...]:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", task):
        raise ValueError("Invalid RoboTwin task name")
    module = ast.parse((Path(root) / "envs" / f"{task}.py").read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == task)
    play = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "play_once")
    assignments = [node.value for node in ast.walk(play) if isinstance(node, ast.Assign)
                   and any(ast.unparse(target) == "self.info['info']" for target in node.targets)]
    if not assignments:
        # handover_block has no object placeholders and returns an empty info.
        return ()
    if len(assignments) != 1 or not isinstance(assignments[0], ast.Dict):
        raise ValueError("Pinned RoboTwin metadata structure changed")
    expressions = []
    for key, value in zip(assignments[0].keys, assignments[0].values):
        if not isinstance(key, ast.Constant) or not re.fullmatch(r"\{[A-Z]\}", str(key.value)):
            continue
        expression = ast.Expression(body=value)
        for node in ast.walk(expression):
            if not isinstance(node, READ_ONLY_NODES) or (isinstance(node, ast.Name) and node.id != "self"):
                raise ValueError(f"Metadata expression is not a read-only scene attribute: {task}/{key.value}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError("Private scene attributes are not allowed")
        expressions.append((key.value, expression))
    return tuple(expressions)


def recover_scene_info(task: str, environment, root: Path) -> dict[str, str]:
    descriptors = {}
    for key, expression in object_expressions(task, str(root.resolve())):
        value = eval(compile(expression, f"{task}_scene_metadata", "eval"), {"__builtins__": {}}, {"self": environment})
        if not isinstance(value, str) or not value:
            raise ValueError(f"Scene metadata is missing {key} for {task}")
        descriptors[key] = value
    return descriptors
