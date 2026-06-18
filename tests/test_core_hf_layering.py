from __future__ import annotations

import ast
from pathlib import Path


def test_core_hf_does_not_import_system_modules() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "mean_field" / "core" / "hf"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "mean_field.systems" or alias.name.startswith("mean_field.systems."):
                        offenders.append(f"{path.relative_to(root.parents[3])}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "mean_field.systems" or module.startswith("mean_field.systems."):
                    offenders.append(f"{path.relative_to(root.parents[3])}: from {module} import ...")

    assert offenders == []
