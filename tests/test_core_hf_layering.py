from __future__ import annotations

import ast
import importlib
import importlib.util
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


def test_core_hf_root_is_an_empty_namespace_and_tracked_code_uses_owners() -> None:
    repository = Path(__file__).resolve().parents[1]
    source_root = repository / "src"
    init_path = source_root / "mean_field" / "core" / "hf" / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    assert len(init_tree.body) == 2
    assert isinstance(init_tree.body[0], ast.Expr)
    assert isinstance(init_tree.body[0].value, ast.Constant)
    assert isinstance(init_tree.body[0].value.value, str)
    assert isinstance(init_tree.body[1], ast.Assign)
    assert len(init_tree.body[1].targets) == 1
    assert isinstance(init_tree.body[1].targets[0], ast.Name)
    assert init_tree.body[1].targets[0].id == "__all__"
    assert ast.literal_eval(init_tree.body[1].value) == ()

    package = importlib.import_module("mean_field.core.hf")
    assert package.__all__ == ()

    offenders: list[str] = []
    for search_root in (source_root, repository / "tests", repository / "scripts"):
        for path in sorted(search_root.rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            package_name: str | None = None
            if path.is_relative_to(source_root):
                relative = path.relative_to(source_root).with_suffix("")
                parts = relative.parts
                module_name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
                package_name = (
                    module_name if parts[-1] == "__init__" else module_name.rpartition(".")[0]
                )
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    resolved = node.module or ""
                    if node.level and package_name:
                        resolved = importlib.util.resolve_name(
                            "." * node.level + resolved,
                            package_name,
                        )
                    if resolved == "mean_field.core.hf":
                        offenders.append(f"{path.relative_to(repository)}: flat from-import")
                    if resolved == "mean_field.core" and any(
                        alias.name == "hf" for alias in node.names
                    ):
                        offenders.append(f"{path.relative_to(repository)}: parent package import")
                elif isinstance(node, ast.Import) and any(
                    alias.name == "mean_field.core.hf" for alias in node.names
                ):
                    offenders.append(f"{path.relative_to(repository)}: flat package import")

    assert offenders == []
