from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = REPO_ROOT / "scripts" / "mean_field_tools.py"
OLD_COMMAND = "run_rlg_hbn_tdhf_finite_q"
RETAINED_COMMANDS = {
    "merge_tbg_crpa_chunks",
    "qualify_tpt_bulk_parent_wavefunctions",
    "run_tbg_crpa_chunk",
    "tpt_bulk_paw_oracle",
    "tpt_wannier_q0_hf",
}
RETIRED_RUNTIME = "mean_field.devtools._runtime"
MAINTAINED_DEVTOOL_MODULES = tuple(sorted(RETAINED_COMMANDS))


def _load_dispatcher():
    spec = importlib.util.spec_from_file_location("_surface_dispatcher", DISPATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_dispatcher(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DISPATCHER), *args], cwd=REPO_ROOT,
        check=False, capture_output=True, text=True,
    )


def test_dispatcher_is_exactly_the_five_crpa_tpt_devtools() -> None:
    dispatcher = _load_dispatcher()
    source = DISPATCHER.read_text(encoding="utf-8")

    assert set(dispatcher.MODULE_COMMANDS) == RETAINED_COMMANDS
    assert not hasattr(dispatcher, "CLI_GROUPS")
    assert not hasattr(dispatcher, "CLI_ALIASES")
    assert not hasattr(dispatcher, "cli_main")
    assert "mean_field.cli" not in source
    assert "CLI_GROUPS" not in source
    assert "CLI_ALIASES" not in source


def test_retired_rlg_hbn_finite_q_surface_remains_absent() -> None:
    assert importlib.util.find_spec(f"mean_field.devtools.{OLD_COMMAND}") is None
    assert OLD_COMMAND not in DISPATCHER.read_text(encoding="utf-8")


def test_devtool_runtime_owner_is_retired_from_all_maintained_callers() -> None:
    assert importlib.util.find_spec(RETIRED_RUNTIME) is None
    devtools = REPO_ROOT / "src" / "mean_field" / "devtools"
    for module_name in MAINTAINED_DEVTOOL_MODULES:
        source = (devtools / f"{module_name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "mean_field.devtools._runtime" not in source
        assert (
            "mean_field.runtime",
            "ensure_not_running_compute_on_login_node",
        ) in imports
        if module_name in {
            "qualify_tpt_bulk_parent_wavefunctions",
            "tpt_bulk_paw_oracle",
        }:
            assert ("mean_field.core.io", "write_json_artifact") in imports


def test_dispatcher_help_is_devtool_only() -> None:
    completed = _run_dispatcher("--help")

    assert completed.returncode == 0
    assert "Developer-tool commands:" in completed.stdout
    assert "CLI groups:" not in completed.stdout
    assert "CLI aliases:" not in completed.stdout
    assert "tdbg" not in completed.stdout
    for command in RETAINED_COMMANDS:
        assert command in completed.stdout


def test_each_retained_dispatcher_command_help_runs_in_a_subprocess() -> None:
    for command in sorted(RETAINED_COMMANDS):
        completed = _run_dispatcher(command, "--help")
        assert completed.returncode == 0, (command, completed.stderr)
        assert "usage:" in completed.stdout.lower()


def test_dispatcher_rejects_retired_rlg_and_tdbg_package_commands() -> None:
    for command in (OLD_COMMAND, "tdbg"):
        completed = _run_dispatcher(command)
        assert completed.returncode != 0
        assert completed.stdout == ""
        assert completed.stderr.strip() == f"Unknown command: {command}"


def test_canonical_package_cli_entry_help_and_parse_remain_available() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["mean-field"] == "mean_field.cli:main"

    completed = subprocess.run(
        [sys.executable, "-m", "mean_field.cli", "tdbg", "projected-hf", "--help"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert "usage: mean-field tdbg projected-hf" in completed.stdout

    from mean_field.cli import build_parser

    parsed = build_parser().parse_args(
        ["tdbg", "projected-hf", "config.json", "--dry-run"]
    )
    assert (parsed.command, parsed.tdbg_command) == ("tdbg", "projected-hf")
    assert parsed.config_path == Path("config.json")
    assert parsed.dry_run is True
