#!/usr/bin/env python3

from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mean_field.cli import main as cli_main


CLI_GROUPS = {"benchmarks", "bm", "hf", "tmbg"}
CLI_ALIASES = {
    "run_tmbg_checkpoints": ("tmbg", "reproduce-checkpoints"),
    "run_tmbg_ktilde_diagnostics": ("tmbg", "diagnose-ktilde-symmetry"),
}
MODULE_COMMANDS = {
    "backfill_canonical_hf_sidecars": ("mean_field.devtools.backfill_canonical_hf_sidecars", ()),
    "compare_tbg_crpa_fig1e": ("mean_field.devtools.compare_tbg_crpa_fig1e", ()),
    "merge_rlg_hbn_parallel_hf": ("mean_field.devtools.merge_rlg_hbn_parallel_hf", ()),

    "merge_tbg_crpa_chunks": ("mean_field.devtools.merge_tbg_crpa_chunks", ()),
    "prepare_tbg_crpa_bm": ("mean_field.devtools.prepare_tbg_crpa_bm", ()),
    "run_rlg_hbn_paper_hf": ("mean_field.devtools.run_rlg_hbn_paper_hf", ()),
    "run_rlg_hbn_tdhf_q0": ("mean_field.devtools.run_rlg_hbn_tdhf_q0", ()),
    "run_rlg_hbn_tdhf_finite_q": ("mean_field.devtools.run_rlg_hbn_tdhf_finite_q", ()),
    "run_tbg_crpa_chunk": ("mean_field.devtools.run_tbg_crpa_chunk", ()),
    "sync_benchmarks": ("mean_field.devtools.sync_benchmarks", ()),
    "sync_b0_benchmark": ("mean_field.devtools.sync_benchmarks", ("b0",)),
    "sync_b0_benchmarks": ("mean_field.devtools.sync_benchmarks", ("b0",)),
    "sync_bm_unstrained_benchmark": ("mean_field.devtools.sync_benchmarks", ("bm-unstrained",)),
    "validate_rlg_hbn_fig6_prereqs": ("mean_field.devtools.validate_rlg_hbn_fig6_prereqs", ()),
    "validate_tbg_crpa_artifact": ("mean_field.devtools.validate_tbg_crpa_artifact", ()),
}


def _normalize_command(text: str) -> str:
    return text.removesuffix(".py").replace("-", "_")


def _print_help() -> int:
    print("Usage: python scripts/mean_field_tools.py <command> [args...]")
    print("")
    print("CLI groups:")
    for name in sorted(CLI_GROUPS):
        print(f"  {name}")
    print("")
    print("Tool commands:")
    for name in sorted(MODULE_COMMANDS):
        print(f"  {name}")
    print("")
    print("CLI aliases:")
    for name, prefix in sorted(CLI_ALIASES.items()):
        print(f"  {name} -> {' '.join(prefix)}")
    return 0


def _run_module(module_name: str, argv_prefix: tuple[str, ...], argv: list[str]) -> int:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise SystemExit(f"Module {module_name} does not expose main().")

    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0], *argv_prefix, *argv]
    try:
        result = module.main()
    finally:
        sys.argv = saved_argv

    return 0 if result is None else int(result)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"help", "--help", "-h"} or _normalize_command(args[0]) == "help":
        return _print_help()

    command = _normalize_command(args[0])
    rest = args[1:]

    if command in CLI_GROUPS:
        return int(cli_main([command, *rest]))

    if command in CLI_ALIASES:
        prefix = CLI_ALIASES[command]
        return int(cli_main([*prefix, *rest]))

    if command in MODULE_COMMANDS:
        module_name, argv_prefix = MODULE_COMMANDS[command]
        return _run_module(module_name, argv_prefix, rest)

    raise SystemExit(f"Unknown command: {args[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
