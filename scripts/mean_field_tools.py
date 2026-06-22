#!/usr/bin/env python3

from __future__ import annotations

import importlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODULE_COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {}


def _normalize_command(text: str) -> str:
    return text.removesuffix(".py").replace("-", "_")


def _print_help() -> int:
    print("Usage: python scripts/mean_field_tools.py <command> [args...]")
    print("")
    print("Tool commands:")
    for name in sorted(MODULE_COMMANDS):
        print(f"  {name}")
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

    if command in MODULE_COMMANDS:
        module_name, argv_prefix = MODULE_COMMANDS[command]
        return _run_module(module_name, argv_prefix, rest)

    raise SystemExit(f"Unknown command: {args[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
