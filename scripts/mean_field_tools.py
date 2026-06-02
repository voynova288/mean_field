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
    "compare_tbg_crpa_fig1e": ("mean_field.devtools.compare_tbg_crpa_fig1e", ()),
    "compare_b0_lk24_python_julia_outputs": ("mean_field.devtools.compare_b0_lk24_python_julia_outputs", ()),
    "diagnose_tbg_crpa_epsilon": ("mean_field.devtools.diagnose_tbg_crpa_epsilon", ()),
    "merge_tbg_crpa_chunks": ("mean_field.devtools.merge_tbg_crpa_chunks", ()),
    "merge_rlg_hbn_parallel_hf": ("mean_field.devtools.merge_rlg_hbn_parallel_hf", ()),
    "plot_tbg_zhang_fig10_scf_grid": ("mean_field.devtools.plot_tbg_zhang_fig10_scf_grid", ()),
    "prepare_tbg_crpa_bm": ("mean_field.devtools.prepare_tbg_crpa_bm", ()),
    "resample_b0_density_stack": ("mean_field.devtools.resample_b0_density_stack", ()),
    "run_atmg_fig3_band_plot": ("mean_field.devtools.run_atmg_fig3_band_plot", ()),
    "run_b0_bm_benchmark": ("mean_field.devtools.run_b0_bm_benchmark", ()),
    "run_b0_restricted_hf_benchmark_case": ("mean_field.devtools.run_b0_restricted_hf_benchmark_case", ()),
    "run_custom_b0_hf_case": ("mean_field.devtools.run_custom_b0_hf_case", ()),
    "run_htg_hf": ("mean_field.devtools.run_htg_hf", ()),
    "run_htg_fig9b_boundary_targeted_diagnostic": (
        "mean_field.devtools.run_htg_fig9b_boundary_targeted_diagnostic",
        (),
    ),
    "run_htg_fig9b_exact_anchor_scan": ("mean_field.devtools.run_htg_fig9b_exact_anchor_scan", ()),
    "run_htg_paper_figures": ("mean_field.devtools.run_htg_paper_figures", ()),
    "run_rlg_hbn_paper_fig2_band_plot": ("mean_field.devtools.run_rlg_hbn_paper_fig2_band_plot", ()),
    "run_tbg_crpa_chunk": ("mean_field.devtools.run_tbg_crpa_chunk", ()),
    "run_tbg_crpa_convergence_scan": ("mean_field.devtools.run_tbg_crpa_convergence_scan", ()),
    "run_tbg_crpa_epsilon": ("mean_field.devtools.run_tbg_crpa_epsilon", ()),
    "run_tbg_crpa_hf_case": ("mean_field.devtools.run_tbg_crpa_hf_case", ()),
    "run_tbg_zhang_fig10": ("mean_field.devtools.run_tbg_zhang_fig10", ()),
    "run_bare_hf_framework_band_plots_against_liu_ref": (
        "mean_field.devtools.run_bare_hf_framework_band_plots_against_liu_ref",
        (),
    ),
    "run_tdbg_fig3_chern_band_plot": ("mean_field.devtools.run_tdbg_fig3_chern_band_plot", ()),
    "run_tdbg_reference_band_plot": ("mean_field.devtools.run_tdbg_reference_band_plot", ()),
    "run_tmbg_fig2_band_plot": ("mean_field.devtools.run_tmbg_fig2_band_plot", ()),
    "run_tmbg_fig2_chern_band_plot": ("mean_field.devtools.run_tmbg_fig2_chern_band_plot", ()),
    "run_tmbg_polshyn_figs1_abc": ("mean_field.devtools.run_tmbg_polshyn_figs1_abc", ()),
    "sync_benchmarks": ("mean_field.devtools.sync_benchmarks", ()),
    "sync_b0_benchmark": ("mean_field.devtools.sync_benchmarks", ("b0",)),
    "sync_b0_benchmarks": ("mean_field.devtools.sync_benchmarks", ("b0",)),
    "sync_bm_unstrained_benchmark": ("mean_field.devtools.sync_benchmarks", ("bm-unstrained",)),
    "validate_bare_hf_frameworks_against_liu_ref": (
        "mean_field.devtools.validate_bare_hf_frameworks_against_liu_ref",
        (),
    ),
    "validate_bare_split_equivalence": ("mean_field.devtools.validate_bare_split_equivalence", ()),
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
