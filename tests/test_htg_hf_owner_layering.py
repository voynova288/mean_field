from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path


_EXPECTED_EXPORTS = {
    "_hf_types": [
        "HTGGroundStateScan",
        "HTGHFPathResult",
        "HTGHartreeFockRun",
        "HTGHartreeFockState",
        "HTGInteractionComponents",
        "HTGInteractionPathResult",
        "HTGProjectedBasisData",
        "HTGSeedOccupationSummary",
        "VALLEY_SEQUENCE",
    ],
    "_hf_reference": [
        "hermitian_residual",
        "htg_band_reference_occupations",
        "htg_filling_from_density",
        "htg_gap_estimate",
        "htg_gap_from_occupation_mask",
        "htg_occupation_mask_from_density",
        "htg_occupied_bands_per_k",
        "htg_occupied_state_count",
        "htg_projector_from_density",
        "moire_cell_area_nm2",
        "projector_idempotency_residual",
    ],
    "_hf_initialization": [
        "HTGDensityBuilder",
        "HTGInitializer",
        "build_htg_density_from_hamiltonian",
        "htg_flavor_occupation_counts_for_init_mode",
        "htg_seed_occupation_summary",
        "initialize_htg_density",
        "normalize_htg_init_mode",
    ],
    "_hf_basis": [
        "build_htg_overlap_blocks",
        "build_htg_overlap_blocks_between",
        "build_htg_projected_basis",
        "build_htg_projected_basis_for_kvec",
        "centered_projection_band_indices",
        "reciprocal_shift_labels",
    ],
    "_hf_interaction_path": [
        "build_htg_interaction_components",
        "evaluate_htg_hf_path",
        "evaluate_htg_interaction_path",
    ],
    "_hf_runner": [
        "build_htg_hf_kernel",
        "build_htg_hf_problem",
        "compute_background_densities",
        "compute_background_density",
        "run_htg_hf",
        "scan_htg_ground_state",
    ],
    "_hf_contracts": [
        "HTGRunHFConfig",
        "htg_hf_run_to_hf_result",
        "htg_hf_run_to_hf_run_result",
        "run_htg_hf_config_adapter",
    ],
    "_supercell_shared": (),
    "_supercell_types": [
        "HTGSupercell",
        "HTGSupercellProjectedBasisData",
        "HTGSupercellHartreeFockState",
        "HTGSupercellHartreeFockRun",
        "HTGSupercellGroundStateScan",
        "HTGSupercellPathResult",
        "HTGSupercellSCFGridPathSamples",
        "HTGSupercellHFWavefunctionGrid",
    ],
    "_supercell_geometry": [
        "htg_tripled_fractional_supercell",
        "htg_doubled_fractional_supercell",
        "htg_common_area6_fractional_supercell",
        "htg_default_fractional_supercell",
        "htg_minimal_fractional_supercell",
        "supercell_fold_representatives",
        "build_htg_supercell_uniform_grid",
        "extract_htg_supercell_scf_grid_path",
        "extract_htg_supercell_inspection_scf_grid_path",
        "htg_supercell_full_boundary_sewing_transform",
        "htg_supercell_full_boundary_sewing_transforms",
        "build_htg_supercell_hf_wavefunction_grid",
    ],
    "_supercell_basis": [
        "htg_supercell_reference_diagonal",
        "htg_supercell_occupied_count_per_k",
        "htg_supercell_filling_from_density",
        "build_htg_supercell_projected_basis",
        "build_htg_supercell_projected_basis_for_kvec",
        "build_htg_supercell_overlap_blocks",
        "build_htg_supercell_overlap_blocks_between",
    ],
    "_supercell_runner": [
        "HTGSupercellDensityBuilder",
        "HTGSupercellInitializer",
        "initialize_htg_supercell_density",
        "build_htg_supercell_hf_kernel",
        "build_htg_supercell_hf_problem",
        "run_htg_supercell_hf",
        "scan_htg_supercell_ground_state",
    ],
    "_supercell_path_io": [
        "build_htg_supercell_gamma_path",
        "evaluate_htg_supercell_hf_path",
        "save_htg_supercell_run_npz",
        "save_htg_supercell_path_npz",
    ],
}

_SUPERCELL_FACADE_OWNERS = {
    "_supercell_types": _EXPECTED_EXPORTS["_supercell_types"],
    "_supercell_geometry": _EXPECTED_EXPORTS["_supercell_geometry"],
    "_supercell_basis": _EXPECTED_EXPORTS["_supercell_basis"],
    "_supercell_runner": _EXPECTED_EXPORTS["_supercell_runner"][2:],
    "_supercell_path_io": _EXPECTED_EXPORTS["_supercell_path_io"],
}

_SUPERCELL_CONTRACT_EXPORTS = [
    "HTGSupercellRunHFConfig",
    "htg_supercell_hf_run_to_hf_result",
    "htg_supercell_hf_run_to_hf_run_result",
    "run_htg_supercell_hf_config_adapter",
]


def test_htg_hf_owners_have_explicit_dependencies_and_exports() -> None:
    owner_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mean_field"
        / "systems"
        / "htg"
    )
    for module_name, expected in _EXPECTED_EXPORTS.items():
        module = import_module(f"mean_field.systems.htg.{module_name}")
        assert module.__all__ == expected

        tree = ast.parse(
            (owner_dir / f"{module_name}.py").read_text(encoding="utf-8")
        )
        assert not any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "globals"
            for node in ast.walk(tree)
        )

def test_htg_supercell_facade_preserves_owner_identity_and_private_bridges() -> None:
    facade = import_module("mean_field.systems.htg.supercell")
    contracts = import_module("mean_field.systems.htg.supercell_contracts")
    basis = import_module("mean_field.systems.htg._supercell_basis")
    runner = import_module("mean_field.systems.htg._supercell_runner")

    expected_facade_exports = [
        name
        for owner_exports in _SUPERCELL_FACADE_OWNERS.values()
        for name in owner_exports
    ]
    assert len(expected_facade_exports) == 36
    assert set(facade.__all__) == set(expected_facade_exports)
    assert len(facade.__all__) == len(expected_facade_exports)
    assert contracts.__all__ == _SUPERCELL_CONTRACT_EXPORTS

    for owner_name, exports in _SUPERCELL_FACADE_OWNERS.items():
        owner = import_module(f"mean_field.systems.htg.{owner_name}")
        for name in exports:
            assert getattr(facade, name) is getattr(owner, name)

    reference_builder = basis._supercell_reference_density_blocks
    assert facade._supercell_reference_density_blocks is reference_builder
    assert runner._supercell_reference_density_blocks is reference_builder
    assert contracts._supercell_reference_density_blocks is reference_builder
