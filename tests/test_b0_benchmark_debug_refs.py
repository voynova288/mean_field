from pathlib import Path

import numpy as np

from mean_field.reference_oracles import load_b0_suite, load_complex_stack_tsv, load_complex_tensor4_tsv


def test_benchmark_case_debug_reference_paths() -> None:
    case = load_b0_suite().get("theta_120_nu_-2_ivc_ground")
    assert case.initial_density_override_path().name == "initial_density_sp_seed_001.tsv"
    assert case.reference_first_iteration_interaction_path().name == "reference_first_iteration_interaction_sp_seed_001.tsv"
    assert case.reference_first_iteration_hamiltonian_path().name == "reference_first_iteration_hamiltonian_sp_seed_001.tsv"
    assert case.reference_first_iteration_density_path().name == "reference_first_iteration_density_sp_seed_001.tsv"
    assert case.reference_first_iteration_summary_path().name == "reference_first_iteration_summary_sp_seed_001.txt"
    assert case.reference_iteration_input_density_path(10).name == "reference_iteration_010_input_density_sp_seed_001.tsv"
    assert case.reference_iteration_interaction_path(10).name == "reference_iteration_010_interaction_sp_seed_001.tsv"
    assert case.reference_iteration_hamiltonian_path(10).name == "reference_iteration_010_hamiltonian_sp_seed_001.tsv"
    assert case.reference_iteration_updated_density_path(10).name == "reference_iteration_010_updated_density_sp_seed_001.tsv"
    assert case.reference_iteration_summary_path(10).name == "reference_iteration_010_summary_sp_seed_001.txt"
    assert case.bm_grid_reference_uk_path().name == "bm_theta_120_lk19_lg9_uk_reference.tsv"
    assert case.bm_grid_reference_uk_summary_path().name == "bm_theta_120_lk19_lg9_uk_reference_summary.txt"
    assert case.bm_grid_reference_uk_path(lk=24).name == "bm_theta_120_lk24_lg9_uk_reference.tsv"
    assert case.bm_grid_reference_uk_summary_path(lk=24).name == "bm_theta_120_lk24_lg9_uk_reference_summary.txt"


def test_load_complex_stack_tsv_infers_shape(tmp_path: Path) -> None:
    path = tmp_path / "stack.tsv"
    path.write_text(
        "# nrow=2\n"
        "# ncol=3\n"
        "# nk=2\n"
        "0\t0\t0\t1.0\t0.5\n"
        "0\t1\t2\t-2.0\t0.0\n"
        "1\t0\t1\t0.0\t-3.0\n",
        encoding="utf-8",
    )

    stack = load_complex_stack_tsv(path)
    assert stack.shape == (2, 3, 2)
    assert stack[0, 0, 0] == 1.0 + 0.5j
    assert stack[1, 2, 0] == -2.0 + 0.0j
    assert stack[0, 1, 1] == 0.0 - 3.0j
    assert np.count_nonzero(stack) == 3


def test_load_complex_stack_tsv_respects_explicit_shape(tmp_path: Path) -> None:
    path = tmp_path / "stack.tsv"
    path.write_text("0\t0\t0\t1.0\t0.0\n", encoding="utf-8")

    stack = load_complex_stack_tsv(path, shape=(3, 3, 2))
    assert stack.shape == (3, 3, 2)
    assert stack[0, 0, 0] == 1.0 + 0.0j


def test_load_complex_tensor4_tsv_infers_shape(tmp_path: Path) -> None:
    path = tmp_path / "tensor4.tsv"
    path.write_text(
        "# n0=2\n"
        "# n1=2\n"
        "# n2=2\n"
        "# n3=2\n"
        "0\t0\t0\t0\t1.0\t0.5\n"
        "1\t0\t1\t0\t-2.0\t0.0\n"
        "0\t1\t0\t1\t0.0\t-3.0\n",
        encoding="utf-8",
    )

    tensor = load_complex_tensor4_tsv(path)
    assert tensor.shape == (2, 2, 2, 2)
    assert tensor[0, 0, 0, 0] == 1.0 + 0.5j
    assert tensor[1, 0, 1, 0] == -2.0 + 0.0j
    assert tensor[0, 1, 0, 1] == 0.0 - 3.0j
    assert np.count_nonzero(tensor) == 3
