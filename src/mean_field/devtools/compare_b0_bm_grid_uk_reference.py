from __future__ import annotations

import argparse

import numpy as np

from mean_field import load_b0_suite
from mean_field.benchmarks import load_complex_tensor4_tsv
from mean_field.systems.tbg.zero_field import build_b0_uniform_lattice
from mean_field.systems.tbg.zero_field.model import solve_bm_model
from mean_field.systems.tbg.zero_field.runners import build_b0_reference_parameters


def compare_case(benchmark_id: str) -> None:
    case = load_b0_suite().get(benchmark_id)
    reference_path = case.bm_grid_reference_uk_path()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing BM grid Uk reference for {benchmark_id}: {reference_path}")

    params = build_b0_reference_parameters(case.theta_deg)
    grid = build_b0_uniform_lattice(params, case.lk)
    native = solve_bm_model(params, grid.kvec, lg=case.lg, sigma_rotation=True)
    reference_uk = load_complex_tensor4_tsv(reference_path, shape=native.uk.shape)

    raw_diff = native.uk - reference_uk
    raw_fro = float(np.linalg.norm(raw_diff))
    raw_max_abs = float(np.max(np.abs(raw_diff)))

    projector_diff_max = 0.0
    overlap_unitarity_max = 0.0
    aligned_diff_sq = 0.0
    min_singular = 1.0
    for ieta in range(native.n_eta):
        for ik in range(native.nk):
            native_block = native.uk[:, :, ieta, ik]
            reference_block = reference_uk[:, :, ieta, ik]
            overlap = native_block.conj().T @ reference_block
            singular = np.linalg.svd(overlap, compute_uv=False)
            min_singular = min(min_singular, float(np.min(singular)))
            overlap_unitarity_max = max(overlap_unitarity_max, float(np.max(np.abs(overlap.conj().T @ overlap - np.eye(native.nb)))))
            projector_diff = native_block @ native_block.conj().T - reference_block @ reference_block.conj().T
            projector_diff_max = max(projector_diff_max, float(np.linalg.norm(projector_diff)))
            u_mat, _, vh_mat = np.linalg.svd(overlap)
            aligned = native_block @ (u_mat @ vh_mat)
            aligned_diff_sq += float(np.linalg.norm(aligned - reference_block) ** 2)

    aligned_fro = float(np.sqrt(aligned_diff_sq))

    print(f"benchmark_id={case.benchmark_id}")
    print(f"reference_path={reference_path}")
    print(f"uk_shape={native.uk.shape}")
    print(f"raw_diff_fro={raw_fro:.12e}")
    print(f"raw_diff_max_abs={raw_max_abs:.12e}")
    print(f"aligned_diff_fro={aligned_fro:.12e}")
    print(f"projector_diff_max_fro={projector_diff_max:.12e}")
    print(f"overlap_unitarity_max_abs={overlap_unitarity_max:.12e}")
    print(f"min_overlap_singular_value={min_singular:.12e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare native Python BM eigenvectors against the Julia grid Uk benchmark reference.")
    parser.add_argument("benchmark_id", help="Benchmark case id from the bundled B0 suite.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_case(args.benchmark_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
