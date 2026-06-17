from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mean_field.api import load_result, required_artifact_files
from mean_field.benchmarks import BMUnstrainedReference
from mean_field.core.lattice import KPath
from mean_field.runtime import RuntimeEnvironment
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import (
    BMUnstrainedBenchmarkRun,
    BMUnstrainedParity,
    BMUnstrainedRun,
    BMUnstrainedRuntime,
    BMUnstrainedRuntimeParity,
    BMSolution,
    complex_to_pair,
    write_bm_unstrained_benchmark_contract_sidecars,
)


def _runtime_environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        hostname="test-node",
        cpu_model="unit-test-cpu",
        slurm_partition="debug",
        slurm_nodelist="test001",
        slurm_cpus_per_task=1,
        blas_threads=1,
        numba_threads=1,
        sys_cpu_threads=1,
        process_count=1,
        backend_choice="numpy",
        threadpoolctl_info=(),
        thread_env={},
        jit_warmup_included=False,
        python_version="3.test",
        numpy_version=np.__version__,
    )


def _bm_solution(params: TBGParameters, *, nk: int = 2) -> BMSolution:
    lg = 1
    dim = 4 * lg * lg
    nb = 2
    n_eta = 2
    n_spin = 2
    nt = n_spin * n_eta * nb
    return BMSolution(
        params=params,
        lattice_kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.2j], dtype=np.complex128)[:nk],
        lg=lg,
        nlocal=4,
        n_eta=n_eta,
        n_spin=n_spin,
        nb=nb,
        hamiltonian=np.zeros((dim, dim, n_eta, nk), dtype=np.complex128),
        sigma_z=np.zeros((nt, nt, nk), dtype=np.complex128),
        uk=np.zeros((dim, nb, n_eta, nk), dtype=np.complex128),
        spectrum=np.asarray([[[-10.0, -9.0], [1.0, 2.0]], [[3.0, 4.0], [10.0, 11.0]]], dtype=float)[:, :, :nk],
        gvec=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        periodic_g_grid=True,
    )


def _bm_benchmark_result(tmp_path: Path) -> BMUnstrainedBenchmarkRun:
    params = TBGParameters.from_degrees(1.05)
    path = KPath(
        kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.0j], dtype=np.complex128),
        kdist=np.asarray([0.0, 0.1], dtype=float),
        labels=("K", "G"),
        node_indices=(1, 2),
    )
    run = BMUnstrainedRun(
        params=params,
        path=path,
        path_solution=_bm_solution(params),
        grid_solution=_bm_solution(params),
        k_middle_gap_mev=1.25,
        valence_bandwidth_mev=2.5,
        conduction_bandwidth_mev=3.5,
        runtime=BMUnstrainedRuntime(
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:01",
            path_elapsed_sec=0.1,
            grid_elapsed_sec=0.2,
            total_elapsed_sec=0.3,
            environment=_runtime_environment(),
        ),
    )
    reference = BMUnstrainedReference(
        theta_deg=1.05,
        root=tmp_path / "reference",
        summary_path=tmp_path / "reference" / "summary.txt",
        path_nodes_path=tmp_path / "reference" / "nodes.tsv",
        path_tsv_path=tmp_path / "reference" / "path.tsv",
        grid_kvec_path=tmp_path / "reference" / "grid.tsv",
    )
    parity = BMUnstrainedParity(
        kdist_max_abs_diff=0.0,
        max_abs_band_diff_mev=0.01,
        rms_band_diff_mev=0.005,
        mean_abs_band_diff_mev=0.003,
        k_middle_gap_diff_mev=0.002,
        valence_bandwidth_diff_mev=0.004,
        conduction_bandwidth_diff_mev=0.006,
    )
    runtime_parity = BMUnstrainedRuntimeParity(
        path_elapsed_sec_delta=0.0,
        path_elapsed_sec_ratio=1.0,
        grid_elapsed_sec_delta=0.0,
        grid_elapsed_sec_ratio=1.0,
        total_elapsed_sec_delta=0.0,
        total_elapsed_sec_ratio=1.0,
    )
    return BMUnstrainedBenchmarkRun(
        reference=reference,
        run=run,
        parity=parity,
        runtime_reference=None,
        runtime_parity=runtime_parity,
    )


def test_tbg_zero_field_bm_unstrained_contract_sidecars_are_metadata_only(tmp_path: Path) -> None:
    result = _bm_benchmark_result(tmp_path)
    output_dir = tmp_path / "bm"
    output_dir.mkdir()
    path_tsv = output_dir / "path.tsv"
    plot_png = output_dir / "bands.png"
    path_tsv.write_text("0.0\t1.0\n", encoding="utf-8")
    plot_png.write_text("not a real png for metadata-only test\n", encoding="utf-8")

    paths = write_bm_unstrained_benchmark_contract_sidecars(
        output_dir,
        result,
        artifact_paths={"path_tsv": path_tsv, "band_plot_png": plot_png},
    )

    assert paths["manifest.json"] == output_dir / "manifest.json"
    assert {path.name for path in output_dir.iterdir()} >= set(required_artifact_files()) | {"path.tsv", "bands.png"}
    loaded = load_result(output_dir)
    assert loaded.manifest["metadata"]["workflow"] == "tbg.zero_field.bm_unstrained_benchmark"
    assert loaded.manifest["metadata"]["runner_kind"] == "bm_unstrained_benchmark"
    assert "array_summaries" not in loaded.manifest["metadata"]
    assert loaded.manifest["files"]["path_tsv"] == "path.tsv"
    assert loaded.manifest["files"]["band_plot_png"] == "bands.png"
    assert loaded.conventions is not None and loaded.conventions["density_convention"] == "not_applicable"
    assert loaded.conventions["energy_unit"] == "meV"
    assert loaded.validation is not None and loaded.validation["status"] == "recorded"
    assert loaded.observables is not None and loaded.observables["theta_deg"] == 1.05
    assert loaded.observables["path_solution"]["spectrum_shape"] == [2, 2, 2]
    assert loaded.model is not None and loaded.model["lattice"]["g1_nm_inv_pair"] == complex_to_pair(result.run.params.g1)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_bm_unstrained_benchmark_contract_sidecars(output_dir, result)
