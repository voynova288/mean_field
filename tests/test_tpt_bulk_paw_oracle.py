from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.core.io import read_json_artifact
from mean_field.devtools import qualify_tpt_bulk_parent_wavefunctions as qualifier
from mean_field.devtools import tpt_bulk_paw_oracle as oracle


_PARENT_INCAR = """SYSTEM = test
ENCUT = 300
ALGO = Nothing
ISTART = 1
ICHARG = 0
parallel
#KPAR = 4
#NCORE = 16
LWAVE = .TRUE.
LCHARG = .FALSE.
ISYM = -1
LSORBIT = .TRUE.
NBANDS = 600
NUM_WANN = 320
LWANNIER90 = .TRUE.
WANNIER90_WIN = \"
kmesh_tol = 0.0001
num_iter = 0
guiding_centres = true
\"
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _assert_strict_json_artifact(
    path: Path,
    expected: dict[str, object],
    *,
    readonly: bool,
) -> dict[str, object]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"NaN" not in raw
    assert b"Infinity" not in raw
    loaded = read_json_artifact(path)
    assert loaded == expected
    assert isinstance(loaded, dict)
    if readonly:
        assert path.stat().st_mode & 0o222 == 0
    return loaded

def _selected_provenance_key_paths(owner: object) -> dict[str, list[tuple[str, ...]]]:
    found: dict[str, list[tuple[str, ...]]] = {}

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*path, str(key))
                if str(key).startswith("selected_provenance_"):
                    found.setdefault(str(key), []).append(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(owner, ())
    return found


def test_active_projection_replay_tolerance_is_parent_precision_bound() -> None:
    assert oracle.TPT_BULK_ACTIVE_MMN_PROJECTION_TOLERANCE == float(
        np.finfo(np.float32).eps
    )
    assert oracle.TPT_BULK_ACTIVE_MMN_TEXT_ROUNDING_BOUND == 5.0e-10
    assert (
        oracle.TPT_BULK_ACTIVE_MMN_PROJECTION_TOLERANCE
        > oracle.TPT_BULK_ACTIVE_MMN_TEXT_ROUNDING_BOUND
    )


def test_active_submission_receipt_is_semantically_bound(tmp_path: Path) -> None:
    job_id = 123456
    source_seal_sha256 = "a" * 64
    receipt = {
        "schema": "tpt-bulk-active16-paw-oracle-submission-v1",
        "job_id": job_id,
        "source_seal_sha256": source_seal_sha256,
        "case": "native",
        "account": "hmt03",
        "launcher": "Intel_MPI_mpirun_not_srun",
        "layout": {
            "nodes": 1,
            "allocated_cpus": 64,
            "mpi_ranks": 64,
            "ranks_per_node": 64,
            "kpar": 4,
            "ncore": 1,
            "npar": 16,
            "nbands": 16,
        },
        "authority": {
            "vasp_execution_authorized": True,
            "scientific_authority": False,
        },
    }
    path = tmp_path / f"SUBMISSION-{job_id}.json"
    path.write_text(json.dumps(receipt))
    path.chmod(0o444)
    loaded, digest = oracle._load_active_submission_receipt(
        output_root=tmp_path,
        job_id=job_id,
        source_seal_sha256=source_seal_sha256,
        case_name="native",
        expected_layout=receipt["layout"],
    )
    assert loaded == receipt
    assert digest == _sha256(path)

    path.chmod(0o644)
    receipt["layout"]["mpi_ranks"] = 32
    path.write_text(json.dumps(receipt))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="layout mismatch"):
        oracle._load_active_submission_receipt(
            output_root=tmp_path,
            job_id=job_id,
            source_seal_sha256=source_seal_sha256,
            case_name="native",
            expected_layout={**receipt["layout"], "mpi_ranks": 64},
        )


def test_render_oracle_incar_is_restart_only_and_selects_explicit_shells() -> None:
    native = oracle.render_oracle_incar(_PARENT_INCAR, oracle.ORACLE_CASES["native"])
    assert "ALGO = Nothing" in native
    assert "KPAR = 4" in native
    assert "NCORE = 1" in native
    assert "LWAVE = .FALSE." in native
    assert "LWRITE_MMN_AMN = .TRUE." in native
    assert "kmesh_tol = 1.0e-4" in native
    assert "search_shells = 12" in native
    assert "shell_list = 1, 2, 4" in native
    assert "LWANNIER90_RUN = .TRUE." not in native

    pure_g = oracle.render_oracle_incar(_PARENT_INCAR, oracle.ORACLE_CASES["pure_g"])
    assert "kmesh_tol = 1.0e-7" in pure_g
    assert "search_shells = 2200" in pure_g
    assert "shell_list = 28, 60, 2195" in pure_g

    mixed = oracle.render_oracle_incar(_PARENT_INCAR, oracle.ORACLE_CASES["mixed_qg"])
    assert "kmesh_tol = 1.0e-7" in mixed
    assert "search_shells = 6529" in mixed
    assert "shell_list = 1, 2, 6529" in mixed
    assert "skip_B1_tests" not in mixed


def test_active_transfer_plans_are_closed_exact_and_stock_packable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "POSCAR").write_bytes(b"accepted-poscar-for-planner\n")
    reciprocal = 2.0 * np.pi * np.linalg.inv(
        np.diag([3.7037999630, 18.5991001129, 13.9531002045])
    ).T
    source = SimpleNamespace(
        root=root,
        hr_sha256="583ddf207e81231dfc3641f4a9895bdc146e1d4e093bb9469776337f8b312b14",
        reciprocal_rows_angstrom_inv=reciprocal,
    )
    monkeypatch.setattr(oracle, "load_tpt_bulk_source", lambda _root: source)
    monkeypatch.setattr(
        oracle,
        "TPT_BULK_POSCAR_SHA256",
        _sha256(root / "POSCAR"),
    )

    g0 = oracle.build_active_transfer_plan(source_root=root, local_field_shell=0)
    g1 = oracle.build_active_transfer_plan(source_root=root, local_field_shell=1)
    assert g1["schema"] == "tpt-bulk-active16-paw-transfer-plan-v2"
    assert (
        g0["inventory"]["required_channel_count_including_zero"],
        g0["inventory"]["stock_vasp_export_channel_count_including_zero"],
        g0["inventory"]["discarded_stock_shell_extras"],
        g0["wannier90"]["shell_count"],
        g0["wannier90"]["batch_count"],
    ) == (285, 325, 40, 62, 34)
    assert (
        g1["inventory"]["required_channel_count_including_zero"],
        g1["inventory"]["stock_vasp_export_channel_count_including_zero"],
        g1["inventory"]["discarded_stock_shell_extras"],
        g1["wannier90"]["shell_count"],
        g1["wannier90"]["max_shell_index"],
        g1["wannier90"]["batch_count"],
    ) == (1811, 1965, 154, 338, 6529, 215)

    shell_indices = {
        tuple(record["absolute_numerator"]): record["shell_index"]
        for record in g1["wannier90"]["shells"]
    }
    assert shell_indices[(0, 4, 0)] == 28
    assert shell_indices[(0, 0, 4)] == 60
    assert shell_indices[(12, 0, 0)] == 2195
    assert shell_indices[(18, 2, 2)] == 6529

    channels = {
        tuple(channel["physical_q_numerator"]): channel
        for channel in g1["inventory"]["channels"]
    }
    assert len(channels) == 1811
    for transfer, channel in channels.items():
        reverse = tuple(-value for value in transfer)
        assert reverse in channels
        assert channels[reverse]["label"] == -channel["label"]
        reconstructed = tuple(
            q + size * g
            for q, size, g in zip(
                channel["q_mesh_shift"],
                (12, 4, 4),
                channel["g_integer"],
                strict=True,
            )
        )
        assert reconstructed == transfer
    assert g1["inventory"]["complete_q_residue_count"] == 192
    assert g1["inventory"]["inversion_closed"] is True
    assert channels[(0, 0, 0)]["source"] == "exact_paw_orthonormality_identity"
    assert oracle._canonical_q_and_g((-6, 0, 0), (12, 4, 4)) == (
        (6, 0, 0),
        (-1, 0, 0),
    )
    assert oracle._canonical_q_and_g((-18, 0, 0), (12, 4, 4)) == (
        (6, 0, 0),
        (-2, 0, 0),
    )
    assert oracle._canonical_q_and_g((0, -2, 0), (12, 4, 4)) == (
        (0, 2, 0),
        (0, -1, 0),
    )

    retained: list[tuple[int, int, int]] = []
    duplicates: list[tuple[int, int, int]] = []
    discarded: set[tuple[int, int, int]] = set()
    owner_counts: dict[tuple[int, int, int], int] = {}
    for batch in g1["wannier90"]["batches"]:
        assert len(batch["shell_list"]) == 3
        assert batch["nntot"] <= 12
        assert batch["fixed_shell_svd_rank"] == 3
        assert batch["b1_complete"] is True
        assert batch["skip_b1_tests"] is False
        assert batch["b1_metrics"]["max_abs_completeness_residual"] <= 1.0e-7
        retained.extend(map(tuple, batch["retained_physical_q_numerators"]))
        duplicates.extend(map(tuple, batch["duplicate_validation_q_numerators"]))
        discarded.update(map(tuple, batch["discarded_extra_physical_q_numerators"]))
        for occurrence in batch["shell_occurrences"]:
            key = tuple(occurrence["absolute_numerator"])
            owner_counts[key] = owner_counts.get(key, 0) + int(
                occurrence["is_canonical_owner"]
            )

    assert len(retained) == len(set(retained)) == 1810
    assert set(retained) == set(channels) - {(0, 0, 0)}
    assert len(discarded) == 154
    assert len(set(retained) | discarded) == 1964
    assert len(owner_counts) == 338
    assert set(owner_counts.values()) == {1}
    assert g1["wannier90"]["shell_occurrence_count"] == 645
    assert g1["wannier90"]["duplicate_shell_occurrence_count"] == 307
    max_batch = next(
        batch for batch in g1["wannier90"]["batches"] if 6529 in batch["shell_list"]
    )
    assert max_batch["shell_list"] == [1, 2, 6529]
    assert max_batch["nntot"] == 12
    assert duplicates
    assert all(value is False for value in g1["authority"].values())


def test_render_active_oracle_incar_uses_transformed_16_band_restart() -> None:
    rendered = oracle.render_active_oracle_incar(
        _PARENT_INCAR,
        oracle.ORACLE_CASES["native"],
    )
    for required in (
        "NBANDS = 16",
        "NELECT = 8",
        "NUM_WANN = 16",
        "LSCDM = .TRUE.",
        "ALGO = Nothing",
        "LWRITE_MMN_AMN = .TRUE.",
        "shell_list = 1, 2, 4",
        "guiding_centres = false",
    ):
        assert required in rendered
    assert "NBANDS = 600" not in rendered
    assert "LWANNIER90_RUN = .TRUE." not in rendered
    assert "guiding_centres = true" not in rendered

    test_layout = oracle.render_active_oracle_incar(
        _PARENT_INCAR,
        oracle.ORACLE_CASES["native"],
        kpar=3,
    )
    assert "\nKPAR = 3\n" in test_layout
    assert "\nKPAR = 4\n" not in test_layout


def test_render_oracle_incar_rejects_nonaccepted_restart_shape() -> None:
    with pytest.raises(ValueError, match="missing required restart tags"):
        oracle.render_oracle_incar(
            _PARENT_INCAR.replace("ALGO = Nothing", "ALGO = Fast"),
            oracle.ORACLE_CASES["native"],
        )


def test_prepare_oracle_capsule_is_source_bound_and_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    payloads = {
        "INCAR": _PARENT_INCAR.encode(),
        "KPOINTS": b"accepted-kpoints\n",
        "POSCAR": b"accepted-poscar\n",
        "POTCAR": b"accepted-potcar\n",
        "WAVECAR": b"accepted-wavecar\n",
        "wannier90.win": b"accepted-win\n",
    }
    for name, payload in payloads.items():
        (parent / name).write_bytes(payload)
    expected = {name: _sha256(parent / name) for name in payloads}
    fake_vasp = tmp_path / "vasp"
    fake_vasp.write_bytes(b"fake-vasp-binary\n")

    monkeypatch.setattr(oracle, "_PARENT_HASHES", expected)
    monkeypatch.setattr(oracle, "TPT_BULK_VASP_NCL_EXECUTABLE", fake_vasp)
    monkeypatch.setattr(
        oracle,
        "_copy_wavecar",
        lambda source, destination, *, mode: shutil.copy2(source, destination),
    )

    output = tmp_path / "capsule"
    seal = oracle.prepare_oracle_capsule(
        parent_root=parent,
        output_root=output,
        case=oracle.ORACLE_CASES["pure_g"],
    )
    saved = json.loads((output / "SOURCE_SEAL.json").read_text())
    assert seal == saved
    assert saved["case"]["shell_list"] == [28, 60, 2195]
    assert saved["runtime"]["selected_provenance_version"] == (
        oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION
    )
    assert saved["runtime"]["selected_provenance_scope"] == (
        oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE
    )
    assert {
        key for key in saved["runtime"] if key.startswith("selected_provenance_")
    } == {
        "selected_provenance_version",
        "selected_provenance_scope",
        "selected_provenance_hashes_at_preparation",
    }
    assert set(saved["runtime"]["selected_provenance_hashes_at_preparation"]) == set(
        oracle._RUNTIME_SOURCE_RELATIVE_PATHS
    )
    assert (output / "SOURCE_SEAL.json").read_bytes().endswith(b"\n")
    assert saved["safety"]["production_density_vertex_authority"] is False
    assert saved["safety"]["production_hf_authority"] is False
    assert "mpirun -np ${SLURM_NTASKS}" in saved["execution"]["command"]
    assert "srun" not in saved["execution"]["command"]
    assert saved["capsule_inventory"]["WAVECAR"]["sha256"] == expected["WAVECAR"]
    assert (output / "WAVECAR").stat().st_ino != (parent / "WAVECAR").stat().st_ino
    assert stat.S_IMODE((output / "WAVECAR").stat().st_mode) == 0o444

    with pytest.raises(FileExistsError, match="refusing to replace"):
        oracle.prepare_oracle_capsule(
            parent_root=parent,
            output_root=output,
            case=oracle.ORACLE_CASES["pure_g"],
        )


def _write_mmn(path: Path, blocks: list[tuple[tuple[int, int, int, int, int], np.ndarray]]) -> None:
    num_bands = blocks[0][1].shape[0]
    num_kpoints = max(header[0] for header, _ in blocks)
    nntot = len(blocks) // num_kpoints
    with path.open("w") as handle:
        handle.write("synthetic\n")
        handle.write(f"{num_bands} {num_kpoints} {nntot}\n")
        for header, matrix in blocks:
            handle.write(" ".join(str(value) for value in header) + "\n")
            for target_ket in range(num_bands):
                for source_bra in range(num_bands):
                    value = matrix[source_bra, target_ket]
                    handle.write(f"{value.real:.16e} {value.imag:.16e}\n")


def test_compare_wannier90_mmn_files_streams_exact_inventory(tmp_path: Path) -> None:
    blocks = [
        ((1, 2, 0, 0, 0), np.array([[1.0, 2.0j], [-2.0j, 0.5]], dtype=np.complex128)),
        ((2, 1, 0, 0, 0), np.array([[0.25, -1.0j], [1.0j, 0.75]], dtype=np.complex128)),
    ]
    reference = tmp_path / "reference.mmn"
    candidate = tmp_path / "candidate.mmn"
    _write_mmn(reference, blocks)
    _write_mmn(candidate, blocks)
    candidate.chmod(0o444)

    report = oracle.compare_wannier90_mmn_files(
        reference_path=reference,
        candidate_path=candidate,
        num_bands=2,
        num_kpoints=2,
        nntot=1,
        reference_sha256=_sha256(reference),
    )
    assert report["block_count"] == 2
    assert report["max_abs_difference"] == 0.0
    assert report["header_inventory_exact_match"] is True
    assert report["candidate_sha256"] == _sha256(candidate)


def test_validate_pure_g_mmn_file_checks_same_k_reverse_closure(tmp_path: Path) -> None:
    matrix_0 = np.array([[1.0, 2.0j], [-3.0j, 0.5]], dtype=np.complex128)
    matrix_1 = np.array([[0.25, -1.0j], [2.0j, 0.75]], dtype=np.complex128)
    blocks = [
        ((1, 1, 1, 0, 0), matrix_0),
        ((1, 1, -1, 0, 0), matrix_0.conj().T),
        ((2, 2, 1, 0, 0), matrix_1),
        ((2, 2, -1, 0, 0), matrix_1.conj().T),
    ]
    candidate = tmp_path / "pure-g.mmn"
    _write_mmn(candidate, blocks)
    candidate.chmod(0o444)

    report = oracle.validate_pure_g_mmn_file(
        candidate_path=candidate,
        num_bands=2,
        num_kpoints=2,
        expected_shifts=((1, 0, 0), (-1, 0, 0)),
    )
    assert report["block_count"] == 4
    assert report["same_k_inventory"] is True
    assert report["shift_counts"] == {"1,0,0": 2, "-1,0,0": 2}
    assert report["reverse_closure_max_abs_difference"] == 0.0


def test_validate_general_qg_mmn_checks_headers_and_matrix_reverse_closure(
    tmp_path: Path,
) -> None:
    forward_a = np.array([[1.0, 2.0j], [-3.0j, 0.5]], dtype=np.complex128)
    forward_b = np.array([[0.25, -1.0j], [2.0j, 0.75]], dtype=np.complex128)
    blocks = [
        ((1, 2, 0, 0, 0), forward_a),
        ((1, 2, -1, 0, 0), forward_b),
        ((2, 1, 0, 0, 0), forward_a.conj().T),
        ((2, 1, 1, 0, 0), forward_b.conj().T),
    ]
    candidate = tmp_path / "general-qg.mmn"
    _write_mmn(candidate, blocks)
    candidate.chmod(0o444)

    report = oracle.validate_general_qg_mmn_file(
        candidate_path=candidate,
        num_bands=2,
        kpoint_numerators=((0, 0, 0), (1, 0, 0)),
        expected_transfers=((-1, 0, 0), (1, 0, 0)),
        mesh=(2, 1, 1),
        check_matrix_reverse_closure=True,
    )
    assert report["block_count"] == 4
    assert report["header_inventory_exact"] is True
    assert report["header_reverse_closed"] is True
    assert report["reverse_closure_max_abs_difference"] == 0.0
    assert report["transfer_counts"] == {"-1,0,0": 2, "1,0,0": 2}


def test_submission_receipt_is_parsed_and_cross_bound(tmp_path: Path) -> None:
    path = tmp_path / "SUBMISSION-123.json"
    receipt = {
        "schema": "tpt-bulk-vasp-paw-oracle-submission-v6",
        "job_id": 123,
        "account": "hmt03",
        "nodes": 2,
        "ntasks": 32,
        "ntasks_per_node": 16,
        "exclusive": True,
        "source_seal_sha256": "a" * 64,
        "launcher": "mpirun -np 32",
        "expected_runtime": {
            "mpi_ranks": 32,
            "kpar_groups": 4,
            "ncore": 1,
            "npar": 8,
            "nbands": 600,
        },
        "production_density_vertex_authority": False,
        "production_hf_authority": False,
    }
    path.write_text(json.dumps(receipt))
    path.chmod(0o444)
    loaded, digest = oracle._load_and_validate_submission_receipt(
        path=path,
        job_id=123,
        source_seal_sha256="a" * 64,
        expected_case="native",
    )
    assert loaded == receipt
    assert digest == _sha256(path)

    receipt["source_seal_sha256"] = "b" * 64
    path.chmod(0o644)
    path.write_text(json.dumps(receipt))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="source_seal_sha256 mismatch"):
        oracle._load_and_validate_submission_receipt(
            path=path,
            job_id=123,
            source_seal_sha256="a" * 64,
            expected_case="native",
        )


def test_submission_receipt_rejects_writable_or_missing_mixed_denials(
    tmp_path: Path,
) -> None:
    path = tmp_path / "SUBMISSION-456.json"
    receipt = {
        "schema": "tpt-bulk-vasp-paw-oracle-submission-v6",
        "job_id": 456,
        "account": "hmt03",
        "nodes": 2,
        "ntasks": 32,
        "ntasks_per_node": 16,
        "exclusive": True,
        "source_seal_sha256": "c" * 64,
        "launcher": "mpirun -np 32",
        "expected_runtime": {
            "mpi_ranks": 32,
            "kpar_groups": 4,
            "ncore": 1,
            "npar": 8,
            "nbands": 600,
        },
        "oracle_case": {
            "name": "mixed_qg",
            "kmesh_tolerance": "1.0e-7",
            "search_shells": 6529,
            "shell_list": [1, 2, 6529],
            "expected_physical_q_numerators": [
                list(value) for value in oracle._mixed_qg_expected_transfers()
            ],
        },
        "production_density_vertex_authority": False,
        "production_hf_authority": False,
        "arbitrary_finite_q_plus_g_exporter_authority": False,
        "screening_authority": False,
        "filling_reference_authority": False,
    }
    path.write_text(json.dumps(receipt))
    with pytest.raises(PermissionError, match="read-only"):
        oracle._load_and_validate_submission_receipt(
            path=path,
            job_id=456,
            source_seal_sha256="c" * 64,
            expected_case="mixed_qg",
        )

    receipt.pop("screening_authority")
    path.write_text(json.dumps(receipt))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="explicitly deny screening_authority"):
        oracle._load_and_validate_submission_receipt(
            path=path,
            job_id=456,
            source_seal_sha256="c" * 64,
            expected_case="mixed_qg",
        )


def test_mixed_validation_receipt_and_exact_fft_k_ordering(tmp_path: Path) -> None:
    science_job_id = 456
    validation_job_id = 457
    receipt = {
        "schema": "tpt-bulk-mixed-qg-validation-submission-v2",
        "validation_job_id": validation_job_id,
        "science_job_id": science_job_id,
        "dependency": f"afterok:{science_job_id}",
        "account": "hmt03",
        "partitions": ["regular256", "regular6430"],
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 64,
        "exclusive": True,
        "explicit_time_limit": None,
        "action": "validate-mixed-qg",
        "source_seal_sha256": "a" * 64,
        "science_submission_sha256": "b" * 64,
        "validator_source_sha256": "c" * 64,
        "zero_science_postflight": True,
        "production_density_vertex_authority": False,
        "production_hf_authority": False,
    }
    receipt_path = tmp_path / f"VALIDATION-SUBMISSION-{validation_job_id}.json"
    receipt_path.write_text(json.dumps(receipt))
    receipt_path.chmod(0o444)
    loaded, digest = oracle._load_mixed_qg_validation_submission_receipt(
        output_root=tmp_path,
        validation_job_id=validation_job_id,
        science_job_id=science_job_id,
        source_seal_sha256="a" * 64,
        science_submission_sha256="b" * 64,
        validator_source_sha256="c" * 64,
    )
    assert loaded == receipt
    assert digest == _sha256(receipt_path)

    axes = (
        (0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2, -1),
        (0, 1, 2, -1),
        (0, 1, 2, -1),
    )
    points = [(x, y, z) for z in axes[2] for y in axes[1] for x in axes[0]]

    def write_win(path: Path, ordered: list[tuple[int, int, int]]) -> None:
        rows = ["begin kpoints"]
        rows.extend(
            f"{x / 12:.12f} {y / 4:.12f} {z / 4:.12f}"
            for x, y, z in ordered
        )
        rows.append("end kpoints")
        path.write_text("\n".join(rows) + "\n")

    win_path = tmp_path / "wannier90.win"
    write_win(win_path, points)
    assert oracle._read_win_kpoint_numerators(win_path) == tuple(points)
    points[0], points[1] = points[1], points[0]
    write_win(win_path, points)
    with pytest.raises(ValueError, match="FFT ordering"):
        oracle._read_win_kpoint_numerators(win_path)


def test_prepare_oracle_capsule_rejects_parent_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    for name in oracle._PARENT_HASHES:
        (parent / name).write_text(_PARENT_INCAR if name == "INCAR" else name)
    expected = {name: _sha256(parent / name) for name in oracle._PARENT_HASHES}
    expected["KPOINTS"] = "0" * 64
    fake_vasp = tmp_path / "vasp"
    fake_vasp.write_text("fake")
    monkeypatch.setattr(oracle, "_PARENT_HASHES", expected)
    monkeypatch.setattr(oracle, "TPT_BULK_VASP_NCL_EXECUTABLE", fake_vasp)

    with pytest.raises(ValueError, match="KPOINTS hash mismatch"):
        oracle.prepare_oracle_capsule(
            parent_root=parent,
            output_root=tmp_path / "capsule",
            case=oracle.ORACLE_CASES["native"],
        )
    assert not (tmp_path / "capsule").exists()


def _current_oracle_selected_hashes() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[1]
    return {
        relative: _sha256(repository_root / relative)
        for relative in oracle._RUNTIME_SOURCE_RELATIVE_PATHS
    }


def test_selected_provenance_v2_inventory_constants_and_exact_paths() -> None:
    expected_oracle_paths = {
        "scripts/mean_field_tools.py",
        "scripts/submit_mean_field.sbatch",
        "src/mean_field/runtime.py",
        "src/mean_field/core/io/__init__.py",
        "src/mean_field/core/io/artifacts.py",
        "src/mean_field/devtools/tpt_bulk_paw_oracle.py",
        "src/mean_field/systems/tpt_bulk/source.py",
        "src/mean_field/systems/tpt_bulk/wavefunctions.py",
    }
    assert oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION == (
        "paw_oracle_selected_provenance_v2"
    )
    assert oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE == (
        "paw_oracle_v2_selected_set_not_complete_python_import_closure"
    )
    assert set(oracle._RUNTIME_SOURCE_RELATIVE_PATHS) == expected_oracle_paths
    assert "src/mean_field/devtools/_runtime.py" not in expected_oracle_paths

    expected_qualifier_paths = {
        "scripts/mean_field_tools.py",
        "scripts/submit_mean_field.sbatch",
        "src/mean_field/runtime.py",
        "src/mean_field/core/io/__init__.py",
        "src/mean_field/core/io/artifacts.py",
        "src/mean_field/devtools/qualify_tpt_bulk_parent_wavefunctions.py",
        "src/mean_field/systems/tpt_bulk/__init__.py",
        "src/mean_field/systems/tpt_bulk/microscopic_hf.py",
        "src/mean_field/systems/tpt_bulk/source.py",
        "src/mean_field/systems/tpt_bulk/wavefunctions.py",
    }
    assert qualifier.PARENT_QUALIFICATION_SELECTED_PROVENANCE_VERSION == (
        "parent_qualification_selected_provenance_v2"
    )
    assert qualifier.PARENT_QUALIFICATION_SELECTED_PROVENANCE_SCOPE == (
        "parent_qualification_v2_selected_set_not_complete_python_import_closure"
    )
    assert set(qualifier._PARENT_QUALIFICATION_SELECTED_PROVENANCE_PATHS) == (
        expected_qualifier_paths
    )


def test_selected_provenance_rejects_legacy_unknown_types_and_bad_sha() -> None:
    current = _current_oracle_selected_hashes()
    key = "selected_provenance_hashes_at_preparation"
    valid = {
        "selected_provenance_version": oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION,
        "selected_provenance_scope": oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE,
        key: current,
    }
    invalid_cases = (
        ({key: current}, "missing selected_provenance_version"),
        (
            {
                **valid,
                "selected_provenance_version": (
                    "paw_oracle_selected_provenance_v1"
                ),
            },
            "unknown",
        ),
        ({**valid, "selected_provenance_version": 2}, "must be a string"),
        ([], "container must be a dictionary"),
        ({**valid, key: []}, "must be a dictionary"),
        ({**valid, key: {**current, next(iter(current)): "BAD"}}, "64-hex"),
    )
    for owner, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            oracle._selected_provenance_inventory(
                owner,
                domain="synthetic oracle seal",
            )


def test_selected_provenance_reports_complete_missing_extra_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _current_oracle_selected_hashes()
    removed = sorted(current)[:2]
    malformed = {key: value for key, value in current.items() if key not in removed}
    malformed.update({"extra/a.py": "a" * 64, "extra/b.py": "b" * 64})
    called = False

    def forbidden_authorization(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("keyset drift entered external authorization")

    monkeypatch.setattr(
        oracle,
        "ensure_not_running_compute_on_login_node",
        lambda _workload: None,
    )
    monkeypatch.setattr(
        oracle,
        "_load_external_revalidation_authorization",
        forbidden_authorization,
    )
    monkeypatch.setattr(
        oracle,
        "_load_and_validate_submission_receipt",
        lambda **_kwargs: ({}, "b" * 64),
    )
    output_root = tmp_path / "oracle"
    output_root.mkdir()
    (output_root / "SOURCE_SEAL.json").write_text(
        json.dumps(
            {
                "schema": "tpt-bulk-vasp-paw-oracle-capsule-v1",
                "case": {"name": "native"},
                "scope": {"num_bands": 600, "num_kpoints": 192},
                "runtime": {
                    "selected_provenance_version": (
                        oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION
                    ),
                    "selected_provenance_scope": (
                        oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE
                    ),
                    "selected_provenance_hashes_at_preparation": malformed,
                },
            }
        )
    )
    with pytest.raises(ValueError) as error:
        oracle.validate_native_oracle_outputs(
            parent_root=tmp_path / "parent",
            output_root=output_root,
            job_id=123,
            revalidation_seal_path=tmp_path / "authorization.json",
        )
    message = str(error.value)
    assert f"missing={removed}" in message
    assert "extra=['extra/a.py', 'extra/b.py']" in message
    assert called is False


def test_selected_provenance_same_keyset_content_drift_reaches_existing_gate_shape() -> None:
    current = _current_oracle_selected_hashes()
    changed_path = sorted(current)[0]
    preparation = dict(current)
    preparation[changed_path] = "0" * 64
    prepared, observed, drift = oracle._selected_provenance_inventory(
        {
            "selected_provenance_version": oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION,
            "selected_provenance_scope": oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE,
            "selected_provenance_hashes_at_preparation": preparation,
        },
        domain="synthetic oracle seal",
    )
    assert set(prepared) == set(observed) == set(oracle._RUNTIME_SOURCE_RELATIVE_PATHS)
    assert drift == {
        changed_path: {
            "at_preparation": "0" * 64,
            "at_validation": current[changed_path],
        }
    }



def test_selected_provenance_current_inventory_missing_is_domain_value_error(
    tmp_path: Path,
) -> None:
    current = _current_oracle_selected_hashes()
    with pytest.raises(ValueError) as error:
        oracle._selected_provenance_inventory(
            {
                "selected_provenance_version": (
                    oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION
                ),
                "selected_provenance_scope": (
                    oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE
                ),
                "selected_provenance_hashes_at_preparation": current,
            },
            domain="synthetic missing-current seal",
            repository_root=tmp_path,
        )
    assert "synthetic missing-current seal" in str(error.value)
    assert f"missing={sorted(current)}" in str(error.value)
    assert "extra=[]" in str(error.value)


def test_active_handoff_validates_full_selected_inventory_before_wavecar_hash(
    tmp_path: Path,
) -> None:
    active_root = tmp_path / "active"
    active_root.mkdir()
    current = _current_oracle_selected_hashes()
    missing_path = sorted(current)[0]
    malformed = {key: value for key, value in current.items() if key != missing_path}
    (active_root / "ACTIVE_WAVECAR_SEAL.json").write_text(
        json.dumps(
            {
                "schema": "tpt-bulk-active-spinor-wavecar-v1",
                "status": "prepared_not_yet_paw_oracle_qualified",
                "scope": {
                    "active_ranks_1based": [233, 248],
                    "mesh": [12, 4, 4],
                },
                "selected_provenance_version": (
                    oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION
                ),
                "selected_provenance_scope": (
                    oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE
                ),
                "selected_provenance_hashes_at_preparation": malformed,
                "output": {"sha256": "a" * 64},
            }
        )
    )
    with pytest.raises(ValueError) as error:
        oracle.prepare_active_oracle_capsule(
            parent_root=tmp_path / "parent",
            active_wavecar_root=active_root,
            output_root=tmp_path / "capsule",
            case=oracle.ORACLE_CASES["native"],
        )
    assert f"missing=['{missing_path}']; extra=[]" in str(error.value)
    assert not (tmp_path / "capsule").exists()



def test_qualify_parent_tiny_backend_writes_strict_v5_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_numpy = np

    class TinyNumpyProxy:
        def __getattr__(self, name: str) -> object:
            return getattr(original_numpy, name)

        @staticmethod
        def eye(_size: int) -> np.ndarray:
            return original_numpy.eye(1)

    source = SimpleNamespace(
        lattice_angstrom=np.eye(1),
        hr_sha256="1" * 64,
    )
    eigensystem = SimpleNamespace(
        k_fractional=np.zeros((1, 1)),
        active_energies_ev=np.zeros((1, 1)),
        active_eigenvectors=np.ones((1, 1, 1), dtype=np.complex128),
    )
    checkpoint_metadata = SimpleNamespace(
        mp_grid=qualifier.TPT_BULK_SOURCE_MESH,
        num_kpts=192,
        num_bands=600,
        num_wann=320,
        kpoints_fractional=np.zeros((1, 1)),
        lwindow=np.ones((1, 1), dtype=bool),
        nntot=1,
        ndimwin=np.array([1]),
        sha256="2" * 64,
    )
    wavecar_metadata = SimpleNamespace(
        num_kpts=192,
        num_bands=600,
        kpoints_fractional=np.zeros((1, 1)),
        lattice_angstrom=np.eye(1),
        eigenvalues_ev=np.zeros((1, 1)),
        occupations=np.ones((1, 1)),
        encut_ev=1.0,
        plane_wave_coefficients_per_k=np.array([1]),
        g_vectors_per_k=np.array([1]),
    )

    class TinyCheckpoint:
        metadata = checkpoint_metadata

        def __enter__(self) -> "TinyCheckpoint":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def dft_to_wannier_gauge(_ik: int) -> np.ndarray:
            return np.ones((1, 1), dtype=np.complex128)

    class TinyWavecar:
        metadata = wavecar_metadata
        sha256 = "3" * 64

        def __enter__(self) -> "TinyWavecar":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read_spinor_coefficients(
            _ik: int,
            _outer: np.ndarray,
        ) -> np.ndarray:
            return np.ones((1, 1), dtype=np.complex128)

        @staticmethod
        def g_vectors(_ik: int) -> np.ndarray:
            return np.zeros((1, 3), dtype=np.int64)

    block = SimpleNamespace(
        source_k_0based=0,
        target_k_0based=0,
        overlap_source_bra_target_ket=np.ones((1, 1), dtype=np.complex128),
        target_cell_shift_integer=(0, 0, 0),
    )
    monkeypatch.setattr(qualifier, "np", TinyNumpyProxy())
    monkeypatch.setattr(qualifier, "range", lambda _stop: range(1), raising=False)
    monkeypatch.setattr(qualifier, "load_tpt_bulk_source", lambda _root: source)
    monkeypatch.setattr(
        qualifier,
        "build_tpt_bulk_active_eigensystem",
        lambda *_args, **_kwargs: eigensystem,
    )
    monkeypatch.setattr(
        qualifier,
        "Wannier90CheckpointReader",
        lambda *_args, **_kwargs: TinyCheckpoint(),
    )
    monkeypatch.setattr(
        qualifier,
        "WavecarSpinorReader",
        lambda *_args, **_kwargs: TinyWavecar(),
    )
    monkeypatch.setattr(
        qualifier,
        "read_wannier90_eig",
        lambda *_args, **_kwargs: np.zeros((1, 1)),
    )
    monkeypatch.setattr(
        qualifier,
        "project_spinor_plane_wave_coefficients",
        lambda *_args, **_kwargs: np.ones((1, 1), dtype=np.complex128),
    )
    monkeypatch.setattr(
        qualifier,
        "contract_spinor_plane_wave_density_vertex",
        lambda **_kwargs: np.ones((1, 1), dtype=np.complex128),
    )
    monkeypatch.setattr(
        qualifier,
        "iter_wannier90_mmn_blocks",
        lambda *_args, **_kwargs: iter((block,)),
    )

    report = qualifier.qualify_parent(
        source_root=tmp_path / "source",
        parent_root=tmp_path / "parent",
        active_ranks_1based=(1, 1),
        anchor_k_indices=(0,),
        mmn_source_k_indices=(0,),
    )
    expected_hashes = {
        relative: _sha256(Path(__file__).resolve().parents[1] / relative)
        for relative in qualifier._PARENT_QUALIFICATION_SELECTED_PROVENANCE_PATHS
    }
    provenance_keys = {
        "selected_provenance_version",
        "selected_provenance_scope",
        "selected_provenance_hashes_at_qualification_start",
    }
    assert report["schema"] == "tpt-bulk-parent-wavefunction-bridge-qualification-v5"
    assert {key for key in report if key.startswith("selected_provenance_")} == (
        provenance_keys
    )
    assert report["selected_provenance_version"] == (
        qualifier.PARENT_QUALIFICATION_SELECTED_PROVENANCE_VERSION
    )
    assert report["selected_provenance_scope"] == (
        qualifier.PARENT_QUALIFICATION_SELECTED_PROVENANCE_SCOPE
    )
    assert report["selected_provenance_hashes_at_qualification_start"] == (
        expected_hashes
    )
    assert _selected_provenance_key_paths(report) == {
        key: [(key,)] for key in provenance_keys
    }

    report_path = tmp_path / "QUALIFICATION.json"
    qualifier._atomic_write_new_json(report_path, report)
    _assert_strict_json_artifact(report_path, report, readonly=False)

def test_prepare_active_wavecar_tiny_backend_writes_readonly_v2_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_numpy = np

    class TinyNumpyProxy:
        def __getattr__(self, name: str) -> object:
            return getattr(original_numpy, name)

        @staticmethod
        def empty(
            shape: tuple[int, ...],
            dtype: object = float,
        ) -> np.ndarray:
            if shape == (192, 600, 1):
                return original_numpy.empty((1, 1, 1), dtype=dtype)
            return original_numpy.empty(shape, dtype=dtype)

    checkpoint_metadata = SimpleNamespace(
        num_kpts=192,
        num_bands=600,
        num_wann=320,
        mp_grid=oracle.TPT_BULK_SOURCE_MESH,
        kpoints_fractional=np.zeros((1, 1)),
    )

    class TinyCheckpoint:
        metadata = checkpoint_metadata

        def __enter__(self) -> "TinyCheckpoint":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def dft_to_wannier_gauge(_ik: int) -> np.ndarray:
            return np.ones((1, 1), dtype=np.complex128)

    source = SimpleNamespace(hr_sha256="4" * 64)
    eigensystem = SimpleNamespace(
        nk=1,
        k_fractional=np.zeros((1, 1)),
        active_energies_ev=np.zeros((1, 1)),
        active_eigenvectors=np.ones((1, 1, 1), dtype=np.complex128),
        max_antihermitian_before_symmetrization_ev=0.0,
        max_eigenpair_residual_ev=0.0,
    )

    def tiny_writer(
        _source: Path,
        destination: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        destination.write_bytes(b"tiny-active-wavecar\n")
        return {
            "coefficient_dtype": "complex128",
            "backend": "tiny_monkeypatch",
        }

    monkeypatch.setattr(oracle, "np", TinyNumpyProxy())
    monkeypatch.setattr(oracle, "range", lambda _stop: range(1), raising=False)
    monkeypatch.setattr(oracle, "ensure_not_running_compute_on_login_node", lambda _: None)
    monkeypatch.setattr(oracle, "load_tpt_bulk_source", lambda _root: source)
    monkeypatch.setattr(
        oracle,
        "build_tpt_bulk_active_eigensystem",
        lambda *_args, **_kwargs: eigensystem,
    )
    monkeypatch.setattr(
        oracle,
        "Wannier90CheckpointReader",
        lambda *_args, **_kwargs: TinyCheckpoint(),
    )
    monkeypatch.setattr(oracle, "write_projected_spinor_wavecar", tiny_writer)

    output_root = tmp_path / "active-wavecar"
    seal = oracle.prepare_active_wavecar(
        source_root=tmp_path / "source",
        parent_root=tmp_path / "parent",
        output_root=output_root,
        active_ranks_1based=(1, 1),
        serialized_occupied_rank=1,
    )
    expected_hashes = _current_oracle_selected_hashes()
    provenance_keys = {
        "selected_provenance_version",
        "selected_provenance_scope",
        "selected_provenance_hashes_at_preparation",
    }
    assert {key for key in seal if key.startswith("selected_provenance_")} == (
        provenance_keys
    )
    assert seal["selected_provenance_version"] == (
        oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION
    )
    assert seal["selected_provenance_scope"] == (
        oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE
    )
    assert seal["selected_provenance_hashes_at_preparation"] == expected_hashes
    assert _selected_provenance_key_paths(seal) == {
        key: [(key,)] for key in provenance_keys
    }
    seal_path = output_root / "ACTIVE_WAVECAR_SEAL.json"
    _assert_strict_json_artifact(seal_path, seal, readonly=True)

def _valid_selected_provenance(
    hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "selected_provenance_version": oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION,
        "selected_provenance_scope": oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE,
        "selected_provenance_hashes_at_preparation": (
            _current_oracle_selected_hashes() if hashes is None else hashes
        ),
    }


def test_selected_provenance_field_builders_have_exact_schema_keys(
    tmp_path: Path,
) -> None:
    current = _current_oracle_selected_hashes()
    assert oracle._selected_provenance_preparation_fields(current) == (
        _valid_selected_provenance(current)
    )
    report_fields = oracle._selected_provenance_report_fields(current, current, {})
    assert set(report_fields) == {
        "selected_provenance_version",
        "selected_provenance_scope",
        "selected_provenance_hashes_at_preparation",
        "selected_provenance_hashes_at_validation",
        "selected_provenance_drift",
    }
    qualifier_fields = qualifier._qualification_provenance_fields(current)
    assert set(qualifier_fields) == {
        "selected_provenance_version",
        "selected_provenance_scope",
        "selected_provenance_hashes_at_qualification_start",
    }
    assert qualifier_fields["selected_provenance_version"] == (
        "parent_qualification_selected_provenance_v2"
    )
    qualifier_identity = qualifier._qualification_report_identity(current)
    assert set(qualifier_identity) == {
        "schema",
        "status",
        *qualifier_fields,
    }
    assert qualifier_identity["schema"] == (
        "tpt-bulk-parent-wavefunction-bridge-qualification-v5"
    )
    report_path = tmp_path / "QUALIFICATION.json"
    qualifier._atomic_write_new_json(report_path, qualifier_identity)
    assert report_path.read_bytes().endswith(b"\n")
    oracle_path = tmp_path / "PAW_REPORT.json"
    oracle._write_json_new(oracle_path, report_fields)
    assert oracle_path.read_bytes().endswith(b"\n")


def test_selected_provenance_rejects_missing_unknown_scope_uppercase_and_legacy() -> None:
    current = _current_oracle_selected_hashes()
    valid = _valid_selected_provenance(current)
    hashes_key = "selected_provenance_hashes_at_preparation"
    first = next(iter(current))
    invalid_cases = (
        (
            {key: value for key, value in valid.items() if key != "selected_provenance_scope"},
            "missing selected_provenance_scope",
        ),
        (
            {**valid, "selected_provenance_scope": "unknown"},
            "unknown selected_provenance_scope",
        ),
        (
            {**valid, hashes_key: {**current, first: "A" * 64}},
            "lowercase 64-hex",
        ),
        (
            {
                "selected_provenance_version": oracle.PAW_ORACLE_SELECTED_PROVENANCE_VERSION,
                "selected_provenance_scope": oracle.PAW_ORACLE_SELECTED_PROVENANCE_SCOPE,
                "runtime_source_hashes_at_preparation": current,
            },
            "legacy selected-provenance fields",
        ),
    )
    for owner, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            oracle._selected_provenance_inventory(
                owner,
                domain="synthetic oracle seal",
            )


def test_strict_json_reader_rejects_duplicates_constants_and_nonobjects(
    tmp_path: Path,
) -> None:
    first = oracle._RUNTIME_SOURCE_RELATIVE_PATHS[0]
    a_hash = "a" * 64
    b_hash = "b" * 64
    malformed = {
        "duplicate version": (
            '{"selected_provenance_version":"a",'
            '"selected_provenance_version":"b"}'
        ),
        "duplicate inventory key": (
            '{"selected_provenance_hashes_at_preparation":{'
            + json.dumps(first)
            + ":"
            + json.dumps(a_hash)
            + ","
            + json.dumps(first)
            + ":"
            + json.dumps(b_hash)
            + "}}"
        ),
        "NaN": '{"value":NaN}',
        "Infinity": '{"value":Infinity}',
        "top-level list": "[]",
    }
    for name, text in malformed.items():
        path = tmp_path / f"{name.replace(' ', '-')}.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            oracle._read_json_object(path, domain=name)


def test_version_and_scope_fail_before_authorization_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _current_oracle_selected_hashes()
    called = False

    def forbidden_authorization(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid provenance reached authorization")

    monkeypatch.setattr(
        oracle,
        "_load_external_revalidation_authorization",
        forbidden_authorization,
    )
    invalid = (
        {
            **_valid_selected_provenance(current),
            "selected_provenance_version": "paw_oracle_selected_provenance_v1",
        },
        {
            **_valid_selected_provenance(current),
            "selected_provenance_scope": "unknown",
        },
    )
    for owner in invalid:
        with pytest.raises(ValueError):
            oracle._validate_postflight_provenance(
                owner=owner,
                domain="synthetic postflight seal",
                output_root=tmp_path,
                job_id=123,
                source_seal_sha256="a" * 64,
                submission_receipt_sha256="b" * 64,
                revalidation_seal_path=tmp_path / "authorization.json",
            )
    assert called is False


def _minimal_postflight_seal(kind: str, hashes: dict[str, str]) -> dict[str, object]:
    runtime = _valid_selected_provenance(hashes)
    if kind == "active":
        layout = oracle._ACTIVE_ORACLE_LAYOUTS["regular64"]
        layout_keys = (
            "allocated_cpus",
            "mpi_ranks",
            "ranks_per_node",
            "kpar",
            "ncore",
            "npar",
        )
        return {
            "schema": "tpt-bulk-active16-vasp-paw-oracle-capsule-v1",
            "status": "prepared_not_submitted",
            "case": {"name": "native", "shell_list": [1, 2, 4]},
            "scope": {"num_bands": 16, "num_kpoints": 192},
            "execution": {
                "layout_profile": "regular64",
                "submission_layout": {
                    "nodes": 1,
                    **{key: layout[key] for key in layout_keys},
                    "nbands": 16,
                },
            },
            "runtime": runtime,
        }
    case = {
        "native": {"name": "native", "shell_list": [1, 2, 4]},
        "pure_g": {"name": "pure_g", "shell_list": [28, 60, 2195]},
        "mixed_qg": {"name": "mixed_qg", "shell_list": [1, 2, 6529]},
    }[kind]
    return {
        "schema": "tpt-bulk-vasp-paw-oracle-capsule-v1",
        "case": case,
        "scope": {"num_bands": 600, "num_kpoints": 192},
        "runtime": runtime,
    }


def _call_postflight_validator(
    kind: str,
    *,
    tmp_path: Path,
    supply_revalidation_path: bool = True,
) -> dict[str, object]:
    output_root = tmp_path / kind
    revalidation_seal_path = (
        tmp_path / "authorization.json" if supply_revalidation_path else None
    )
    if kind == "active":
        return oracle.validate_active_oracle_outputs(
            source_root=tmp_path / "source",
            parent_root=tmp_path / "parent",
            reference_root=tmp_path / "reference",
            output_root=output_root,
            job_id=123,
            revalidation_seal_path=revalidation_seal_path,
        )
    if kind == "mixed_qg":
        return oracle.validate_mixed_qg_oracle_outputs(
            parent_root=tmp_path / "parent",
            output_root=output_root,
            job_id=123,
            validation_job_id=124,
            revalidation_seal_path=revalidation_seal_path,
        )
    validator = {
        "native": oracle.validate_native_oracle_outputs,
        "pure_g": oracle.validate_pure_g_oracle_outputs,
    }[kind]
    return validator(
        parent_root=tmp_path / "parent",
        output_root=output_root,
        job_id=123,
        revalidation_seal_path=revalidation_seal_path,
    )


_POSTFLIGHT_KINDS = ("active", "native", "pure_g", "mixed_qg")


def _patch_postflight_entry_readers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    authorization_loader: object | None = None,
) -> None:
    """Replace only scheduler/authorization boundary readers used before science."""

    monkeypatch.setattr(oracle, "ensure_not_running_compute_on_login_node", lambda _: None)
    monkeypatch.setattr(
        oracle,
        "_load_active_submission_receipt",
        lambda **_kwargs: ({}, "b" * 64),
    )
    monkeypatch.setattr(
        oracle,
        "_load_and_validate_submission_receipt",
        lambda **_kwargs: ({}, "b" * 64),
    )
    if authorization_loader is not None:
        monkeypatch.setattr(
            oracle,
            "_load_external_revalidation_authorization",
            authorization_loader,
        )


def _write_minimal_postflight_seal(
    root: Path,
    kind: str,
    hashes: dict[str, str],
) -> Path:
    output_root = root / kind
    output_root.mkdir(parents=True)
    seal = _minimal_postflight_seal(kind, hashes)
    if kind == "active":
        seal["capsule_inventory"] = {}
        seal["active_wavecar"] = {"seal_sha256": "d" * 64}
    else:
        seal["parent_inventory"] = {}
        seal["capsule_inventory"] = {}
    seal_path = output_root / "SOURCE_SEAL.json"
    seal_path.write_text(json.dumps(seal) + "\n", encoding="utf-8")
    return seal_path


@pytest.mark.parametrize("kind", _POSTFLIGHT_KINDS)
@pytest.mark.parametrize(
    "raw_json",
    (
        '{"schema":"first","schema":"second"}\n',
        '{"value":NaN}\n',
        '{"value":Infinity}\n',
        '[]\n',
    ),
    ids=("duplicate-key", "nan", "infinity", "non-object"),
)
def test_actual_postflights_reject_malformed_source_seal_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    raw_json: str,
) -> None:
    authorization_calls: list[dict[str, object]] = []

    def forbidden_authorization(**kwargs):
        authorization_calls.append(kwargs)
        raise AssertionError("malformed SOURCE_SEAL reached authorization")

    _patch_postflight_entry_readers(
        monkeypatch,
        authorization_loader=forbidden_authorization,
    )
    output_root = tmp_path / kind
    output_root.mkdir()
    (output_root / "SOURCE_SEAL.json").write_text(raw_json, encoding="utf-8")

    with pytest.raises(ValueError):
        _call_postflight_validator(kind, tmp_path=tmp_path)
    assert authorization_calls == []


@pytest.mark.parametrize("kind", _POSTFLIGHT_KINDS)
@pytest.mark.parametrize(
    "mutation",
    ("missing-scope", "unknown-scope", "uppercase-sha256"),
)
def test_actual_postflights_reject_invalid_selected_provenance_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    mutation: str,
) -> None:
    current = _current_oracle_selected_hashes()
    seal = _minimal_postflight_seal(kind, current)
    runtime = seal["runtime"]
    assert isinstance(runtime, dict)
    if mutation == "missing-scope":
        runtime.pop("selected_provenance_scope")
    elif mutation == "unknown-scope":
        runtime["selected_provenance_scope"] = "unknown"
    else:
        hashes = runtime["selected_provenance_hashes_at_preparation"]
        assert isinstance(hashes, dict)
        hashes[sorted(hashes)[0]] = "A" * 64

    authorization_calls: list[dict[str, object]] = []

    def forbidden_authorization(**kwargs):
        authorization_calls.append(kwargs)
        raise AssertionError("invalid selected provenance reached authorization")

    _patch_postflight_entry_readers(
        monkeypatch,
        authorization_loader=forbidden_authorization,
    )
    output_root = tmp_path / kind
    output_root.mkdir()
    (output_root / "SOURCE_SEAL.json").write_text(
        json.dumps(seal) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        _call_postflight_validator(kind, tmp_path=tmp_path)
    assert authorization_calls == []


@pytest.mark.parametrize("kind", _POSTFLIGHT_KINDS)
def test_actual_postflight_authorization_matrix_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    current = _current_oracle_selected_hashes()
    _patch_postflight_entry_readers(monkeypatch)

    drift_root = tmp_path / "drift-without-authorization"
    changed = dict(current)
    changed[sorted(changed)[0]] = "0" * 64
    _write_minimal_postflight_seal(drift_root, kind, changed)
    with pytest.raises(ValueError, match="requires external artifact revalidation"):
        _call_postflight_validator(
            kind,
            tmp_path=drift_root,
            supply_revalidation_path=False,
        )

    unexpected_root = tmp_path / "no-drift-with-authorization"
    _write_minimal_postflight_seal(unexpected_root, kind, current)
    unexpected_authorization = unexpected_root / "authorization.json"
    unexpected_authorization.write_text("{}\n", encoding="utf-8")
    unexpected_authorization.chmod(0o444)
    with pytest.raises(ValueError, match="authorization"):
        _call_postflight_validator(kind, tmp_path=unexpected_root)

    malformed_root = tmp_path / "drift-with-malformed-authorization"
    _write_minimal_postflight_seal(malformed_root, kind, changed)
    malformed_authorization = malformed_root / "authorization.json"
    malformed_authorization.write_text(
        '{"schema":"first","schema":"second"}\n',
        encoding="utf-8",
    )
    malformed_authorization.chmod(0o444)
    with pytest.raises(ValueError):
        _call_postflight_validator(kind, tmp_path=malformed_root)


def _install_tiny_postflight_science_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install tiny deterministic science/Slurm readers, leaving closure logic real."""

    _patch_postflight_entry_readers(monkeypatch)
    monkeypatch.setattr(
        oracle,
        "_load_mixed_qg_validation_submission_receipt",
        lambda **_kwargs: ({}, "e" * 64),
    )
    monkeypatch.setattr(
        oracle,
        "_query_slurm_accounting",
        lambda _job_id: {
            "state": "COMPLETED",
            "exit_code": "0:0",
            "nodes": "1",
            "allocated_cpus": "64",
            "raw_rows": [],
        },
    )
    monkeypatch.setattr(oracle, "_validate_parent", lambda _root: {})
    monkeypatch.setattr(
        oracle,
        "load_tpt_bulk_source",
        lambda root: SimpleNamespace(root=Path(root)),
    )
    monkeypatch.setattr(
        oracle,
        "build_tpt_bulk_active_eigensystem",
        lambda *_args, **_kwargs: SimpleNamespace(
            active_energies_ev=np.zeros((16, 192), dtype=float)
        ),
    )
    monkeypatch.setattr(
        oracle,
        "read_wannier90_eig",
        lambda _path, *, num_bands, num_kpts, **_kwargs: np.zeros(
            (num_kpts, num_bands), dtype=float
        ),
    )

    def tiny_mmn_result(**kwargs):
        candidate_path = Path(kwargs["candidate_path"])
        return {
            "passed": True,
            "max_abs_difference": 0.0,
            "reverse_closure_max_abs_difference": 0.0,
            "candidate_sha256": _sha256(candidate_path),
        }

    monkeypatch.setattr(oracle, "compare_wannier90_mmn_files", tiny_mmn_result)
    monkeypatch.setattr(oracle, "compare_projected_active_mmn_files", tiny_mmn_result)
    monkeypatch.setattr(oracle, "validate_pure_g_mmn_file", tiny_mmn_result)
    monkeypatch.setattr(oracle, "validate_general_qg_mmn_file", tiny_mmn_result)
    monkeypatch.setattr(
        oracle,
        "_read_win_kpoint_numerators",
        lambda _path: [(0, 0, 0)],
    )

    preflight = {
        "schema": "tpt-bulk-w90-general-transfer-b1-preflight-v1",
        "status": "passed_b1_complete_shell_1_2_6529",
        "shell_list": [1, 2, 6529],
        "search_shells": 6529,
        "nntot": 12,
        "b1_complete": True,
        "skip_b1_tests": False,
        "source_kpoint_one_rows": [
            {"physical_q_numerator": list(transfer)}
            for transfer in oracle._mixed_qg_expected_transfers()
        ],
        "authority": {
            "shell_index_mapping": True,
            "paw_overlap_authority": False,
            "production_hf_authority": False,
        },
        "hashes": {"wannier90.nnkp": "f" * 64},
    }
    preflight_path = tmp_path / "tiny-mixed-qg-preflight.json"
    preflight_path.write_text(json.dumps(preflight) + "\n", encoding="utf-8")
    preflight_path.chmod(0o444)
    monkeypatch.setattr(oracle, "TPT_BULK_MIXED_QG_PREFLIGHT_PATH", preflight_path)
    monkeypatch.setattr(
        oracle,
        "TPT_BULK_MIXED_QG_PREFLIGHT_SHA256",
        _sha256(preflight_path),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")


@pytest.mark.parametrize("kind", _POSTFLIGHT_KINDS)
def test_actual_postflights_publish_strict_final_reports_through_real_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    current = _current_oracle_selected_hashes()
    _write_minimal_postflight_seal(tmp_path, kind, current)
    output_root = tmp_path / kind
    required_outputs = (
        "OUTCAR",
        "wannier90.amn",
        "wannier90.eig",
        "wannier90.mmn",
        "wannier90.win",
        "wannier90.wout",
    )
    layout = oracle._ACTIVE_ORACLE_LAYOUTS["regular64"]
    outcar = "\n".join(
        (
            f"running {layout['mpi_ranks']} mpi-ranks, on 1 nodes",
            f"distrk: each k-point on {layout['npar']} cores, {layout['kpar']} groups",
            f"distr: one band on NCORE= {layout['ncore']} cores, {layout['npar']} groups",
            "running   32 mpi-ranks, on    2 nodes",
            "distrk:  each k-point on    8 cores,    4 groups",
            "distr:  one band on NCORE=   1 cores,    8 groups",
            "Computing MMN (overlap matrix elements)",
        )
    ) + "\n"
    for name in required_outputs:
        (output_root / name).write_text(
            outcar if name == "OUTCAR" else f"tiny {name}\n",
            encoding="utf-8",
        )

    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    certificate_path = reference_root / "ARTIFACT-REVALIDATION-507717.json"
    certificate_path.write_text("{}\n", encoding="utf-8")
    certificate_path.chmod(0o444)
    _install_tiny_postflight_science_readers(tmp_path, monkeypatch)

    real_oracle_sha256 = oracle._sha256

    def tiny_science_sha256(path: Path) -> str:
        candidate = Path(path).expanduser().resolve()
        if candidate == certificate_path.resolve():
            return "eb567ffb3babd407881fe5b6d47b107cefd035690e23732321875c4f8be8a754"
        return real_oracle_sha256(candidate)

    monkeypatch.setattr(oracle, "_sha256", tiny_science_sha256)
    report = _call_postflight_validator(
        kind,
        tmp_path=tmp_path,
        supply_revalidation_path=False,
    )
    assert isinstance(report, dict)

    report_name = (
        "ACTIVE-VALIDATION-123.json" if kind == "active" else "VALIDATION-123.json"
    )
    report_path = output_root / report_name
    loaded = _assert_strict_json_artifact(report_path, report, readonly=True)
    assert loaded["schema"] == {
        "active": "tpt-bulk-active16-paw-overlap-validation-v1",
        "native": "tpt-bulk-native-paw-overlap-validation-v1",
        "pure_g": "tpt-bulk-pure-g-paw-overlap-validation-v1",
        "mixed_qg": "tpt-bulk-mixed-qg-paw-overlap-validation-v1",
    }[kind]
    selected_paths = _selected_provenance_key_paths(loaded)
    expected_selected = {
        "selected_provenance_version",
        "selected_provenance_scope",
        "selected_provenance_hashes_at_preparation",
        "selected_provenance_hashes_at_validation",
        "selected_provenance_drift",
    }
    assert set(selected_paths) == expected_selected
    assert selected_paths == {key: [(key,)] for key in expected_selected}


def test_four_actual_postflights_route_same_keyset_drift_to_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _current_oracle_selected_hashes()
    changed = dict(current)
    changed[sorted(changed)[0]] = "0" * 64
    calls: list[str] = []

    monkeypatch.setattr(oracle, "ensure_not_running_compute_on_login_node", lambda _: None)
    monkeypatch.setattr(
        oracle,
        "_load_active_submission_receipt",
        lambda **_kwargs: ({}, "b" * 64),
    )
    monkeypatch.setattr(
        oracle,
        "_load_and_validate_submission_receipt",
        lambda **_kwargs: ({}, "b" * 64),
    )

    def reached_authorization(**_kwargs):
        calls.append("called")
        raise RuntimeError("authorization reached")

    monkeypatch.setattr(
        oracle,
        "_load_external_revalidation_authorization",
        reached_authorization,
    )
    for kind in ("active", "native", "pure_g", "mixed_qg"):
        output_root = tmp_path / kind
        output_root.mkdir()
        (output_root / "SOURCE_SEAL.json").write_text(
            json.dumps(_minimal_postflight_seal(kind, changed)) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="authorization reached"):
            _call_postflight_validator(kind, tmp_path=tmp_path)
    assert calls == ["called"] * 4


def test_four_actual_postflights_reject_keyset_drift_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _current_oracle_selected_hashes()
    removed = sorted(current)[0]
    malformed = {key: value for key, value in current.items() if key != removed}
    called = False

    monkeypatch.setattr(oracle, "ensure_not_running_compute_on_login_node", lambda _: None)
    monkeypatch.setattr(
        oracle,
        "_load_active_submission_receipt",
        lambda **_kwargs: ({}, "b" * 64),
    )
    monkeypatch.setattr(
        oracle,
        "_load_and_validate_submission_receipt",
        lambda **_kwargs: ({}, "b" * 64),
    )

    def forbidden_authorization(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("keyset drift entered authorization")

    monkeypatch.setattr(
        oracle,
        "_load_external_revalidation_authorization",
        forbidden_authorization,
    )
    for kind in ("active", "native", "pure_g", "mixed_qg"):
        output_root = tmp_path / kind
        output_root.mkdir()
        (output_root / "SOURCE_SEAL.json").write_text(
            json.dumps(_minimal_postflight_seal(kind, malformed)) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="keyset mismatch"):
            _call_postflight_validator(kind, tmp_path=tmp_path)
    assert called is False


def test_active_handoff_accepts_exact_inventory_and_rejects_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    for name in oracle._PARENT_HASHES:
        if name != "WAVECAR":
            (parent / name).write_text(
                _PARENT_INCAR if name == "INCAR" else name,
                encoding="utf-8",
            )
    expected = {
        name: (_sha256(parent / name) if name != "WAVECAR" else "f" * 64)
        for name in oracle._PARENT_HASHES
    }
    fake_vasp = tmp_path / "vasp"
    fake_vasp.write_bytes(b"fake-vasp\n")
    monkeypatch.setattr(oracle, "_PARENT_HASHES", expected)
    monkeypatch.setattr(oracle, "TPT_BULK_VASP_NCL_EXECUTABLE", fake_vasp)
    monkeypatch.setattr(
        oracle,
        "_copy_wavecar",
        lambda source, destination, *, mode: shutil.copy2(source, destination),
    )

    active_root = tmp_path / "active"
    active_root.mkdir()
    active_wavecar = active_root / "WAVECAR"
    active_wavecar.write_bytes(b"active-wavecar\n")
    active_wavecar.chmod(0o444)
    current = _current_oracle_selected_hashes()
    active_seal = {
        "schema": "tpt-bulk-active-spinor-wavecar-v1",
        "status": "prepared_not_yet_paw_oracle_qualified",
        "scope": {"active_ranks_1based": [233, 248], "mesh": [12, 4, 4]},
        **_valid_selected_provenance(current),
        "output": {"sha256": _sha256(active_wavecar)},
    }
    active_seal_path = active_root / "ACTIVE_WAVECAR_SEAL.json"
    active_seal_path.write_text(json.dumps(active_seal) + "\n", encoding="utf-8")

    output_root = tmp_path / "active-oracle"
    saved = oracle.prepare_active_oracle_capsule(
        parent_root=parent,
        active_wavecar_root=active_root,
        output_root=output_root,
        case=oracle.ORACLE_CASES["native"],
    )
    assert {
        key for key in saved["runtime"] if key.startswith("selected_provenance_")
    } == {
        "selected_provenance_version",
        "selected_provenance_scope",
        "selected_provenance_hashes_at_preparation",
    }
    assert (output_root / "SOURCE_SEAL.json").read_bytes().endswith(b"\n")

    changed_path = sorted(current)[0]
    active_seal["selected_provenance_hashes_at_preparation"][changed_path] = "0" * 64
    active_seal_path.write_text(json.dumps(active_seal) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        oracle.prepare_active_oracle_capsule(
            parent_root=parent,
            active_wavecar_root=active_root,
            output_root=tmp_path / "rejected-active-oracle",
            case=oracle.ORACLE_CASES["native"],
        )
