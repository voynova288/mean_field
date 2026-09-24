from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np
import pytest

from mean_field.systems.tpt_bulk.wavefunctions import (
    Wannier90CheckpointReader,
    WavecarSpinorReader,
    contract_spinor_plane_wave_density_vertex,
    iter_wannier90_mmn_blocks,
    project_spinor_plane_wave_coefficients,
    read_wannier90_eig,
    reciprocal_wrap_for_mesh_transfer,
    write_projected_spinor_wavecar,
)


def _write_fortran_record(handle, payload: bytes) -> None:
    handle.write(struct.pack("<i", len(payload)))
    handle.write(payload)
    handle.write(struct.pack("<i", len(payload)))


def _fortran_bytes(array: np.ndarray, dtype: str) -> bytes:
    values = np.asarray(array, dtype=np.dtype(dtype), order="F")
    return values.tobytes(order="F")


def _write_tiny_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray]:
    num_bands = 3
    num_wann = 2
    num_kpts = 2
    lwindow = np.asarray(
        [[True, True], [True, False], [False, True]], dtype=bool
    )
    u_opt = np.zeros((num_bands, num_wann, num_kpts), dtype=np.complex128)
    u_opt[:2, :, 0] = np.eye(2)
    u_opt[:2, :, 1] = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    u = np.zeros((num_wann, num_wann, num_kpts), dtype=np.complex128)
    u[:, :, 0] = np.eye(2)
    u[:, :, 1] = np.diag([1j, -1j])

    with path.open("wb") as handle:
        _write_fortran_record(handle, b"synthetic checkpoint".ljust(33))
        _write_fortran_record(handle, struct.pack("<i", num_bands))
        _write_fortran_record(handle, struct.pack("<i", 0))
        _write_fortran_record(handle, b"")
        _write_fortran_record(handle, _fortran_bytes(np.eye(3), "<f8"))
        _write_fortran_record(handle, _fortran_bytes(2 * np.pi * np.eye(3), "<f8"))
        _write_fortran_record(handle, struct.pack("<i", num_kpts))
        _write_fortran_record(handle, _fortran_bytes(np.asarray([2, 1, 1]), "<i4"))
        kpoints_fortran = np.asarray([[0.0, 0.5], [0.0, 0.0], [0.0, 0.0]])
        _write_fortran_record(handle, _fortran_bytes(kpoints_fortran, "<f8"))
        _write_fortran_record(handle, struct.pack("<i", 1))
        _write_fortran_record(handle, struct.pack("<i", num_wann))
        _write_fortran_record(handle, b"postwann".ljust(20))
        _write_fortran_record(handle, struct.pack("<i", -1))
        _write_fortran_record(handle, struct.pack("<d", 0.25))
        logical = np.where(lwindow, -1, 0).astype("<i4")
        _write_fortran_record(handle, logical.tobytes(order="F"))
        _write_fortran_record(handle, _fortran_bytes(np.asarray([2, 2]), "<i4"))
        _write_fortran_record(handle, _fortran_bytes(u_opt, "<c16"))
        _write_fortran_record(handle, _fortran_bytes(u, "<c16"))
    return u_opt, u


def test_wannier90_checkpoint_reader_reconstructs_compact_outer_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny.chk"
    u_opt, u = _write_tiny_checkpoint(path)
    with Wannier90CheckpointReader(path) as reader:
        metadata = reader.metadata
        assert metadata.num_bands == 3
        assert metadata.num_wann == 2
        assert metadata.num_kpts == 2
        assert metadata.mp_grid == (2, 1, 1)
        assert metadata.nntot == 1
        np.testing.assert_allclose(metadata.kpoints_fractional, [[0, 0, 0], [0.5, 0, 0]])
        gauge0 = reader.dft_to_wannier_gauge(0)
        gauge1 = reader.dft_to_wannier_gauge(1)

    expected0 = np.zeros((3, 2), dtype=np.complex128)
    expected0[[0, 1]] = u_opt[:2, :, 0] @ u[:, :, 0]
    expected1 = np.zeros((3, 2), dtype=np.complex128)
    expected1[[0, 2]] = u_opt[:2, :, 1] @ u[:, :, 1]
    np.testing.assert_allclose(gauge0, expected0)
    np.testing.assert_allclose(gauge1, expected1)
    np.testing.assert_allclose(gauge0.conj().T @ gauge0, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(gauge1.conj().T @ gauge1, np.eye(2), atol=1e-14)


def test_read_wannier90_eig_requires_band_fastest_order(tmp_path: Path) -> None:
    path = tmp_path / "wannier90.eig"
    path.write_text("1 1 -1.0\n2 1 0.5\n1 2 -0.8\n2 2 0.7\n")
    energies = read_wannier90_eig(path, num_bands=2, num_kpts=2)
    np.testing.assert_allclose(energies, [[-1.0, 0.5], [-0.8, 0.7]])
    path.write_text("2 1 0.5\n1 1 -1.0\n1 2 -0.8\n2 2 0.7\n")
    with pytest.raises(ValueError, match="band indices"):
        read_wannier90_eig(path, num_bands=2, num_kpts=2)


def test_mmn_reader_preserves_source_bra_target_ket_matrix_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wannier90.mmn"
    first = np.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    second = np.asarray([[9 + 1j, 2 + 3j], [4 + 5j, 6 + 7j]])
    with path.open("w") as handle:
        handle.write("synthetic mmn\n2 1 2\n")
        for shift, matrix in (((1, 0, 0), first), ((-1, 0, 0), second)):
            handle.write(f"1 1 {shift[0]} {shift[1]} {shift[2]}\n")
            for n in range(2):
                for m in range(2):
                    value = matrix[m, n]
                    handle.write(f"{value.real:.17g} {value.imag:.17g}\n")
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    blocks = list(
        iter_wannier90_mmn_blocks(
            path,
            expected_num_bands=2,
            expected_num_kpts=1,
            expected_nntot=2,
            expected_sha256=expected_sha256,
        )
    )
    assert [block.target_cell_shift_integer for block in blocks] == [
        (1, 0, 0),
        (-1, 0, 0),
    ]
    np.testing.assert_allclose(blocks[0].overlap_source_bra_target_ket, first)
    np.testing.assert_allclose(blocks[1].overlap_source_bra_target_ket, second)
    with pytest.raises(ValueError, match="wannier90.mmn hash mismatch"):
        list(iter_wannier90_mmn_blocks(path, expected_sha256="0" * 64))
    with path.open("a") as handle:
        handle.write("unexpected trailing record\n")
    with pytest.raises(ValueError, match="trailing content"):
        list(iter_wannier90_mmn_blocks(path))


def test_projected_spinor_wavecar_round_trip_preserves_active_coefficients(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.WAVECAR"
    output_path = tmp_path / "active.WAVECAR"
    record_length = 128
    lattice = 10.0 * np.eye(3)
    coefficients = np.asarray(
        [
            [[[1.0 + 0.0j], [0.0 + 2.0j]]],
            [[[3.0 + 0.0j], [0.0 + 4.0j]]],
        ],
        dtype="<c8",
    ).reshape((2, 2, 1))
    with source_path.open("wb") as handle:
        def write_record(index: int, values: np.ndarray) -> None:
            handle.seek(index * record_length)
            np.ascontiguousarray(values).tofile(handle)

        write_record(0, np.asarray((record_length, 1, 45200), dtype="<f8"))
        write_record(
            1,
            np.concatenate(
                (
                    np.asarray((2, 2, 1.0), dtype="<f8"),
                    lattice.reshape(-1).astype("<f8"),
                    np.asarray((0.0,), dtype="<f8"),
                )
            ),
        )
        write_record(
            2,
            np.asarray((2, 0, 0, 0, -1, 0, 1, 1, 0, 0), dtype="<f8"),
        )
        write_record(3, coefficients[0].reshape(-1))
        write_record(4, coefficients[1].reshape(-1))
        write_record(
            5,
            np.asarray((2, 0, 0, 0, -0.5, 0, 1, 0.5, 0, 0), dtype="<f8"),
        )
        write_record(6, coefficients[0].reshape(-1))
        write_record(7, coefficients[1].reshape(-1))
        handle.truncate(8 * record_length)

    gauge = np.repeat(
        np.asarray([[[1.0], [1.0j]]], dtype=np.complex128) / np.sqrt(2.0),
        2,
        axis=0,
    )
    energies = np.asarray([[0.25], [0.5]])
    occupations = np.asarray([[1.0], [0.0]])
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    metadata = write_projected_spinor_wavecar(
        source_path,
        output_path,
        dft_to_active_gauge_by_k=gauge,
        active_eigenvalues_ev=energies,
        active_occupations=occupations,
        expected_source_sha256=source_sha256,
    )
    assert metadata["active_num_bands"] == 1
    assert metadata["num_kpoints"] == 2
    assert output_path.stat().st_mode & 0o777 == 0o444
    with WavecarSpinorReader(output_path) as reader:
        assert reader.metadata.num_bands == 1
        assert reader.metadata.num_kpts == 2
        assert reader.metadata.format_tag == 45210
        assert reader.metadata.coefficient_dtype == np.dtype("<c16")
        np.testing.assert_allclose(reader.metadata.eigenvalues_ev, energies)
        actual = reader.read_spinor_coefficients(0, [0])
    expected = project_spinor_plane_wave_coefficients(coefficients, gauge[0])
    np.testing.assert_allclose(actual, expected, atol=0.0)

    parallel_path = tmp_path / "active-parallel.WAVECAR"
    parallel_metadata = write_projected_spinor_wavecar(
        source_path,
        parallel_path,
        dft_to_active_gauge_by_k=gauge,
        active_eigenvalues_ev=energies,
        active_occupations=occupations,
        expected_source_sha256=source_sha256,
        projection_workers=2,
    )
    assert parallel_metadata["projection_workers"] == 2
    assert parallel_path.read_bytes() == output_path.read_bytes()


def test_plane_wave_projection_and_finite_g_contraction_match_literal_sum() -> None:
    dft = np.asarray(
        [
            [[1.0, 2.0], [0.5j, 0.0]],
            [[-0.25, 0.75j], [1.0, -0.5]],
        ],
        dtype=np.complex128,
    )
    gauge = np.asarray([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128)
    source = project_spinor_plane_wave_coefficients(dft, gauge)
    target = source * np.asarray([1.0, -1j])[:, None, None]
    source_g = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    target_g = np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.int64)
    rho = contract_spinor_plane_wave_density_vertex(
        source_coefficients=source,
        source_g_integer=source_g,
        target_coefficients=target,
        target_g_integer=target_g,
        reciprocal_shift_integer=(1, 0, 0),
    )
    expected = np.zeros((2, 2), dtype=np.complex128)
    for m in range(2):
        for n in range(2):
            for spinor in range(2):
                for source_index, g_source in enumerate(source_g):
                    g_target = g_source + np.asarray([1, 0, 0])
                    target_index = np.flatnonzero(
                        np.all(target_g == g_target[None, :], axis=1)
                    )[0]
                    expected[m, n] += (
                        target[m, spinor, target_index].conjugate()
                        * source[n, spinor, source_index]
                    )
    np.testing.assert_allclose(rho, expected, atol=1e-14)


def test_reciprocal_wrap_closes_signed_mesh_transfer() -> None:
    wrap = reciprocal_wrap_for_mesh_transfer(
        source_k_fractional=(-1.0 / 12.0, -0.25, -0.25),
        target_k_fractional=(0.0, 0.0, 0.0),
        q_mesh_shift=(1, 1, 1),
        mesh=(12, 4, 4),
    )
    np.testing.assert_array_equal(wrap, [0, 0, 0])
    boundary_wrap = reciprocal_wrap_for_mesh_transfer(
        source_k_fractional=(5.0 / 12.0, 0.0, 0.0),
        target_k_fractional=(-0.5, 0.0, 0.0),
        q_mesh_shift=(1, 0, 0),
        mesh=(12, 4, 4),
    )
    np.testing.assert_array_equal(boundary_wrap, [1, 0, 0])
