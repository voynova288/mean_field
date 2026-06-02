#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import socket

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.RnG_hBN import load_projected_basis_cache, rlg_hbn_reference_density


PANEL_RE = re.compile(r"^xi(?P<xi>-?\d+)_V(?P<v_mev>-?\d+)meV$")


@dataclass(frozen=True)
class HFTopologyResult:
    band_indices: tuple[int, ...]
    chern_number: float
    rounded_chern_number: int
    berry_curvature: np.ndarray
    min_link_singular_value: float
    min_link_location: tuple[str, int, int]
    topology_method: str
    boundary_mode: str
    valley: int

    @property
    def integer_residual(self) -> float:
        return float(abs(self.chern_number - self.rounded_chern_number))

    @property
    def is_nearly_integer(self) -> bool:
        return self.integer_residual < 1.0e-6


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _panel_values(panel_name: str) -> tuple[int, float]:
    match = PANEL_RE.match(panel_name)
    if match is None:
        raise ValueError(f"Cannot parse panel name {panel_name!r}")
    return int(match.group("xi")), float(match.group("v_mev"))


def _string_from_archive(archive: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in archive.files:
        return ""
    value = archive[key]
    try:
        return str(value.item())
    except Exception:
        return str(value)


def _sector_indices(*, n_spin: int, n_eta: int, n_band: int, spin: int, eta: int) -> np.ndarray:
    idx = np.arange(int(n_spin) * int(n_eta) * int(n_band), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(n_band)),
        order="F",
    )
    return np.asarray(idx[int(spin), int(eta), :], dtype=int)


def _grid_lookup(k_grid_frac: np.ndarray, mesh_size: int) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, frac in enumerate(np.asarray(k_grid_frac, dtype=float).reshape((-1, 2))):
        key = tuple(int(round(float(value) * int(mesh_size))) % int(mesh_size) for value in frac)
        if key in lookup:
            raise ValueError(f"Duplicate k-grid key {key}")
        lookup[key] = int(idx)
    return lookup


def _normalize_boundary_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"periodic", "hf_periodic", "wrap", "wrapped"}:
        return "periodic"
    if normalized in {"zero_fill", "zerofill", "zhang_zero_fill", "finite_cutoff"}:
        return "zero_fill"
    raise ValueError(f"Unsupported sewing boundary mode {mode!r}")


def _shift_projected_subspace(
    vectors: np.ndarray,
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    dm: int,
    dn: int,
    boundary_mode: str,
) -> np.ndarray:
    """Shift reciprocal-grid components of a selected physical subspace.

    The RLG/hBN projected basis stores vectors as
    (local orbital, G1 index, G2 index) flattened in Fortran order.  Crossing
    the moire-BZ boundary by +b_i represents the same physical Bloch state only
    after relabelling the reciprocal component G -> G + valley*b_i.  In the
    flattened raw-G representation this means target_G gets source_{G+valley*b_i}.
    """

    arr = np.asarray(vectors, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"Expected selected vectors with shape (basis_dim, n_subspace), got {arr.shape}")
    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    local = int(local_basis_size)
    n_subspace = int(arr.shape[1])
    if arr.shape[0] != local * nx * ny:
        raise ValueError(
            f"Vector basis dimension {arr.shape[0]} does not match local_basis_size*grid_shape={local * nx * ny}"
        )

    grid = arr.reshape(local, nx, ny, n_subspace, order="F")
    mode = _normalize_boundary_mode(boundary_mode)
    dm = int(dm)
    dn = int(dn)
    if mode == "periodic":
        shifted = np.roll(grid, shift=(0, dm, dn, 0), axis=(0, 1, 2, 3))
        return shifted.reshape(arr.shape, order="F")

    out = np.zeros_like(grid)
    if abs(dm) >= nx or abs(dn) >= ny:
        return out.reshape(arr.shape, order="F")
    if dm >= 0:
        dst_x = slice(dm, nx)
        src_x = slice(0, nx - dm)
    else:
        dst_x = slice(0, nx + dm)
        src_x = slice(-dm, nx)
    if dn >= 0:
        dst_y = slice(dn, ny)
        src_y = slice(0, ny - dn)
    else:
        dst_y = slice(0, ny + dn)
        src_y = slice(-dn, ny)
    out[:, dst_x, dst_y, :] = grid[:, src_x, src_y, :]
    return out.reshape(arr.shape, order="F")


def _sew_boundary_target(
    target: np.ndarray,
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    valley: int,
    reciprocal_step: tuple[int, int],
    boundary_mode: str,
) -> np.ndarray:
    valley_sign = int(valley)
    if valley_sign not in {1, -1}:
        raise ValueError(f"Expected valley=+/-1 for boundary sewing, got {valley}")
    return _shift_projected_subspace(
        target,
        local_basis_size=int(local_basis_size),
        grid_shape=grid_shape,
        dm=-valley_sign * int(reciprocal_step[0]),
        dn=-valley_sign * int(reciprocal_step[1]),
        boundary_mode=boundary_mode,
    )


def _unit_subspace_link(left: np.ndarray, right: np.ndarray, *, atol: float = 1.0e-14) -> tuple[complex, float]:
    overlap = np.asarray(left, dtype=np.complex128).conjugate().T @ np.asarray(right, dtype=np.complex128)
    if overlap.shape == (1, 1):
        value = complex(overlap[0, 0])
        magnitude = abs(value)
        if magnitude <= atol:
            raise ValueError("near-zero line-bundle overlap link; selected band is not isolated on this mesh")
        return value / magnitude, float(magnitude)

    singular_values = np.linalg.svd(overlap, compute_uv=False)
    min_singular = float(np.min(singular_values))
    determinant = complex(np.linalg.det(overlap))
    determinant_magnitude = abs(determinant)
    if min_singular <= atol or determinant_magnitude <= atol:
        raise ValueError("near-zero subspace overlap link; selected subspace is not isolated on this mesh")
    return determinant / determinant_magnitude, min_singular


def _compute_projected_basis_topology(
    wavefunctions: np.ndarray,
    *,
    band_indices: tuple[int, ...],
    local_basis_size: int,
    grid_shape: tuple[int, int],
    valley: int,
    boundary_mode: str,
    sew_boundaries: bool,
) -> HFTopologyResult:
    vectors = np.asarray(wavefunctions, dtype=np.complex128)
    if vectors.ndim != 4:
        raise ValueError(f"Expected wavefunctions shape (mesh_x, mesh_y, basis_dim, n_subspace), got {vectors.shape}")
    mesh_x, mesh_y = vectors.shape[:2]
    ux = np.zeros((mesh_x, mesh_y), dtype=np.complex128)
    uy = np.zeros((mesh_x, mesh_y), dtype=np.complex128)
    min_link = float("inf")
    min_location = ("", -1, -1)
    resolved_boundary_mode = _normalize_boundary_mode(boundary_mode)

    for ix in range(mesh_x):
        ix_next = (ix + 1) % mesh_x
        for iy in range(mesh_y):
            iy_next = (iy + 1) % mesh_y
            left = vectors[ix, iy]

            right_x = vectors[ix_next, iy]
            if sew_boundaries and ix == mesh_x - 1:
                right_x = _sew_boundary_target(
                    right_x,
                    local_basis_size=int(local_basis_size),
                    grid_shape=grid_shape,
                    valley=int(valley),
                    reciprocal_step=(1, 0),
                    boundary_mode=resolved_boundary_mode,
                )
            ux[ix, iy], mag = _unit_subspace_link(left, right_x)
            if mag < min_link:
                min_link = float(mag)
                min_location = ("b1", int(ix), int(iy))

            right_y = vectors[ix, iy_next]
            if sew_boundaries and iy == mesh_y - 1:
                right_y = _sew_boundary_target(
                    right_y,
                    local_basis_size=int(local_basis_size),
                    grid_shape=grid_shape,
                    valley=int(valley),
                    reciprocal_step=(0, 1),
                    boundary_mode=resolved_boundary_mode,
                )
            uy[ix, iy], mag = _unit_subspace_link(left, right_y)
            if mag < min_link:
                min_link = float(mag)
                min_location = ("b2", int(ix), int(iy))

    berry_curvature = np.zeros((mesh_x, mesh_y), dtype=float)
    for ix in range(mesh_x):
        ix_next = (ix + 1) % mesh_x
        for iy in range(mesh_y):
            iy_next = (iy + 1) % mesh_y
            plaquette = ux[ix, iy] * uy[ix_next, iy] / (ux[ix, iy_next] * uy[ix, iy])
            berry_curvature[ix, iy] = float(np.angle(plaquette))

    chern_number = float(np.sum(berry_curvature) / (2.0 * np.pi))
    method = "fhs_projected_basis_sewn" if sew_boundaries else "fhs_projected_basis_unsewn_diagnostic"
    return HFTopologyResult(
        band_indices=tuple(int(value) for value in band_indices),
        chern_number=chern_number,
        rounded_chern_number=int(np.rint(chern_number)),
        berry_curvature=berry_curvature,
        min_link_singular_value=float(min_link),
        min_link_location=min_location,
        topology_method=method,
        boundary_mode=resolved_boundary_mode,
        valley=int(valley),
    )


def _selected_hf_wavefunctions_on_grid(
    *,
    hamiltonian: np.ndarray,
    basis_wavefunctions: np.ndarray,
    k_grid_frac: np.ndarray,
    mesh_size: int,
    sector: np.ndarray,
    eta: int,
    band_indices: tuple[int, ...],
) -> np.ndarray:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    basis_wavefunctions = np.asarray(basis_wavefunctions, dtype=np.complex128)
    lookup = _grid_lookup(k_grid_frac, mesh_size)
    basis_dim = int(basis_wavefunctions.shape[0])
    selected = np.zeros((int(mesh_size), int(mesh_size), basis_dim, len(band_indices)), dtype=np.complex128)
    for ix in range(int(mesh_size)):
        for iy in range(int(mesh_size)):
            ik = lookup[(ix, iy)]
            block = hamiltonian[:, :, ik][np.ix_(sector, sector)]
            _, coeffs = np.linalg.eigh(block)
            active_basis = basis_wavefunctions[:, :, int(eta), ik]
            physical = active_basis @ coeffs[:, list(band_indices)]
            q_mat, _ = np.linalg.qr(physical)
            selected[ix, iy, :, :] = q_mat[:, : len(band_indices)]
    return selected


def _selected_hf_occupation_on_grid(
    *,
    hamiltonian: np.ndarray,
    occupation_projector: np.ndarray,
    k_grid_frac: np.ndarray,
    mesh_size: int,
    sector: np.ndarray,
    band_indices: tuple[int, ...],
) -> np.ndarray:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    occupation_projector = np.asarray(occupation_projector, dtype=np.complex128)
    lookup = _grid_lookup(k_grid_frac, mesh_size)
    occupations = np.zeros((int(mesh_size), int(mesh_size)), dtype=float)
    for ix in range(int(mesh_size)):
        for iy in range(int(mesh_size)):
            ik = lookup[(ix, iy)]
            block = hamiltonian[:, :, ik][np.ix_(sector, sector)]
            _, coeffs = np.linalg.eigh(block)
            projector_block = occupation_projector[:, :, ik][np.ix_(sector, sector)]
            selected = coeffs[:, list(band_indices)]
            # Saved HF densities use P_ab = <c_a^\dagger c_b>.  For a ket
            # column u, the occupancy is u.T @ P @ u.conj().
            occupations[ix, iy] = float(np.trace(selected.T @ projector_block @ selected.conjugate()).real)
    return occupations


def _stats_payload(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _topology_payload(result) -> dict[str, object]:
    return {
        "band_indices": [int(value) for value in result.band_indices],
        "chern_number": float(result.chern_number),
        "rounded_chern_number": int(result.rounded_chern_number),
        "integer_residual": float(result.integer_residual),
        "is_nearly_integer": bool(result.is_nearly_integer),
        "max_abs_berry_flux_over_pi": float(np.max(np.abs(result.berry_curvature)) / np.pi),
        "min_link_singular_value": float(result.min_link_singular_value),
        "min_link_location": list(result.min_link_location),
        "topology_method": str(result.topology_method),
        "boundary_mode": str(result.boundary_mode),
        "valley": int(result.valley),
    }


def _paper_expected_abs_chern(xi: int) -> int:
    if int(xi) == 0:
        return 0
    if int(xi) == 1:
        return 1
    raise ValueError(f"No Fig. 6 paper Chern expectation for xi={xi}")


def _compute_panel(
    panel_dir: Path,
    *,
    config: dict[str, object],
    cache_dir: Path,
    spin_index: int,
    eta_index: int,
    compare_paper_fig6: bool,
) -> dict[str, object]:
    xi, v_mev = _panel_values(panel_dir.name)
    state_path = panel_dir / "hf_ground_state.npz"
    convergence_path = panel_dir / "hf_convergence.json"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    if not convergence_path.exists():
        raise FileNotFoundError(convergence_path)

    archive = np.load(state_path)
    convergence = _read_json(convergence_path)
    basis_key = str(convergence.get("basis_cache_key") or _string_from_archive(archive, "cache_key_basis"))
    if not basis_key:
        raise ValueError(f"{state_path} does not record a basis cache key")
    basis_data = load_projected_basis_cache(cache_dir, basis_key)

    hamiltonian = np.asarray(archive["hamiltonian"], dtype=np.complex128)
    density_delta = np.asarray(archive["density"], dtype=np.complex128)
    mesh_size = int(config["k_mesh_size"])
    n_spin = 2
    n_eta = 2
    n_band = int(hamiltonian.shape[0]) // (n_spin * n_eta)
    active_valence = int(config["active_valence_bands"])
    if active_valence <= 0 or active_valence >= n_band:
        raise ValueError(f"active_valence_bands={active_valence} is incompatible with n_band={n_band}")
    if basis_data.basis.n_band != n_band:
        raise ValueError(f"Basis n_band={basis_data.basis.n_band} does not match HF n_band={n_band}")
    if basis_data.basis.n_flavor <= int(eta_index):
        raise ValueError(f"eta_index={eta_index} is outside basis flavor count {basis_data.basis.n_flavor}")
    basis_valley = int(basis_data.valleys[int(eta_index)])
    if np.max(np.abs(np.asarray(archive["k_grid_frac"], dtype=float) - np.asarray(basis_data.k_grid_frac, dtype=float))) > 1.0e-10:
        raise ValueError(f"{state_path} k_grid_frac does not match basis cache {basis_key}")

    sector = _sector_indices(
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
        spin=int(spin_index),
        eta=int(eta_index),
    )
    reference_density = rlg_hbn_reference_density(
        int(hamiltonian.shape[0]),
        int(hamiltonian.shape[2]),
        scheme=str(config.get("scheme", "average")),
        active_valence_bands=int(active_valence),
        n_spin=n_spin,
        n_eta=n_eta,
    )
    occupation_projector = density_delta + reference_density
    targets = {
        "top_valence": (active_valence - 1,),
        "occupied_conduction": (active_valence,),
        "central_pair": (active_valence - 1, active_valence),
    }
    results: dict[str, object] = {}
    berry_payload: dict[str, np.ndarray | float | int] = {}
    occupation_payload: dict[str, np.ndarray] = {}
    for name, band_indices in targets.items():
        wavefunctions = _selected_hf_wavefunctions_on_grid(
            hamiltonian=hamiltonian,
            basis_wavefunctions=basis_data.basis.wavefunctions,
            k_grid_frac=np.asarray(archive["k_grid_frac"], dtype=float),
            mesh_size=mesh_size,
            sector=sector,
            eta=int(eta_index),
            band_indices=band_indices,
        )
        selected_occupation = _selected_hf_occupation_on_grid(
            hamiltonian=hamiltonian,
            occupation_projector=occupation_projector,
            k_grid_frac=np.asarray(archive["k_grid_frac"], dtype=float),
            mesh_size=mesh_size,
            sector=sector,
            band_indices=band_indices,
        )
        result = _compute_projected_basis_topology(
            wavefunctions,
            band_indices=tuple(range(len(band_indices))),
            local_basis_size=int(basis_data.basis.local_basis_size),
            grid_shape=tuple(int(value) for value in basis_data.basis.grid_shape),
            valley=basis_valley,
            boundary_mode=str(basis_data.basis.boundary_mode),
            sew_boundaries=True,
        )
        unsewn = _compute_projected_basis_topology(
            wavefunctions,
            band_indices=tuple(range(len(band_indices))),
            local_basis_size=int(basis_data.basis.local_basis_size),
            grid_shape=tuple(int(value) for value in basis_data.basis.grid_shape),
            valley=basis_valley,
            boundary_mode=str(basis_data.basis.boundary_mode),
            sew_boundaries=False,
        )
        zero_fill = _compute_projected_basis_topology(
            wavefunctions,
            band_indices=tuple(range(len(band_indices))),
            local_basis_size=int(basis_data.basis.local_basis_size),
            grid_shape=tuple(int(value) for value in basis_data.basis.grid_shape),
            valley=basis_valley,
            boundary_mode="zero_fill",
            sew_boundaries=True,
        )
        payload = _topology_payload(result)
        payload["hf_sector_band_indices"] = [int(value) for value in band_indices]
        payload["absolute_rounded_chern_number"] = int(abs(result.rounded_chern_number))
        payload["saved_density_occupation_stats"] = _stats_payload(selected_occupation)
        payload["diagnostics"] = {
            "unsewn_raw_periodic_identification": _topology_payload(unsewn),
            "sewn_zero_fill_boundary": _topology_payload(zero_fill),
        }
        results[name] = payload
        berry_payload[f"berry_curvature_{name}"] = np.asarray(result.berry_curvature, dtype=float)
        berry_payload[f"chern_number_{name}"] = float(result.chern_number)
        berry_payload[f"rounded_chern_number_{name}"] = int(result.rounded_chern_number)
        berry_payload[f"berry_curvature_unsewn_{name}"] = np.asarray(unsewn.berry_curvature, dtype=float)
        berry_payload[f"chern_number_unsewn_{name}"] = float(unsewn.chern_number)
        berry_payload[f"berry_curvature_sewn_zero_fill_{name}"] = np.asarray(zero_fill.berry_curvature, dtype=float)
        berry_payload[f"chern_number_sewn_zero_fill_{name}"] = float(zero_fill.chern_number)
        occupation_payload[f"saved_density_occupation_{name}"] = np.asarray(selected_occupation, dtype=float)

    occupied = results["occupied_conduction"]
    assert isinstance(occupied, dict)
    paper_expected_abs = _paper_expected_abs_chern(xi) if compare_paper_fig6 else None
    comparison = None
    if paper_expected_abs is not None:
        observed_abs = int(occupied["absolute_rounded_chern_number"])
        comparison = {
            "paper_reference": "Kwan et al. arXiv:2312.11617v1 Fig. 6",
            "paper_expected_abs_chern": int(paper_expected_abs),
            "observed_abs_rounded_chern": int(observed_abs),
            "matches_paper_abs_chern": bool(observed_abs == int(paper_expected_abs)),
            "note": "Signed Chern follows the local valley/gauge convention; Fig. 6 comparison is in |C|.",
        }

    payload = {
        "panel": panel_dir.name,
        "xi": int(xi),
        "v_mev": float(v_mev),
        "state": str(state_path.resolve()),
        "convergence": str(convergence_path.resolve()),
        "basis_cache_key": basis_key,
        "cache_dir": str(cache_dir),
        "mesh_size": int(mesh_size),
        "spin_index": int(spin_index),
        "eta_index": int(eta_index),
        "basis_valley": int(basis_valley),
        "basis_valleys": [int(value) for value in basis_data.valleys],
        "basis_boundary_mode": str(basis_data.basis.boundary_mode),
        "basis_grid_shape": [int(value) for value in basis_data.basis.grid_shape],
        "basis_local_basis_size": int(basis_data.basis.local_basis_size),
        "active_valence_bands": int(active_valence),
        "n_band_per_sector": int(n_band),
        "saved_density_occupation_convention": "P_ab=<c_a^dagger c_b>; selected ket occupancy is trace(U.T @ P @ U.conj())",
        "chern": results,
        "paper_fig6_comparison": comparison,
    }
    _write_json(panel_dir / "hf_chern_numbers.json", payload)
    np.savez_compressed(panel_dir / "hf_berry_curvature.npz", **berry_payload)
    np.savez_compressed(panel_dir / "hf_saved_density_occupations.npz", **occupation_payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RLG/hBN HF middle-band Chern numbers from saved Fig. 6 states.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--spin-index", type=int, default=0)
    parser.add_argument("--eta-index", type=int, default=0)
    parser.add_argument("--compare-paper-fig6", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_not_running_compute_on_login_node("RLG/hBN HF Chern number computation")
    source_dir = args.source_dir.resolve()
    config = _read_json(source_dir / "paper_hf_config.json")
    cache_dir = args.cache_dir.resolve() if args.cache_dir is not None else Path(str(config["cache_dir"])).resolve()
    panel_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir() and (path / "hf_ground_state.npz").exists())
    if not panel_dirs:
        raise FileNotFoundError(f"No panel hf_ground_state.npz files found under {source_dir}")

    panels = [
        _compute_panel(
            panel_dir,
            config=config,
            cache_dir=cache_dir,
            spin_index=int(args.spin_index),
            eta_index=int(args.eta_index),
            compare_paper_fig6=bool(args.compare_paper_fig6),
        )
        for panel_dir in panel_dirs
    ]
    summary = {
        "source_dir": str(source_dir),
        "cache_dir": str(cache_dir),
        "hostname": socket.gethostname(),
        "spin_index": int(args.spin_index),
        "eta_index": int(args.eta_index),
        "compare_paper_fig6": bool(args.compare_paper_fig6),
        "panels": panels,
    }
    _write_json(source_dir / "hf_chern_summary.json", summary)

    report_lines = [
        "# RLG/hBN Fig. 6 HF Chern Summary",
        "",
        "| panel | occupied conduction C | rounded | |C| | max flux/pi | occ min/mean/max | paper | match |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for panel in panels:
        occupied = panel["chern"]["occupied_conduction"]
        comparison = panel.get("paper_fig6_comparison") or {}
        report_lines.append(
            "| {panel} | {chern:.8f} | {rounded} | {abs_rounded} | {flux:.6f} | {occ} | {expected} | {match} |".format(
                panel=panel["panel"],
                chern=float(occupied["chern_number"]),
                rounded=int(occupied["rounded_chern_number"]),
                abs_rounded=int(occupied["absolute_rounded_chern_number"]),
                flux=float(occupied["max_abs_berry_flux_over_pi"]),
                occ="{min:.6f}/{mean:.6f}/{max:.6f}".format(**occupied["saved_density_occupation_stats"]),
                expected=comparison.get("paper_expected_abs_chern", "n/a"),
                match=comparison.get("matches_paper_abs_chern", "n/a"),
            )
        )
    (source_dir / "hf_chern_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    for panel in panels:
        occupied = panel["chern"]["occupied_conduction"]
        comparison = panel.get("paper_fig6_comparison") or {}
        print(
            f"[chern] {panel['panel']} occupied_conduction="
            f"{float(occupied['chern_number']):.8f} rounded={int(occupied['rounded_chern_number'])} "
            f"abs={int(occupied['absolute_rounded_chern_number'])} "
            f"paper_abs={comparison.get('paper_expected_abs_chern', 'n/a')} "
            f"match={comparison.get('matches_paper_abs_chern', 'n/a')}",
            flush=True,
        )
    print(f"[done] summary={source_dir / 'hf_chern_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
