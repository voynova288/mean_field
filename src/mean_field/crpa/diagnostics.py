from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .workflow import CRPAResult


@dataclass(frozen=True)
class CRPADiagnosticSummary:
    q_peak_nm_inv: float
    eps_total_peak: float
    eps_total_q0: float
    eps_total_q04: float
    eps_total_q08: float
    eps_total_q12: float
    eps_diag_imag_max_abs: float
    chi0_antihermitian_max_abs: float
    radial_std_max_0_1p2: float
    full_q_abs_max_nm_inv: float


def _nm_inv(values: np.ndarray | complex | float, graphene_lattice_angstrom: float) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128) / (float(graphene_lattice_angstrom) / 10.0)


def _xy_nm_inv(values: np.ndarray, graphene_lattice_angstrom: float) -> tuple[np.ndarray, np.ndarray]:
    scaled = _nm_inv(values, graphene_lattice_angstrom)
    return np.asarray(scaled.real, dtype=float), np.asarray(scaled.imag, dtype=float)


def q_shell_index(q_shift: Iterable[int]) -> int:
    """Return a hexagonal shell index for integer moire reciprocal shifts."""

    m, n = (int(v) for v in q_shift)
    return int(max(abs(m), abs(n), abs(m + n)))


def _representative_curve(
    q_abs_nm_inv: np.ndarray,
    eps_total: np.ndarray,
    *,
    x_max_nm_inv: float = 1.2,
    bin_width_nm_inv: float = 0.0125,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_abs_nm_inv, dtype=float).reshape(-1)
    eps = np.asarray(eps_total, dtype=float).reshape(-1)
    finite = np.isfinite(q) & np.isfinite(eps) & (q >= 0.0) & (q <= float(x_max_nm_inv))
    q = q[finite]
    eps = eps[finite]
    if q.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=int)

    bins = np.arange(0.0, float(x_max_nm_inv) + float(bin_width_nm_inv), float(bin_width_nm_inv))
    zero = np.isclose(q, 0.0, atol=1.0e-14)
    xs: list[float] = [0.0] if np.any(zero) else []
    ys: list[float] = [float(np.median(eps[zero]))] if np.any(zero) else []
    counts: list[int] = [int(np.count_nonzero(zero))] if np.any(zero) else []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        in_bin = (q > lo) & (q < hi)
        if hi >= float(x_max_nm_inv):
            in_bin = (q > lo) & (q <= hi)
        if not np.any(in_bin):
            continue
        xs.append(float(np.median(q[in_bin])))
        ys.append(float(np.median(eps[in_bin])))
        counts.append(int(np.count_nonzero(in_bin)))

    order = np.argsort(xs)
    return np.asarray(xs, dtype=float)[order], np.asarray(ys, dtype=float)[order], np.asarray(counts, dtype=int)[order]


def write_crpa_epsilon_diagnostics_csv(result: CRPAResult, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    eps_bn = float(result.coulomb_params.epsilon_bn)
    lattice_a = float(result.coulomb_params.graphene_lattice_angstrom)
    qe_x, qe_y = _xy_nm_inv(result.q_tilde[:, None], lattice_a)
    q_x, q_y = _xy_nm_inv(result.physical_q_vectors, lattice_a)
    q_abs = np.abs(_nm_inv(result.physical_q_vectors, lattice_a)).real
    q_vectors = np.asarray(result.q_vectors, dtype=np.complex128)
    qv_x, qv_y = _xy_nm_inv(q_vectors[None, :], lattice_a)
    eps_diag = np.diagonal(np.asarray(result.dielectric_matrix, dtype=np.complex128), axis1=1, axis2=2)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "qe_x_nm_inv",
                "qe_y_nm_inv",
                "Q_x_nm_inv",
                "Q_y_nm_inv",
                "q_x_nm_inv",
                "q_y_nm_inv",
                "q_abs_nm_inv",
                "Q_shell_index",
                "eps_diag_real",
                "eps_diag_imag",
                "eps_crpa",
                "eps_total",
            ]
        )
        for iq in range(result.q_tilde.shape[0]):
            for ishift, shift in enumerate(np.asarray(result.q_shifts, dtype=int)):
                eps_value = complex(eps_diag[iq, ishift])
                eps_crpa = float(eps_value.real)
                writer.writerow(
                    [
                        f"{float(qe_x[iq, 0]):.16e}",
                        f"{float(qe_y[iq, 0]):.16e}",
                        f"{float(qv_x[0, ishift]):.16e}",
                        f"{float(qv_y[0, ishift]):.16e}",
                        f"{float(q_x[iq, ishift]):.16e}",
                        f"{float(q_y[iq, ishift]):.16e}",
                        f"{float(q_abs[iq, ishift]):.16e}",
                        str(q_shell_index(shift)),
                        f"{float(eps_value.real):.16e}",
                        f"{float(eps_value.imag):.16e}",
                        f"{eps_crpa:.16e}",
                        f"{float(eps_bn * eps_crpa):.16e}",
                    ]
                )
    return path


def write_radial_bin_stats(
    result: CRPAResult,
    output_path: Path | str,
    *,
    bin_width_nm_inv: float = 0.025,
    x_max_nm_inv: float | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    eps_total = np.asarray(result.effective_epsilon, dtype=float) * float(result.coulomb_params.epsilon_bn)
    q_abs = np.abs(_nm_inv(result.physical_q_vectors, result.coulomb_params.graphene_lattice_angstrom)).real
    q = q_abs.reshape(-1)
    eps = eps_total.reshape(-1)
    finite = np.isfinite(q) & np.isfinite(eps)
    if x_max_nm_inv is not None:
        finite &= q <= float(x_max_nm_inv)
    q = q[finite]
    eps = eps[finite]
    max_q = float(x_max_nm_inv) if x_max_nm_inv is not None else float(np.max(q))
    bins = np.arange(0.0, max_q + float(bin_width_nm_inv), float(bin_width_nm_inv))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "q_lo_nm_inv",
                "q_hi_nm_inv",
                "q_center_nm_inv",
                "eps_total_mean",
                "eps_total_std",
                "eps_total_min",
                "eps_total_median",
                "eps_total_max",
                "n_points",
            ]
        )
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            in_bin = (q >= lo) & (q < hi)
            if hi >= max_q:
                in_bin = (q >= lo) & (q <= hi)
            if not np.any(in_bin):
                continue
            values = eps[in_bin]
            writer.writerow(
                [
                    f"{float(lo):.16e}",
                    f"{float(hi):.16e}",
                    f"{0.5 * (float(lo) + float(hi)):.16e}",
                    f"{float(np.mean(values)):.16e}",
                    f"{float(np.std(values)):.16e}",
                    f"{float(np.min(values)):.16e}",
                    f"{float(np.median(values)):.16e}",
                    f"{float(np.max(values)):.16e}",
                    str(int(values.size)),
                ]
            )
    return path


def _extract_reciprocal_norms(result: CRPAResult) -> tuple[float, float]:
    shifts = np.asarray(result.q_shifts, dtype=int)
    vectors = np.asarray(result.q_vectors, dtype=np.complex128)
    lattice_a = float(result.coulomb_params.graphene_lattice_angstrom)

    def lookup(target: tuple[int, int]) -> float:
        rows = np.flatnonzero(np.all(shifts == np.asarray(target, dtype=int), axis=1))
        if rows.size:
            return float(abs(_nm_inv(vectors[int(rows[0])], lattice_a)))
        q_rows = np.flatnonzero(np.all(np.asarray(result.q_indices, dtype=int) == np.asarray(target, dtype=int), axis=1))
        if q_rows.size:
            return float(abs(_nm_inv(result.q_tilde[int(q_rows[0])] * float(result.lk), lattice_a)))
        return float("nan")

    return lookup((1, 0)), lookup((0, 1))


def diagnostic_parameter_summary(result: CRPAResult) -> dict[str, object]:
    theta_rad = math.radians(float(result.theta_deg))
    lattice_a_nm = float(result.coulomb_params.graphene_lattice_angstrom) / 10.0
    moire_lattice_nm = lattice_a_nm / (2.0 * math.sin(theta_rad / 2.0))
    g1_norm, g2_norm = _extract_reciprocal_norms(result)
    nb = int(result.bands_per_valley) if result.bands_per_valley is not None else int(4 * result.lg * result.lg)
    flat_start = int(nb // 2 - 1)
    flat_stop = int(flat_start + 1)
    return {
        "theta_deg": float(result.theta_deg),
        "moire_lattice_constant_nm": moire_lattice_nm,
        "g1_abs_nm_inv": g1_norm,
        "g2_abs_nm_inv": g2_norm,
        "eps_BN": float(result.coulomb_params.epsilon_bn),
        "ds_angstrom": float(result.coulomb_params.ds_angstrom),
        "ds_nm": float(result.coulomb_params.ds_nm),
        "k_mesh_size": f"{int(result.lk)}x{int(result.lk)}",
        "G_cutoff_lg": int(result.lg),
        "Q_shell_cutoff_q_lg": int(result.q_lg),
        "Q_shift_count": int(result.q_shifts.shape[0]),
        "number_of_bands_per_valley_in_cRPA": nb,
        "flat_band_indices_retained_window": [flat_start, flat_stop],
        "remote_below_indices_retained_window": [0, flat_start - 1] if flat_start > 0 else [],
        "remote_above_indices_retained_window": [flat_stop + 1, nb - 1] if flat_stop + 1 < nb else [],
        "spin_degeneracy": float(result.metadata.get("spin_degeneracy", 2.0)),
        "spin_degeneracy_handling": str(result.metadata.get("spin_degeneracy_handling", "implicit factor 2")),
        "valley_degeneracy_handling": str(result.metadata.get("valley_degeneracy_handling", "explicit K/Kprime valley sum")),
        "eta_mev": float(result.eta_mev),
        "temperature_mev": float(result.metadata.get("temperature_mev", 0.0)),
        "fermi_level_mev": float(result.metadata.get("fermi_level_mev", 0.0)),
        "form_factor_mode": str(result.metadata.get("form_factor_mode", "")),
        "periodic_g_grid": bool(result.metadata.get("periodic_g_grid", False)),
        "sigma_rotation": bool(result.metadata.get("sigma_rotation", True)),
    }


def _value_at(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    if xs.size == 0:
        return float("nan")
    return float(np.interp(float(x), xs, ys))


def write_diagnostic_plots_and_report(
    result: CRPAResult,
    output_dir: Path | str,
    *,
    bin_width_nm_inv: float = 0.0125,
) -> CRPADiagnosticSummary:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_bn = float(result.coulomb_params.epsilon_bn)
    eps_crpa = np.asarray(result.effective_epsilon, dtype=float)
    eps_total = eps_bn * eps_crpa
    eps_diag = np.diagonal(np.asarray(result.dielectric_matrix, dtype=np.complex128), axis1=1, axis2=2)
    q_abs_nm_inv = np.abs(_nm_inv(result.physical_q_vectors, result.coulomb_params.graphene_lattice_angstrom)).real
    q_flat = q_abs_nm_inv.reshape(-1)
    eps_flat = eps_total.reshape(-1)
    shell = np.asarray([q_shell_index(row) for row in np.asarray(result.q_shifts, dtype=int)], dtype=int)
    shell_flat = np.broadcast_to(shell[None, :], result.effective_epsilon.shape).reshape(-1)

    xs, ys, counts = _representative_curve(
        q_abs_nm_inv,
        eps_total,
        x_max_nm_inv=1.2,
        bin_width_nm_inv=bin_width_nm_inv,
    )
    if ys.size:
        peak_index = int(np.argmax(ys))
        q_peak = float(xs[peak_index])
        eps_peak = float(ys[peak_index])
    else:
        q_peak = float("nan")
        eps_peak = float("nan")

    with (out / "epsilon_fig1e_window_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_abs_nm_inv", "eps_total_median", "n_points"])
        for x, y, n in zip(xs, ys, counts, strict=True):
            writer.writerow([f"{float(x):.16e}", f"{float(y):.16e}", str(int(n))])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig, ax = plt.subplots(figsize=(3.0, 2.55), constrained_layout=True)
    ax.plot(xs, ys, color="#0b8f22", lw=1.25, marker="o", ms=2.2, mec="#0b8f22", mfc="#0b8f22", mew=0.0)
    ax.set_xlim(0.0, 1.2)
    ax.set_ylim(2.0, 22.0)
    ax.set_xticks(np.arange(0.0, 1.2001, 0.2))
    ax.set_yticks([4, 8, 12, 16, 20])
    ax.set_xlabel(r"$|q|$ (nm$^{-1}$)", labelpad=2)
    ax.set_ylabel(r"$\epsilon(q)$", labelpad=2)
    ax.tick_params(top=True, right=True, length=3.0, pad=2)
    ax.minorticks_on()
    ax.text(-0.24, 0.99, "(e)", transform=ax.transAxes, ha="left", va="top", fontsize=12)
    fig.savefig(out / "epsilon_fig1e_window.png", dpi=450)
    fig.savefig(out / "epsilon_fig1e_window.pdf")
    plt.close(fig)

    finite = np.isfinite(q_flat) & np.isfinite(eps_flat)
    order = np.argsort(q_flat[finite], kind="stable")
    q_plot = q_flat[finite][order]
    eps_plot = eps_flat[finite][order]
    shell_plot = shell_flat[finite][order]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    scatter = ax.scatter(q_plot, eps_plot, c=shell_plot, s=7, cmap="viridis", linewidths=0.0, alpha=0.75)
    ax.axvline(1.2, color="black", lw=0.9, ls="--", alpha=0.6)
    ax.set_xlabel(r"$|\tilde{q}+Q|$ (nm$^{-1}$)")
    ax.set_ylabel(r"$\epsilon_{\rm total}=\epsilon_{\rm BN}\epsilon_{\rm cRPA}$")
    ax.set_title("cRPA dielectric diagonal by Q shell")
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Q shell index")
    ax.grid(alpha=0.2)
    fig.savefig(out / "epsilon_by_Q_shell.png", dpi=300)
    fig.savefig(out / "epsilon_by_Q_shell.pdf")
    plt.close(fig)

    diag_imag = float(np.max(np.abs(np.imag(eps_diag)))) if eps_diag.size else 0.0
    antiherm = 0.0
    for chi in np.asarray(result.chi0, dtype=np.complex128):
        antiherm = max(antiherm, float(np.max(np.abs(chi - chi.conjugate().T))))
    q_range = (q_flat >= 0.0) & (q_flat <= 1.2) & np.isfinite(q_flat) & np.isfinite(eps_flat)
    radial_std = 0.0
    if np.any(q_range):
        bins = np.arange(0.0, 1.2 + 0.025, 0.025)
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            in_bin = q_range & (q_flat >= lo) & (q_flat < hi)
            if np.count_nonzero(in_bin) > 1:
                radial_std = max(radial_std, float(np.std(eps_flat[in_bin])))

    summary = CRPADiagnosticSummary(
        q_peak_nm_inv=q_peak,
        eps_total_peak=eps_peak,
        eps_total_q0=_value_at(xs, ys, 0.0),
        eps_total_q04=_value_at(xs, ys, 0.4),
        eps_total_q08=_value_at(xs, ys, 0.8),
        eps_total_q12=_value_at(xs, ys, 1.2),
        eps_diag_imag_max_abs=diag_imag,
        chi0_antihermitian_max_abs=antiherm,
        radial_std_max_0_1p2=radial_std,
        full_q_abs_max_nm_inv=float(np.nanmax(q_flat)) if q_flat.size else float("nan"),
    )

    params = diagnostic_parameter_summary(result)
    (out / "crpa_epsilon_diagnostics_parameters.json").write_text(
        json.dumps(params, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# cRPA Epsilon Diagnostics",
        "",
        "## Parameters",
        "",
    ]
    for key, value in params.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Fig. 1(e) Window Checkpoints",
            "",
            f"- q_peak_nm_inv: {summary.q_peak_nm_inv:.12g}",
            f"- eps_total_peak: {summary.eps_total_peak:.12g}",
            f"- eps_total_q0: {summary.eps_total_q0:.12g}",
            f"- eps_total_q04: {summary.eps_total_q04:.12g}",
            f"- eps_total_q08: {summary.eps_total_q08:.12g}",
            f"- eps_total_q12: {summary.eps_total_q12:.12g}",
            "",
            "## Matrix Diagnostics",
            "",
            f"- eps_diag_imag_max_abs: {summary.eps_diag_imag_max_abs:.16e}",
            f"- chi0_antihermitian_max_abs: {summary.chi0_antihermitian_max_abs:.16e}",
            f"- radial_std_max_0_1p2: {summary.radial_std_max_0_1p2:.16e}",
            f"- full_q_abs_max_nm_inv: {summary.full_q_abs_max_nm_inv:.12g}",
            "",
            "## Files",
            "",
            "- crpa_epsilon_diagnostics.csv",
            "- crpa_epsilon_radial_bin_stats.csv",
            "- epsilon_fig1e_window.png",
            "- epsilon_by_Q_shell.png",
            "- epsilon_fig1e_window_curve.csv",
            "- crpa_epsilon_diagnostics_parameters.json",
        ]
    )
    (out / "crpa_epsilon_diagnostics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_all_epsilon_diagnostics(result: CRPAResult, output_dir: Path | str) -> CRPADiagnosticSummary:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_crpa_epsilon_diagnostics_csv(result, out / "crpa_epsilon_diagnostics.csv")
    write_radial_bin_stats(result, out / "crpa_epsilon_radial_bin_stats.csv", x_max_nm_inv=None)
    write_radial_bin_stats(result, out / "crpa_epsilon_radial_bin_stats_0_1p2.csv", x_max_nm_inv=1.2)
    return write_diagnostic_plots_and_report(result, out)
