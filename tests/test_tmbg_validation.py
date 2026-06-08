from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mean_field.core.lattice import KPath
from mean_field.systems.tmbg import (
    PathBandsResult,
    TMBGModel,
    TMBGParameters,
    TopologyResult,
    ValidationCheck,
    ValidationReport,
    infer_flat_band_indices,
    reproduce_paper_checkpoints,
    validate_physics,
)


def _check_by_name(report, name: str):
    return next(check for check in report.checks if check.name == name)


def test_tmbg_core_validation_passes_default_checks() -> None:
    model = TMBGModel.from_config(1.21, n_shells=1, params=TMBGParameters.full())
    report = validate_physics(model, n_bands=10)

    assert not report.has_failures
    assert _check_by_name(report, "C4.time_reversal").status == "pass"
    assert _check_by_name(report, "C10.c2zt_absent").status == "pass"
    assert _check_by_name(report, "C11.hamiltonian_cross_check").status == "pass"
    assert _check_by_name(report, "C4.k_to_kprime_node_exchange").status == "skipped"
    assert _check_by_name(report, "C9.cutoff_convergence").status == "skipped"


def test_tmbg_validation_can_enable_node_exchange_diagnostic() -> None:
    model = TMBGModel.from_config(1.21, n_shells=1, params=TMBGParameters.full())
    report = validate_physics(model, n_bands=10, include_node_exchange_check=True)
    check = _check_by_name(report, "C4.k_to_kprime_node_exchange")

    assert check.status in {"pass", "fail"}
    assert isinstance(check.value, float)
    assert check.value >= 0.0


def test_tmbg_validation_report_renders_markdown() -> None:
    model = TMBGModel.from_config(1.05, n_shells=1, params=TMBGParameters.minimal())
    report = validate_physics(model, n_bands=8, include_c3_check=False)
    text = report.to_markdown()

    assert "# tMBG Core Physics Validation" in text
    assert "C1.full_hamiltonian_hermitian" in text
    assert "failures: 0" in text


def test_validation_report_can_combine_multiple_reports() -> None:
    first = ValidationReport(
        title="first",
        checks=(ValidationCheck(name="A", status="pass", detail="a"),),
    )
    second = ValidationReport(
        title="second",
        checks=(ValidationCheck(name="B", status="fail", detail="b"),),
    )

    combined = ValidationReport.combine("combined", first, second)

    assert combined.title == "combined"
    assert [check.name for check in combined.checks] == ["A", "B"]
    assert combined.failure_count == 1


def test_infer_flat_band_indices_prefers_windowed_mid_pair() -> None:
    energies = np.asarray(
        [
            [-0.120, -0.090, -0.006, 0.005, 0.080, 0.130],
            [-0.121, -0.091, -0.005, 0.006, 0.081, 0.131],
            [-0.119, -0.089, -0.0065, 0.0045, 0.079, 0.129],
        ],
        dtype=float,
    )

    assert infer_flat_band_indices(energies) == (2, 3)


def test_cross_check_hamiltonian_matches_primary_builder() -> None:
    import mean_field.systems.tmbg.validation as validation_module

    model = TMBGModel.from_config(1.21, n_shells=1, params=TMBGParameters.full())
    diffs = validation_module.cross_check_hamiltonian(model, valley=1)

    assert diffs["G_vectors"] < 1.0e-12
    assert diffs["Gamma"] < 1.0e-12
    assert diffs["K"] < 1.0e-12
    assert diffs["M"] < 1.0e-12


def _fake_kpath() -> KPath:
    return KPath(
        kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.0j, 0.2 + 0.0j, 0.3 + 0.0j], dtype=np.complex128),
        kdist=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=float),
        labels=("K", "Gamma", "M", "Kprime"),
        node_indices=(1, 2, 3, 4),
    )


def _fake_energies(model_name: str, theta_deg: float, delta_ev: float, staggered_ev: float) -> np.ndarray:
    key = (model_name, round(theta_deg, 2), round(delta_ev, 3), round(staggered_ev, 3))
    lookup = {
        ("minimal", 1.07, 0.0, 0.0): np.asarray(
            [
                [-0.180, -0.055, -0.0010, 0.0010, 0.060, 0.170],
                [-0.181, -0.054, -0.0015, 0.0015, 0.061, 0.171],
                [-0.179, -0.056, -0.0008, 0.0008, 0.059, 0.169],
                [-0.180, -0.055, -0.0012, 0.0012, 0.060, 0.170],
            ],
            dtype=float,
        ),
        ("full", 1.21, 0.0, 0.0): np.asarray(
            [
                [-0.220, -0.080, -0.012, 0.006, 0.048, 0.150],
                [-0.218, -0.082, -0.010, 0.008, 0.050, 0.148],
                [-0.221, -0.079, -0.014, 0.005, 0.047, 0.151],
                [-0.219, -0.081, -0.011, 0.007, 0.049, 0.149],
            ],
            dtype=float,
        ),
        ("full", 1.21, 0.06, 0.0): np.asarray(
            [
                [-0.235, -0.090, -0.014, 0.009, 0.052, 0.165],
                [-0.232, -0.092, -0.011, 0.011, 0.054, 0.163],
                [-0.236, -0.089, -0.016, 0.008, 0.051, 0.166],
                [-0.233, -0.091, -0.012, 0.010, 0.053, 0.164],
            ],
            dtype=float,
        ),
        ("full", 1.21, -0.04, 0.0): np.asarray(
            [
                [-0.208, -0.074, -0.010, 0.005, 0.046, 0.145],
                [-0.205, -0.076, -0.008, 0.007, 0.048, 0.143],
                [-0.209, -0.073, -0.011, 0.004, 0.045, 0.146],
                [-0.206, -0.075, -0.009, 0.006, 0.047, 0.144],
            ],
            dtype=float,
        ),
        ("full", 1.21, -0.06, 0.0): np.asarray(
            [
                [-0.210, -0.078, -0.018, 0.004, 0.045, 0.146],
                [-0.207, -0.080, -0.007, 0.013, 0.047, 0.144],
                [-0.212, -0.077, -0.017, 0.005, 0.044, 0.147],
                [-0.208, -0.079, -0.008, 0.014, 0.046, 0.145],
            ],
            dtype=float,
        ),
        ("minimal", 1.21, 0.0, 0.0): np.asarray(
            [
                [-0.200, -0.070, -0.0025, 0.0018, 0.045, 0.140],
                [-0.199, -0.071, -0.0018, 0.0025, 0.046, 0.139],
                [-0.201, -0.069, -0.0022, 0.0015, 0.044, 0.141],
                [-0.200, -0.070, -0.0020, 0.0020, 0.045, 0.140],
            ],
            dtype=float,
        ),
        ("minimal", 1.21, 0.06, 0.0): np.asarray(
            [
                [-0.205, -0.073, -0.0031, 0.0016, 0.046, 0.142],
                [-0.204, -0.074, -0.0022, 0.0025, 0.047, 0.141],
                [-0.206, -0.072, -0.0028, 0.0013, 0.045, 0.143],
                [-0.205, -0.073, -0.0025, 0.0021, 0.046, 0.142],
            ],
            dtype=float,
        ),
        ("minimal", 1.21, -0.06, 0.0): np.asarray(
            [
                [-0.204, -0.072, -0.0030, 0.0015, 0.046, 0.142],
                [-0.203, -0.073, -0.0021, 0.0024, 0.047, 0.141],
                [-0.205, -0.071, -0.0027, 0.0012, 0.045, 0.143],
                [-0.204, -0.072, -0.0024, 0.0020, 0.046, 0.142],
            ],
            dtype=float,
        ),
        ("full", 1.21, 0.0, 0.01): np.asarray(
            [
                [-0.221, -0.081, -0.013, 0.008, 0.049, 0.151],
                [-0.219, -0.083, -0.011, 0.010, 0.051, 0.149],
                [-0.222, -0.080, -0.014, 0.007, 0.048, 0.152],
                [-0.220, -0.082, -0.012, 0.009, 0.050, 0.150],
            ],
            dtype=float,
        ),
        ("full", 1.21, 0.0, -0.01): np.asarray(
            [
                [-0.219, -0.079, -0.012, 0.007, 0.048, 0.150],
                [-0.217, -0.081, -0.010, 0.009, 0.050, 0.148],
                [-0.220, -0.078, -0.013, 0.006, 0.047, 0.151],
                [-0.218, -0.080, -0.011, 0.008, 0.049, 0.149],
            ],
            dtype=float,
        ),
    }
    return lookup[key]


def _fake_chern(model_name: str, theta_deg: float, delta_ev: float, staggered_ev: float, band_index: int, valley: int) -> int:
    key = (model_name, round(theta_deg, 2), round(delta_ev, 3), round(staggered_ev, 3), int(band_index))
    lookup = {
        ("full", 1.21, 0.0, 0.0, 2): 2,
        ("full", 1.21, 0.0, 0.0, 3): -3,
        ("full", 1.21, 0.06, 0.0, 2): -2,
        ("full", 1.21, 0.06, 0.0, 3): 1,
        ("full", 1.21, -0.04, 0.0, 2): 1,
        ("full", 1.21, -0.04, 0.0, 3): -2,
        ("full", 1.21, 0.0, 0.01, 2): 2,
        ("full", 1.21, 0.0, 0.01, 3): -1,
        ("full", 1.21, 0.0, -0.01, 2): 1,
        ("full", 1.21, 0.0, -0.01, 3): -2,
    }
    return valley * lookup.get(key, 0)


def test_reproduce_paper_checkpoints_orchestrates_cases_without_running_real_solver(monkeypatch, tmp_path: Path) -> None:
    import mean_field.systems.tmbg.validation as validation_module

    class FakeModel:
        def __init__(self, case) -> None:
            self._case = case
            self.lattice = SimpleNamespace(
                matrix_dim=6,
                g_vectors=np.asarray([0.0 + 0.0j], dtype=np.complex128),
                k_m=0.0 + 0.0j,
            )
            self.theta_deg = case.theta_deg

        def lattice_summary(self):
            return {
                "theta_deg": self._case.theta_deg,
                "n_shells": 1,
                "N_G": 1,
                "matrix_dim": 6,
            }

        def bands_along_standard_path(self, *, points_per_segment: int, n_bands: int | None = None):
            del points_per_segment, n_bands
            return PathBandsResult(
                path=_fake_kpath(),
                energies=_fake_energies(
                    self._case.model_name,
                    self._case.theta_deg,
                    self._case.interlayer_potential,
                    self._case.staggered_potential,
                ),
            )

        def topology_on_grid(self, mesh_size: int, band_indices: int | tuple[int, ...], *, valley: int = 1, n_bands: int | None = None):
            del mesh_size, n_bands
            if isinstance(band_indices, tuple):
                raise AssertionError("checkpoint runner should request one band at a time")
            chern = _fake_chern(
                self._case.model_name,
                self._case.theta_deg,
                self._case.interlayer_potential,
                self._case.staggered_potential,
                int(band_indices),
                int(valley),
            )
            return TopologyResult(
                band_indices=(int(band_indices),),
                valley=int(valley),
                k_grid_frac=np.zeros((4, 4, 2), dtype=float),
                berry_curvature=np.zeros((4, 4), dtype=float),
                chern_number=float(chern),
                rounded_chern_number=int(chern),
            )

        def bands_on_grid(self, mesh_size: int, *, valley: int = 1, n_bands: int | None = None, return_eigenvectors: bool = False):
            del valley, return_eigenvectors
            resolved_n_bands = 6 if n_bands is None else int(n_bands)
            base = _fake_energies(
                self._case.model_name,
                self._case.theta_deg,
                self._case.interlayer_potential,
                self._case.staggered_potential,
            )[0, :resolved_n_bands]
            energies = np.broadcast_to(base, (mesh_size, mesh_size, resolved_n_bands)).copy()
            return SimpleNamespace(
                k_grid_frac=np.zeros((mesh_size, mesh_size, 2), dtype=float),
                kvec=np.zeros((mesh_size, mesh_size), dtype=np.complex128),
                energies=energies,
            )

        def diagonalize(self, k_tilde: complex, *, valley: int = 1, n_bands: int | None = None):
            del k_tilde, valley
            if (
                self._case.model_name == "full"
                and round(self._case.theta_deg, 2) == 1.21
                and round(self._case.interlayer_potential, 3) == 0.0
                and round(self._case.staggered_potential, 3) == 0.0
            ):
                evals = np.asarray([-0.220, -0.080, -0.0004, 0.0004, 0.048, 0.150], dtype=float)
            else:
                evals = _fake_energies(
                    self._case.model_name,
                    self._case.theta_deg,
                    self._case.interlayer_potential,
                    self._case.staggered_potential,
                )[0]
            if n_bands is not None:
                evals = evals[:n_bands]
            evecs = np.eye(self.lattice.matrix_dim, dtype=np.complex128)[:, : evals.size]
            return evals, evecs

    monkeypatch.setattr(
        validation_module,
        "_build_checkpoint_model",
        lambda case, *, n_shells: FakeModel(case),
    )

    plotted = {}

    def fake_write_tmbg_paper_band_figure(output_dir, panels, **kwargs):
        plotted["ylim"] = kwargs.get("ylim")
        plotted["labels"] = [panel.label for panel in panels]
        png_path = Path(output_dir) / "fig2_like_bands.png"
        pdf_path = Path(output_dir) / "fig2_like_bands.pdf"
        png_path.write_text("png", encoding="utf-8")
        pdf_path.write_text("pdf", encoding="utf-8")
        return {"paper_band_plot_png": png_path, "paper_band_plot_pdf": pdf_path}

    monkeypatch.setattr(validation_module, "write_tmbg_paper_band_figure", fake_write_tmbg_paper_band_figure)
    monkeypatch.setattr(
        validation_module,
        "write_tmbg_lattice_plot",
        lambda output_dir, lattice, **kwargs: {
            "lattice_plot_png": Path(output_dir) / "lattice_plot.png",
            "lattice_plot_pdf": Path(output_dir) / "lattice_plot.pdf",
        },
    )
    monkeypatch.setattr(
        validation_module,
        "validate_physics",
        lambda *args, **kwargs: ValidationReport(
            title="core",
            checks=(ValidationCheck(name="C1.full_hamiltonian_hermitian", status="pass", detail="ok"),),
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "diagnose_ktilde_symmetry",
        lambda *args, **kwargs: ValidationReport(
            title="ktilde",
            checks=(ValidationCheck(name="D1.chiral_limit_ktilde_touching", status="pass", detail="ok"),),
        ),
    )

    report = reproduce_paper_checkpoints(
        n_shells=1,
        points_per_segment=4,
        topology_mesh_size=4,
        output_dir=tmp_path,
        verify_opposite_valley=True,
    )

    assert not report.has_failures
    assert _check_by_name(report, "CP1.minimal_magic_angle_bandwidth").status == "pass"
    assert _check_by_name(report, "CP2b.delta_0_band_touching").status == "pass"
    assert _check_by_name(report, "CP3.delta_0_valley_chern").status == "pass"
    assert _check_by_name(report, "CP6.staggered_potential_suppresses_abs3").status == "pass"
    assert plotted["labels"] == ["Δ = 0 meV", "Δ = +60 meV", "Δ = -40 meV"]
    assert plotted["ylim"] == (-0.100, 0.100)
    assert (tmp_path / "paper_checkpoint_report.md").exists()
    assert (tmp_path / "validation_report.md").exists()
    assert (tmp_path / "lattice_info.json").exists()
    assert (tmp_path / "run.log").exists()
    assert (tmp_path / "delta_+000mev" / "bands_path.npz").exists()
    assert (tmp_path / "delta_+000mev" / "bands_grid.npz").exists()
    assert (tmp_path / "delta_+000mev" / "chern_numbers.json").exists()
    assert (tmp_path / "delta_+000mev" / "berry_curvature.npz").exists()
