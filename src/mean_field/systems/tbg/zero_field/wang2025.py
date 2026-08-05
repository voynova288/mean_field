"""Fail-closed preflight for Wang et al. strained-IKS collective modes.

This module records the paper prescription in arXiv:2509.12320v1.  It does
not implement HF or TDHF.  In particular, paper facts, raster-only labels,
later author-family static-HF conventions, and executable-provider receipts
remain separate.  Binding all metadata receipts is therefore not a claim of
executable or paper-reproduction readiness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from numbers import Integral, Real
from typing import Literal, Protocol, runtime_checkable

from mean_field.core.hf.tdhf_signed import TDHFSignedQ, classify_tdhf_signed_q

WANG_2025_ARXIV = "2509.12320v1"
WANG_2025_RESPONSE_SCOPE = "wang_2025_strained_iks_nu2_nu3_tdhf_preflight_v1"
WANG_2025_PDF_SHA256 = (
    "1e648ad06731c46815d3216ff2da9f3dbedd65dd61ab1c237cd944de06a01d94"
)
WANG_2025_SOURCE_ARCHIVE_SHA256 = (
    "85b647bdc9d194f78b1b0072f952cd9468f3dae8c900ecbf61fa74b1578e01f3"
)
WANG_2025_MAIN_TEX_SHA256 = (
    "b029d09487439697c09b2db3cd69b1bcfef4e500d2ed9d17c3f5ca27863bc40c"
)
WANG_2025_FIG6_RASTER_SHA256 = (
    "0e7cfa13df01c74155e8f3ad0e988fd3322906b3e01bdf7cc94dd4e8d1f9607f"
)
WANG_2025_FIELD_RASTER_SHA256 = (
    "7d999dfd250adfccf907dcbc5502d0088104f3555eb82396c332da3ef3f6bd0f"
)
WANG_2025_LARGE_GRID_RASTER_SHA256 = (
    "22424abb08c9240c2439c3997c92df59f96a36178908532e73ceecc2885ddbc7"
)

ReceiptAuthority = Literal["reproduction_choice", "independent_provider_explicit"]
Scenario = Literal[
    "unperturbed",
    "epc",
    "intervalley_coulomb",
    "zeeman",
    "ising_soc",
]
ImplementationRole = Literal[
    "strain_beta",
    "q0_policy",
    "exact_p_ref",
    "iks_scan_and_scf_policy",
    "exact_tdhf_q_list_and_tolerances",
    "scalar_energy_functional",
    "scf_hamiltonian_derivative",
    "finite_q_hessian_derivative",
    "shared_functional_source",
    "tdhf_implementation_provider",
]


def _strict_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


def _commit(value: str, label: str) -> None:
    if len(value) not in (40, 64) or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase 40- or 64-character commit")


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_exact(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Wang 2025 paper-direct {label} was changed")


@dataclass(frozen=True, slots=True)
class WangPrimarySourceReceipt:
    """Immutable local/official primary-source identities."""

    arxiv: str = WANG_2025_ARXIV
    source_url: str = "https://arxiv.org/src/2509.12320v1"
    source_archive_size_bytes: int = 6_524_220
    source_archive_sha256: str = WANG_2025_SOURCE_ARCHIVE_SHA256
    main_tex_path: str = "main.tex"
    main_tex_sha256: str = WANG_2025_MAIN_TEX_SHA256
    local_pdf_evidence_path: str = "reference/2509.12320v1.pdf"
    local_pdf_sha256: str = WANG_2025_PDF_SHA256

    def __post_init__(self) -> None:
        expected = (
            WANG_2025_ARXIV,
            "https://arxiv.org/src/2509.12320v1",
            6_524_220,
            WANG_2025_SOURCE_ARCHIVE_SHA256,
            "main.tex",
            WANG_2025_MAIN_TEX_SHA256,
            "reference/2509.12320v1.pdf",
            WANG_2025_PDF_SHA256,
        )
        actual = (
            self.arxiv,
            self.source_url,
            self.source_archive_size_bytes,
            self.source_archive_sha256,
            self.main_tex_path,
            self.main_tex_sha256,
            self.local_pdf_evidence_path,
            self.local_pdf_sha256,
        )
        _require_exact(actual, expected, "primary-source receipt")
        if not _strict_integer(self.source_archive_size_bytes):
            raise TypeError("Wang source archive size must be an integer")
        for label, value in (
            ("source archive", self.source_archive_sha256),
            ("main.tex", self.main_tex_sha256),
            ("local PDF", self.local_pdf_sha256),
        ):
            _sha256(value, f"Wang {label}")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class WangRasterOnlyTargets:
    """Labels visible in source PNGs but absent from the corresponding captions."""

    main_figure_path: str = "figures/plot_coll.png"
    main_figure_sha256: str = WANG_2025_FIG6_RASTER_SHA256
    epc_g_mev_nm2: float = 50.0
    intervalley_coulomb_mev_nm2: float = 100.0
    field_figure_path: str = "figures/plot_coll_field.png"
    field_figure_sha256: str = WANG_2025_FIELD_RASTER_SHA256
    zeeman_mev: float = 0.4
    ising_soc_mev: float = 0.4
    large_grid_figure_path: str = "figures/plot_coll_large.png"
    large_grid_figure_sha256: str = WANG_2025_LARGE_GRID_RASTER_SHA256
    large_grid_epc_g_mev_nm2: float = 100.0
    main_path_q1_extent: tuple[float, float] = (-0.5, 0.5)
    main_path_q2: float = 0.0
    qualifier: str = "paper_source_raster_label_only_not_caption_text"

    def __post_init__(self) -> None:
        for label, value in (
            ("Fig. 6 EPC strength", self.epc_g_mev_nm2),
            ("Fig. 6 intervalley strength", self.intervalley_coulomb_mev_nm2),
            ("field Zeeman strength", self.zeeman_mev),
            ("field SOC strength", self.ising_soc_mev),
            ("large-grid EPC strength", self.large_grid_epc_g_mev_nm2),
        ):
            _finite_real(value, f"Wang raster-only {label}")
        expected = (
            "figures/plot_coll.png",
            WANG_2025_FIG6_RASTER_SHA256,
            50.0,
            100.0,
            "figures/plot_coll_field.png",
            WANG_2025_FIELD_RASTER_SHA256,
            0.4,
            0.4,
            "figures/plot_coll_large.png",
            WANG_2025_LARGE_GRID_RASTER_SHA256,
            100.0,
            (-0.5, 0.5),
            0.0,
            "paper_source_raster_label_only_not_caption_text",
        )
        actual = (
            self.main_figure_path,
            self.main_figure_sha256,
            float(self.epc_g_mev_nm2),
            float(self.intervalley_coulomb_mev_nm2),
            self.field_figure_path,
            self.field_figure_sha256,
            float(self.zeeman_mev),
            float(self.ising_soc_mev),
            self.large_grid_figure_path,
            self.large_grid_figure_sha256,
            float(self.large_grid_epc_g_mev_nm2),
            tuple(float(value) for value in self.main_path_q1_extent),
            float(self.main_path_q2),
            self.qualifier,
        )
        _require_exact(actual, expected, "raster-only target labels")
        for label, value in (
            ("Fig. 6 raster", self.main_figure_sha256),
            ("field raster", self.field_figure_sha256),
            ("large-grid raster", self.large_grid_figure_sha256),
        ):
            _sha256(value, f"Wang {label}")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class WangQualifiedModeScale:
    label: Literal[
        "additional_gapped_collective_modes",
        "figure_hf_charge_gap",
        "explicitly_broken_former_ngms",
    ]
    value_mev: float | None
    qualifier: Literal[
        "approximately",
        "approximately_figure_scale_actual_likely_much_smaller",
        "small_unquantified",
    ]
    source: str
    acceptance_use: str = "context_only_not_an_exact_threshold"

    def __post_init__(self) -> None:
        if self.value_mev is not None:
            object.__setattr__(
                self, "value_mev", _finite_real(self.value_mev, "Wang mode scale")
            )
        if not self.source.strip():
            raise ValueError("Wang mode scale requires source provenance")
        if self.acceptance_use != "context_only_not_an_exact_threshold":
            raise ValueError("Wang approximate scales cannot become exact thresholds")


@dataclass(frozen=True, slots=True)
class WangModeScales:
    additional_gapped_modes: WangQualifiedModeScale = field(
        default_factory=lambda: WangQualifiedModeScale(
            "additional_gapped_collective_modes",
            15.0,
            "approximately",
            "main.tex lines 477-478: modes at about E approximately 15 meV",
        )
    )
    figure_hf_charge_gap: WangQualifiedModeScale = field(
        default_factory=lambda: WangQualifiedModeScale(
            "figure_hf_charge_gap",
            20.0,
            "approximately_figure_scale_actual_likely_much_smaller",
            "main.tex lines 477-478: approximately 20 meV in Fig. 6; remote bands/beyond-HF lower it",
        )
    )
    former_ngm_gaps: WangQualifiedModeScale = field(
        default_factory=lambda: WangQualifiedModeScale(
            "explicitly_broken_former_ngms",
            None,
            "small_unquantified",
            "main.tex line 470: EPC gives small gaps to three former nu=2 NGMs",
        )
    )

    def __post_init__(self) -> None:
        signature = tuple(
            (item.label, item.value_mev, item.qualifier, item.acceptance_use)
            for item in (
                self.additional_gapped_modes,
                self.figure_hf_charge_gap,
                self.former_ngm_gaps,
            )
        )
        expected = (
            (
                "additional_gapped_collective_modes",
                15.0,
                "approximately",
                "context_only_not_an_exact_threshold",
            ),
            (
                "figure_hf_charge_gap",
                20.0,
                "approximately_figure_scale_actual_likely_much_smaller",
                "context_only_not_an_exact_threshold",
            ),
            (
                "explicitly_broken_former_ngms",
                None,
                "small_unquantified",
                "context_only_not_an_exact_threshold",
            ),
        )
        _require_exact(signature, expected, "qualified mode scales")


_MODE_COUNTS: dict[tuple[int, Scenario], tuple[int, int, str, bool]] = {
    (2, "unperturbed"): (4, 0, "SU2K_x_SU2Kp_IKS_manifold", False),
    (2, "epc"): (1, 0, "spin_singlet_charge_IKS", False),
    (2, "intervalley_coulomb"): (3, 0, "pi_relative_phase_spin_IKS", False),
    (2, "zeeman"): (2, 0, "zeeman_selected_IKS", True),
    (2, "ising_soc"): (2, 0, "ising_soc_selected_IKS", True),
    (3, "unperturbed"): (1, 2, "fully_ferromagnetic_IKS_representative", False),
    (3, "epc"): (3, 0, "valley_antiferromagnetic_IKS", False),
    (3, "intervalley_coulomb"): (1, 1, "valley_ferromagnetic_IKS", False),
    (3, "zeeman"): (1, 0, "field_aligned_valley_ferromagnetic_IKS", False),
    (3, "ising_soc"): (1, 0, "soc_aligned_valley_antiferromagnetic_IKS", False),
}


@dataclass(frozen=True, slots=True)
class WangScenarioModes:
    filling: int
    scenario: Scenario
    linear_goldstones: int
    quadratic_goldstones: int
    ground_state_label: str
    infinitesimal_perturbation_only: bool = False

    def __post_init__(self) -> None:
        if not _strict_integer(self.filling):
            raise TypeError("Wang mode filling must be an integer")
        if self.filling not in (2, 3):
            raise ValueError("Wang benchmark mode filling must be nu=2 or nu=3")
        if (self.filling, self.scenario) not in _MODE_COUNTS:
            raise ValueError("unknown Wang mode-count scenario")
        expected = _MODE_COUNTS[(self.filling, self.scenario)]
        actual = (
            self.linear_goldstones,
            self.quadratic_goldstones,
            self.ground_state_label,
            self.infinitesimal_perturbation_only,
        )
        if any(
            not _strict_integer(value) or int(value) < 0
            for value in (self.linear_goldstones, self.quadratic_goldstones)
        ):
            raise TypeError("Wang linear/quadratic counts must be non-negative integers")
        if actual != expected:
            raise ValueError("Wang paper linear/quadratic mode inventory was changed")

    @property
    def total_goldstone_branches(self) -> int:
        return self.linear_goldstones + self.quadratic_goldstones


def _paper_mode_tuple() -> tuple[WangScenarioModes, ...]:
    order: tuple[Scenario, ...] = (
        "unperturbed",
        "epc",
        "intervalley_coulomb",
        "zeeman",
        "ising_soc",
    )
    return tuple(
        WangScenarioModes(filling, scenario, *_MODE_COUNTS[(filling, scenario)])
        for filling in (2, 3)
        for scenario in order
    )


@dataclass(frozen=True, slots=True)
class WangModeInventory:
    scenarios: tuple[WangScenarioModes, ...] = field(default_factory=_paper_mode_tuple)
    total_soft_modes_nu2: int = 12
    total_soft_modes_nu3: int = 7
    additional_gapped_modes_nu2: int = 8
    additional_gapped_modes_nu3: int = 4

    def __post_init__(self) -> None:
        if self.scenarios != _paper_mode_tuple():
            raise ValueError("Wang paper scenario mode inventory was changed")
        totals = (
            self.total_soft_modes_nu2,
            self.total_soft_modes_nu3,
            self.additional_gapped_modes_nu2,
            self.additional_gapped_modes_nu3,
        )
        if any(not _strict_integer(value) for value in totals):
            raise TypeError("Wang soft-mode totals must use strict integers")
        _require_exact(
            (
                self.total_soft_modes_nu2,
                self.total_soft_modes_nu3,
                self.additional_gapped_modes_nu2,
                self.additional_gapped_modes_nu3,
            ),
            (12, 7, 8, 4),
            "soft-mode totals",
        )
        if self.total_soft_modes_nu2 != 16 - 2**2:
            raise ValueError("Wang nu=2 soft-mode count does not close")
        if self.total_soft_modes_nu3 != 16 - 3**2:
            raise ValueError("Wang nu=3 soft-mode count does not close")

    def for_scenario(self, filling: int, scenario: Scenario) -> WangScenarioModes:
        for item in self.scenarios:
            if item.filling == filling and item.scenario == scenario:
                return item
        raise KeyError((filling, scenario))


@dataclass(frozen=True, slots=True)
class WangPaperPrescription:
    """Paper facts only; none of these fields identify executable code."""

    fillings: tuple[int, int] = (2, 3)
    theta_deg: float = 1.05
    w_aa_mev: float = 80.0
    w_ab_mev: float = 110.0
    dirac_velocity_m_per_s: float = 8.8e5
    hopping_model: str = "local_three_matrix_bm_nonlocal_tunnelling_excluded"
    strain_kind: str = "uniaxial_heterostrain_opposite_half_strain_per_layer"
    strain_percent: float = 0.3
    strain_angle_deg: float = 0.0
    poisson_ratio: float = 0.16
    layer_twist_policy: str = "theta1_plus_half_theta_theta2_minus_half_theta"
    coulomb_kernel: str = "e2_tanh_qd_over_2_epsilon0_epsilonr_q"
    gate_distance_nm: float = 25.0
    epsilon_r: float = 10.0
    subtraction_name: str = "average"
    subtraction_semantics: str = "half_filling_of_central_bands_exact_matrix_unpublished"
    active_bands_per_spin_valley: int = 2
    active_dimension_per_k: int = 8
    tdhf_mesh_shape: tuple[int, int] = (10, 10)
    field_tdhf_mesh_shape: tuple[int, int] = (10, 10)
    full_bz_tdhf_mesh_shape: tuple[int, int] = (10, 10)
    linear_quadratic_confirmation_mesh_shape: tuple[int, int] = (20, 20)
    iks_frame: str = "k_tilde_equals_k_plus_tau_qIKS_over_2"
    q_iks_fractional_G1_G2: tuple[float, float] = (0.5, 0.0)
    tdhf_q_role: str = "independent_transfer_q_not_ground_state_qIKS"
    tdhf_path_statement: str = "along_G1_with_q2_zero"
    phonon_irreps: tuple[str, str] = ("A1", "B1")
    epc_vertices: tuple[str, str] = ("tau_x_sigma_x", "tau_y_sigma_x")
    epc_treatment: str = "Schrieffer_Wolff_effective_intervalley_interaction"
    phonon_energy_mev_qualifier: tuple[float, str] = (160.0, "approximately")
    epc_typical_g_mev_nm2_qualifier: tuple[float, str] = (70.0, "around")
    intervalley_coulomb_treatment: str = (
        "V_q_plus_tau_DeltaK_approximated_by_tunable_V_inter"
    )
    intervalley_coulomb_estimates_mev_nm2: tuple[float, float] = (50.0, 120.0)
    modes: WangModeInventory = field(default_factory=WangModeInventory)
    mode_scales: WangModeScales = field(default_factory=WangModeScales)
    raster_only_targets: WangRasterOnlyTargets = field(
        default_factory=WangRasterOnlyTargets
    )

    def __post_init__(self) -> None:
        for label, value in (
            ("theta", self.theta_deg),
            ("wAA", self.w_aa_mev),
            ("wAB", self.w_ab_mev),
            ("Dirac velocity", self.dirac_velocity_m_per_s),
            ("strain", self.strain_percent),
            ("strain angle", self.strain_angle_deg),
            ("Poisson ratio", self.poisson_ratio),
            ("gate distance", self.gate_distance_nm),
            ("relative dielectric constant", self.epsilon_r),
        ):
            _finite_real(value, f"Wang paper {label}")
        if not _strict_integer(self.active_bands_per_spin_valley) or not _strict_integer(
            self.active_dimension_per_k
        ):
            raise TypeError("Wang active-band counts must use strict integers")
        if len(self.q_iks_fractional_G1_G2) != 2:
            raise ValueError("Wang qIKS must have two reciprocal-basis coordinates")
        for value in self.q_iks_fractional_G1_G2:
            _finite_real(value, "Wang qIKS coordinate")
        direct = (
            self.fillings,
            float(self.theta_deg),
            float(self.w_aa_mev),
            float(self.w_ab_mev),
            float(self.dirac_velocity_m_per_s),
            self.hopping_model,
            self.strain_kind,
            float(self.strain_percent),
            float(self.strain_angle_deg),
            float(self.poisson_ratio),
            self.layer_twist_policy,
            self.coulomb_kernel,
            float(self.gate_distance_nm),
            float(self.epsilon_r),
            self.subtraction_name,
            self.subtraction_semantics,
            self.active_bands_per_spin_valley,
            self.active_dimension_per_k,
            self.tdhf_mesh_shape,
            self.field_tdhf_mesh_shape,
            self.full_bz_tdhf_mesh_shape,
            self.linear_quadratic_confirmation_mesh_shape,
            self.iks_frame,
            tuple(float(x) for x in self.q_iks_fractional_G1_G2),
            self.tdhf_q_role,
            self.tdhf_path_statement,
            self.phonon_irreps,
            self.epc_vertices,
            self.epc_treatment,
            self.phonon_energy_mev_qualifier,
            self.epc_typical_g_mev_nm2_qualifier,
            self.intervalley_coulomb_treatment,
            self.intervalley_coulomb_estimates_mev_nm2,
        )
        expected = (
            (2, 3),
            1.05,
            80.0,
            110.0,
            8.8e5,
            "local_three_matrix_bm_nonlocal_tunnelling_excluded",
            "uniaxial_heterostrain_opposite_half_strain_per_layer",
            0.3,
            0.0,
            0.16,
            "theta1_plus_half_theta_theta2_minus_half_theta",
            "e2_tanh_qd_over_2_epsilon0_epsilonr_q",
            25.0,
            10.0,
            "average",
            "half_filling_of_central_bands_exact_matrix_unpublished",
            2,
            8,
            (10, 10),
            (10, 10),
            (10, 10),
            (20, 20),
            "k_tilde_equals_k_plus_tau_qIKS_over_2",
            (0.5, 0.0),
            "independent_transfer_q_not_ground_state_qIKS",
            "along_G1_with_q2_zero",
            ("A1", "B1"),
            ("tau_x_sigma_x", "tau_y_sigma_x"),
            "Schrieffer_Wolff_effective_intervalley_interaction",
            (160.0, "approximately"),
            (70.0, "around"),
            "V_q_plus_tau_DeltaK_approximated_by_tunable_V_inter",
            (50.0, 120.0),
        )
        _require_exact(direct, expected, "BM/strain/Coulomb/active/mesh/qIKS prescription")
        mesh_dimensions = (
            self.tdhf_mesh_shape
            + self.field_tdhf_mesh_shape
            + self.full_bz_tdhf_mesh_shape
            + self.linear_quadratic_confirmation_mesh_shape
        )
        if any(not _strict_integer(x) for x in self.fillings + mesh_dimensions):
            raise TypeError("Wang fillings and mesh shape must use strict integers")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


_AUTHOR_FILE_HASHES = (
    ("HF_input.json", "d577afffdf80a05a348394c5b813540b8074107fc765454f6e50066d420e25e8"),
    ("constants.py", "8d25bcccd54e41207788ff4a9e1b934a50347fabdad46ba44408fc535573ec62"),
    ("int_input.json", "c143c294ad95cf94d91cfbabd0437556e5c2a342850d54484c9b47caaf84b4de"),
    ("mainProgram.py", "258c97e57164055de3273ba4471cd96be709c1f159e19f73481750c801aed401"),
    ("projectors.py", "d7c7138ddf2107a71c24194ac70790bd27cdc05297ee9cdc997c1dc3882e5ede"),
    ("routines.py", "507e8b9e799f494777d354c9d7d481dd19d6ba42894d393630dd79ef16d02108"),
    ("singleParticle.py", "a050fa545c4d399b227a178bcc4705a110bd7962edcb9e1f69e300b5e1a3e43b"),
)


@dataclass(frozen=True, slots=True)
class WangAuthorFamilyTBGHFReceipt:
    """Later static-HF family evidence, explicitly not target/TDHF authority."""

    repository: str = "https://github.com/ziweiwang-code/TBG-HF.git"
    companion_arxiv: str = "2511.21683"
    commit: str = "0d2a3d742aa901fa45ce46690c1385887165f58c"
    tree: str = "a1588c522474c5a8616c98e5cde49ebcfbaac9d2"
    tag: str = "v1.0.0"
    tag_commit: str = "b180f8f2b627d8a80b74b61a99d96b2cd56a76db"
    archive_url: str = "https://zenodo.org/api/records/17701732/files/ziweiwang-code/TBG-HF-v1.0.0.zip/content"
    archive_sha256: str = "36b7e5d798adf5f158fc1b408d10d60eadaaa5fcfb0739ec9a3e24c164ca57da"
    file_sha256: tuple[tuple[str, str], ...] = _AUTHOR_FILE_HASHES
    beta: float = 3.14
    parent_envelope_ng: tuple[int, int] = (4, 4)
    transfer_envelope_ng: tuple[int, int] = (5, 5)
    include_q0: bool = True
    reference_policy: str = "average_central_half_identity_on_central_bands"
    grid_registration: str = "Gamma_registered_fractional_torus_no_half_shift"
    theta_deg: float = 1.08
    w_aa_mev: float = 70.0
    w_ab_mev: float = 110.0
    has_epc: bool = False
    has_intervalley_coulomb: bool = False
    has_tdhf: bool = False
    target_parameter_match: bool = False
    target_run_authority: bool = False
    tdhf_authority: bool = False

    def __post_init__(self) -> None:
        values = (
            self.repository,
            self.companion_arxiv,
            self.commit,
            self.tree,
            self.tag,
            self.tag_commit,
            self.archive_url,
            self.archive_sha256,
            self.file_sha256,
            float(self.beta),
            self.parent_envelope_ng,
            self.transfer_envelope_ng,
            self.include_q0,
            self.reference_policy,
            self.grid_registration,
            float(self.theta_deg),
            float(self.w_aa_mev),
            float(self.w_ab_mev),
            self.has_epc,
            self.has_intervalley_coulomb,
            self.has_tdhf,
            self.target_parameter_match,
            self.target_run_authority,
            self.tdhf_authority,
        )
        fixed = (
            "https://github.com/ziweiwang-code/TBG-HF.git",
            "2511.21683",
            "0d2a3d742aa901fa45ce46690c1385887165f58c",
            "a1588c522474c5a8616c98e5cde49ebcfbaac9d2",
            "v1.0.0",
            "b180f8f2b627d8a80b74b61a99d96b2cd56a76db",
            "https://zenodo.org/api/records/17701732/files/ziweiwang-code/TBG-HF-v1.0.0.zip/content",
            "36b7e5d798adf5f158fc1b408d10d60eadaaa5fcfb0739ec9a3e24c164ca57da",
            _AUTHOR_FILE_HASHES,
            3.14,
            (4, 4),
            (5, 5),
            True,
            "average_central_half_identity_on_central_bands",
            "Gamma_registered_fractional_torus_no_half_shift",
            1.08,
            70.0,
            110.0,
            False,
            False,
            False,
            False,
            False,
            False,
        )
        for label, value in (
            ("beta", self.beta),
            ("theta", self.theta_deg),
            ("wAA", self.w_aa_mev),
            ("wAB", self.w_ab_mev),
        ):
            _finite_real(value, f"author-family {label}")
        if any(
            not _strict_integer(item)
            for item in self.parent_envelope_ng + self.transfer_envelope_ng
        ):
            raise TypeError("author-family Ng/NG values must use strict integers")
        if values != fixed:
            raise ValueError("later author-family TBG-HF receipt was changed")
        _commit(self.commit, "author-family commit")
        _commit(self.tree, "author-family tree")
        _commit(self.tag_commit, "author-family tag commit")
        _sha256(self.archive_sha256, "author-family archive")
        for path, digest in self.file_sha256:
            if not path.strip():
                raise ValueError("author-family file receipt requires a path")
            _sha256(digest, f"author-family file {path}")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class WangImplementationReceipt:
    role: ImplementationRole
    value: int | float | str
    provider_fingerprint: str
    source_artifact_sha256: str
    authority_kind: ReceiptAuthority
    source: str

    def __post_init__(self) -> None:
        if self.role not in (
            "strain_beta",
            "q0_policy",
            "exact_p_ref",
            "iks_scan_and_scf_policy",
            "exact_tdhf_q_list_and_tolerances",
            "scalar_energy_functional",
            "scf_hamiltonian_derivative",
            "finite_q_hessian_derivative",
            "shared_functional_source",
            "tdhf_implementation_provider",
        ):
            raise ValueError("unknown Wang implementation receipt role")
        if self.authority_kind not in (
            "reproduction_choice",
            "independent_provider_explicit",
        ):
            raise ValueError("invalid Wang implementation receipt authority")
        if isinstance(self.value, bool):
            raise TypeError("Wang implementation receipt value cannot be boolean")
        if _strict_integer(self.value):
            object.__setattr__(self, "value", int(self.value))
        elif isinstance(self.value, Real):
            object.__setattr__(
                self, "value", _finite_real(self.value, "Wang implementation value")
            )
        elif not isinstance(self.value, str) or not self.value.strip():
            raise TypeError("Wang implementation receipt requires a scalar or label")
        if self.role in (
            "scalar_energy_functional",
            "scf_hamiltonian_derivative",
            "finite_q_hessian_derivative",
            "shared_functional_source",
        ):
            if not isinstance(self.value, str):
                raise TypeError(f"Wang {self.role} value must be a SHA256 fingerprint")
            _sha256(self.value, f"Wang {self.role} value")
        _sha256(self.provider_fingerprint, f"Wang {self.role} provider")
        _sha256(self.source_artifact_sha256, f"Wang {self.role} source artifact")
        if not self.source.strip():
            raise ValueError("Wang implementation receipt requires source provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class WangCutoffReceipt:
    cutoff_kind: Literal["parent_plane_wave", "interaction_transfer"]
    value: int | str
    generator_fingerprint: str
    source_artifact_sha256: str
    authority_kind: ReceiptAuthority
    source: str

    def __post_init__(self) -> None:
        if self.cutoff_kind not in ("parent_plane_wave", "interaction_transfer"):
            raise ValueError("unknown Wang cutoff kind")
        if _strict_integer(self.value):
            if int(self.value) <= 0:
                raise ValueError("Wang cutoff integer must be positive")
            object.__setattr__(self, "value", int(self.value))
        elif not isinstance(self.value, str) or not self.value.strip():
            raise TypeError("Wang cutoff must be a positive integer or label")
        if self.authority_kind not in (
            "reproduction_choice",
            "independent_provider_explicit",
        ):
            raise ValueError("invalid Wang cutoff receipt authority")
        _sha256(self.generator_fingerprint, "Wang cutoff generator")
        _sha256(self.source_artifact_sha256, "Wang cutoff source artifact")
        if not self.source.strip():
            raise ValueError("Wang cutoff receipt requires source provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class WangIKSBoostLabel:
    fractional_G1_G2: tuple[float, float]
    raw_grid_shift: tuple[int, int]
    canonical: tuple[int, int]
    reciprocal_carry: tuple[int, int]
    mesh_receipt_fingerprint: str
    role: str = "ground_state_iks_valley_boost_not_tdhf_transfer_q"

    def __post_init__(self) -> None:
        integer_labels = self.raw_grid_shift + self.canonical + self.reciprocal_carry
        if any(not _strict_integer(item) for item in integer_labels):
            raise TypeError("Wang IKS grid labels/carry must use strict integers")
        if len(self.fractional_G1_G2) != 2:
            raise ValueError("Wang IKS boost requires two fractional coordinates")
        for value in self.fractional_G1_G2:
            _finite_real(value, "Wang IKS fractional coordinate")
        expected = ((0.5, 0.0), (5, 0), (5, 0), (0, 0))
        actual = (
            tuple(float(x) for x in self.fractional_G1_G2),
            self.raw_grid_shift,
            self.canonical,
            self.reciprocal_carry,
        )
        if actual != expected or self.role != (
            "ground_state_iks_valley_boost_not_tdhf_transfer_q"
        ):
            raise ValueError("Wang N=10 qIKS boost label was changed")
        _sha256(self.mesh_receipt_fingerprint, "Wang IKS mesh receipt")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class WangTDHFTransferLabel:
    sign: Literal["plus_M", "minus_M"]
    raw: tuple[int, int]
    canonical: tuple[int, int]
    reciprocal_carry: tuple[int, int]
    mesh_receipt_fingerprint: str
    role: str = "tdhf_external_transfer_q_not_ground_state_qIKS"
    q_classification: str = "self_conjugate_exact_M_raw_alias"

    def __post_init__(self) -> None:
        integer_labels = self.raw + self.canonical + self.reciprocal_carry
        if any(not _strict_integer(item) for item in integer_labels):
            raise TypeError("Wang TDHF grid labels/carry must use strict integers")
        expected = {
            "plus_M": ((5, 0), (5, 0), (0, 0)),
            "minus_M": ((-5, 0), (5, 0), (-1, 0)),
        }
        if self.sign not in expected:
            raise ValueError("unknown Wang signed TDHF endpoint")
        if (self.raw, self.canonical, self.reciprocal_carry) != expected[self.sign]:
            raise ValueError("Wang raw signed TDHF endpoint/carry was changed")
        if self.role != "tdhf_external_transfer_q_not_ground_state_qIKS":
            raise ValueError("Wang TDHF q role cannot be merged with qIKS")
        if self.q_classification != "self_conjugate_exact_M_raw_alias":
            raise ValueError("Wang exact-M transfer classification was changed")
        _sha256(self.mesh_receipt_fingerprint, "Wang TDHF mesh receipt")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class WangMeshSignedQReceipt:
    """Executable mesh receipt retaining raw +/-M representatives and carries."""

    shape: tuple[int, int]
    reciprocal_basis_fingerprint: str
    mesh_generator_fingerprint: str
    index_order: str
    raw_tdhf_endpoints: tuple[tuple[int, int], tuple[int, int]]
    canonical_tdhf_endpoints: tuple[tuple[int, int], tuple[int, int]]
    raw_reciprocal_carries: tuple[tuple[int, int], tuple[int, int]]
    endpoint_classification: str
    source_artifact_sha256: str
    source: str
    authority_kind: ReceiptAuthority = "reproduction_choice"

    def __post_init__(self) -> None:
        if self.shape != (10, 10) or any(not _strict_integer(x) for x in self.shape):
            raise ValueError("Wang principal TDHF receipt requires strict 10x10 shape")
        if not self.index_order.strip():
            raise ValueError("Wang mesh receipt must bind index order")
        integer_labels = (
            self.raw_tdhf_endpoints
            + self.canonical_tdhf_endpoints
            + self.raw_reciprocal_carries
        )
        if any(not _strict_integer(item) for pair in integer_labels for item in pair):
            raise TypeError("Wang signed-q labels and carries must use strict integers")
        if self.raw_tdhf_endpoints != ((5, 0), (-5, 0)):
            raise ValueError("Wang receipt must preserve raw +M and -M endpoints")
        if self.canonical_tdhf_endpoints != ((5, 0), (5, 0)):
            raise ValueError("Wang signed M endpoints have the wrong canonical aliases")
        if self.raw_reciprocal_carries != ((0, 0), (-1, 0)):
            raise ValueError("Wang signed M endpoints have the wrong raw carries")
        if self.endpoint_classification != "self_conjugate_exact_M_raw_alias_pair":
            raise ValueError("Wang signed endpoints require exact-M classification")
        for raw, canonical, carry in zip(
            self.raw_tdhf_endpoints,
            self.canonical_tdhf_endpoints,
            self.raw_reciprocal_carries,
            strict=True,
        ):
            derived_canonical = (raw[0] % 10, raw[1] % 10)
            derived_carry = (
                (raw[0] - derived_canonical[0]) // 10,
                (raw[1] - derived_canonical[1]) // 10,
            )
            if canonical != derived_canonical or carry != derived_carry:
                raise ValueError("Wang mesh raw/canonical/carry relation does not close")
        if self.authority_kind not in (
            "reproduction_choice",
            "independent_provider_explicit",
        ):
            raise ValueError("invalid Wang mesh receipt authority")
        _sha256(self.reciprocal_basis_fingerprint, "Wang reciprocal basis")
        _sha256(self.mesh_generator_fingerprint, "Wang mesh generator")
        _sha256(self.source_artifact_sha256, "Wang mesh source artifact")
        if not self.source.strip():
            raise ValueError("Wang mesh receipt requires source provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    @property
    def signed_q_kind(self) -> TDHFSignedQ:
        return classify_tdhf_signed_q(
            plus_raw=self.raw_tdhf_endpoints[0],
            minus_raw=self.raw_tdhf_endpoints[1],
            plus_canonical=self.canonical_tdhf_endpoints[0],
            minus_canonical=self.canonical_tdhf_endpoints[1],
            provenance=f"wang_mesh_signed_q={self.fingerprint}",
        )

    @property
    def iks_boost(self) -> WangIKSBoostLabel:
        return WangIKSBoostLabel(
            (0.5, 0.0),
            (5, 0),
            (5, 0),
            (0, 0),
            self.fingerprint,
        )

    @property
    def plus_m(self) -> WangTDHFTransferLabel:
        return WangTDHFTransferLabel(
            "plus_M", (5, 0), (5, 0), (0, 0), self.fingerprint
        )

    @property
    def minus_m(self) -> WangTDHFTransferLabel:
        return WangTDHFTransferLabel(
            "minus_M", (-5, 0), (5, 0), (-1, 0), self.fingerprint
        )


@dataclass(frozen=True, slots=True)
class Wang2025Spec:
    """Paper prescription plus optional executable metadata receipts."""

    paper: WangPaperPrescription = field(default_factory=WangPaperPrescription)
    primary_source: WangPrimarySourceReceipt = field(
        default_factory=WangPrimarySourceReceipt
    )
    author_family: WangAuthorFamilyTBGHFReceipt = field(
        default_factory=WangAuthorFamilyTBGHFReceipt
    )
    strain_beta: WangImplementationReceipt | None = None
    parent_plane_wave_cutoff: WangCutoffReceipt | None = None
    interaction_transfer_cutoff: WangCutoffReceipt | None = None
    q0_policy: WangImplementationReceipt | None = None
    exact_p_ref: WangImplementationReceipt | None = None
    mesh_registration_and_carries: WangMeshSignedQReceipt | None = None
    iks_scan_and_scf_policy: WangImplementationReceipt | None = None
    exact_tdhf_q_list_and_tolerances: WangImplementationReceipt | None = None
    scalar_energy_functional: WangImplementationReceipt | None = None
    scf_hamiltonian_derivative: WangImplementationReceipt | None = None
    finite_q_hessian_derivative: WangImplementationReceipt | None = None
    shared_functional_source: WangImplementationReceipt | None = None
    tdhf_implementation_provider: WangImplementationReceipt | None = None

    def __post_init__(self) -> None:
        for field_name, expected_role in (
            ("strain_beta", "strain_beta"),
            ("q0_policy", "q0_policy"),
            ("exact_p_ref", "exact_p_ref"),
            ("iks_scan_and_scf_policy", "iks_scan_and_scf_policy"),
            (
                "exact_tdhf_q_list_and_tolerances",
                "exact_tdhf_q_list_and_tolerances",
            ),
            ("scalar_energy_functional", "scalar_energy_functional"),
            ("scf_hamiltonian_derivative", "scf_hamiltonian_derivative"),
            ("finite_q_hessian_derivative", "finite_q_hessian_derivative"),
            ("shared_functional_source", "shared_functional_source"),
            ("tdhf_implementation_provider", "tdhf_implementation_provider"),
        ):
            receipt = getattr(self, field_name)
            if receipt is not None and receipt.role != expected_role:
                raise ValueError(f"Wang {field_name} receipt has the wrong role")
        if (
            self.parent_plane_wave_cutoff is not None
            and self.parent_plane_wave_cutoff.cutoff_kind != "parent_plane_wave"
        ):
            raise ValueError("Wang parent plane-wave cutoff has the wrong kind")
        if (
            self.interaction_transfer_cutoff is not None
            and self.interaction_transfer_cutoff.cutoff_kind
            != "interaction_transfer"
        ):
            raise ValueError("Wang interaction-transfer cutoff has the wrong kind")
        functional_receipts = (
            self.scalar_energy_functional,
            self.scf_hamiltonian_derivative,
            self.finite_q_hessian_derivative,
            self.shared_functional_source,
        )
        if all(receipt is not None for receipt in functional_receipts):
            source_hashes = {
                receipt.source_artifact_sha256
                for receipt in functional_receipts
                if receipt is not None
            }
            if len(source_hashes) != 1:
                raise ValueError(
                    "Wang scalar/SCF/TDHF functional receipts must share one source artifact"
                )

    @classmethod
    def paper_target(cls) -> "Wang2025Spec":
        return cls()

    @property
    def unresolved_authorities(self) -> tuple[str, ...]:
        labels = (
            "strain_beta",
            "parent_plane_wave_cutoff",
            "interaction_transfer_cutoff",
            "q0_policy",
            "exact_p_ref",
            "mesh_registration_and_carries",
            "iks_scan_and_scf_policy",
            "exact_tdhf_q_list_and_tolerances",
            "scalar_energy_functional",
            "scf_hamiltonian_derivative",
            "finite_q_hessian_derivative",
            "shared_functional_source",
            "tdhf_implementation_provider",
        )
        return tuple(label for label in labels if getattr(self, label) is None)

    @property
    def metadata_resolved(self) -> bool:
        return not self.unresolved_authorities

    @property
    def paper_direct_claim_allowed(self) -> bool:
        return False

    def require_metadata_resolved(self) -> None:
        if self.unresolved_authorities:
            raise RuntimeError(
                "Wang 2025 executable authority is incomplete: "
                + ", ".join(self.unresolved_authorities)
            )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@runtime_checkable
class Wang2025ProviderReceiptProtocol(Protocol):
    """Receipt surface only; deliberately has no build/execute methods."""

    fingerprint: str
    spec_fingerprint: str
    source_commit: str
    strain_beta_receipt_fingerprint: str
    parent_plane_wave_cutoff_receipt_fingerprint: str
    interaction_transfer_cutoff_receipt_fingerprint: str
    q0_policy_receipt_fingerprint: str
    exact_p_ref_receipt_fingerprint: str
    mesh_registration_and_carries_receipt_fingerprint: str
    iks_scan_and_scf_policy_receipt_fingerprint: str
    exact_tdhf_q_list_and_tolerances_receipt_fingerprint: str
    scalar_energy_functional_receipt_fingerprint: str
    scf_hamiltonian_derivative_receipt_fingerprint: str
    finite_q_hessian_derivative_receipt_fingerprint: str
    shared_functional_source_receipt_fingerprint: str
    tdhf_implementation_provider_receipt_fingerprint: str


@dataclass(frozen=True, slots=True)
class Wang2025ProviderBinding:
    """Bind complete metadata receipts without claiming executable readiness."""

    spec: Wang2025Spec
    provider: Wang2025ProviderReceiptProtocol

    def __post_init__(self) -> None:
        self.spec.require_metadata_resolved()
        if not isinstance(self.provider, Wang2025ProviderReceiptProtocol):
            raise TypeError("Wang provider binding requires typed metadata receipts")
        _sha256(self.provider.fingerprint, "Wang provider receipt")
        _commit(self.provider.source_commit, "Wang provider source commit")
        if self.provider.spec_fingerprint != self.spec.fingerprint:
            raise ValueError("Wang provider/spec fingerprint mismatch")
        assert self.spec.strain_beta is not None
        assert self.spec.parent_plane_wave_cutoff is not None
        assert self.spec.interaction_transfer_cutoff is not None
        assert self.spec.q0_policy is not None
        assert self.spec.exact_p_ref is not None
        assert self.spec.mesh_registration_and_carries is not None
        assert self.spec.iks_scan_and_scf_policy is not None
        assert self.spec.exact_tdhf_q_list_and_tolerances is not None
        assert self.spec.scalar_energy_functional is not None
        assert self.spec.scf_hamiltonian_derivative is not None
        assert self.spec.finite_q_hessian_derivative is not None
        assert self.spec.shared_functional_source is not None
        assert self.spec.tdhf_implementation_provider is not None
        expected = (
            (
                "strain beta",
                self.provider.strain_beta_receipt_fingerprint,
                self.spec.strain_beta.fingerprint,
            ),
            (
                "parent plane-wave cutoff",
                self.provider.parent_plane_wave_cutoff_receipt_fingerprint,
                self.spec.parent_plane_wave_cutoff.fingerprint,
            ),
            (
                "interaction-transfer cutoff",
                self.provider.interaction_transfer_cutoff_receipt_fingerprint,
                self.spec.interaction_transfer_cutoff.fingerprint,
            ),
            (
                "q0 policy",
                self.provider.q0_policy_receipt_fingerprint,
                self.spec.q0_policy.fingerprint,
            ),
            (
                "exact P_ref",
                self.provider.exact_p_ref_receipt_fingerprint,
                self.spec.exact_p_ref.fingerprint,
            ),
            (
                "mesh/carries",
                self.provider.mesh_registration_and_carries_receipt_fingerprint,
                self.spec.mesh_registration_and_carries.fingerprint,
            ),
            (
                "IKS scan/SCF",
                self.provider.iks_scan_and_scf_policy_receipt_fingerprint,
                self.spec.iks_scan_and_scf_policy.fingerprint,
            ),
            (
                "TDHF q list/tolerances",
                self.provider.exact_tdhf_q_list_and_tolerances_receipt_fingerprint,
                self.spec.exact_tdhf_q_list_and_tolerances.fingerprint,
            ),
            (
                "scalar energy functional",
                self.provider.scalar_energy_functional_receipt_fingerprint,
                self.spec.scalar_energy_functional.fingerprint,
            ),
            (
                "SCF Hamiltonian derivative",
                self.provider.scf_hamiltonian_derivative_receipt_fingerprint,
                self.spec.scf_hamiltonian_derivative.fingerprint,
            ),
            (
                "finite-q Hessian derivative",
                self.provider.finite_q_hessian_derivative_receipt_fingerprint,
                self.spec.finite_q_hessian_derivative.fingerprint,
            ),
            (
                "shared functional source",
                self.provider.shared_functional_source_receipt_fingerprint,
                self.spec.shared_functional_source.fingerprint,
            ),
            (
                "TDHF implementation",
                self.provider.tdhf_implementation_provider_receipt_fingerprint,
                self.spec.tdhf_implementation_provider.fingerprint,
            ),
        )
        for label, actual, required in expected:
            _sha256(actual, f"Wang provider {label}")
            if actual != required:
                raise ValueError(f"Wang provider/{label} fingerprint mismatch")

    @property
    def provider_bound(self) -> bool:
        return True


__all__ = [
    "WANG_2025_ARXIV",
    "WANG_2025_FIELD_RASTER_SHA256",
    "WANG_2025_FIG6_RASTER_SHA256",
    "WANG_2025_LARGE_GRID_RASTER_SHA256",
    "WANG_2025_MAIN_TEX_SHA256",
    "WANG_2025_PDF_SHA256",
    "WANG_2025_RESPONSE_SCOPE",
    "WANG_2025_SOURCE_ARCHIVE_SHA256",
    "Wang2025ProviderBinding",
    "Wang2025ProviderReceiptProtocol",
    "Wang2025Spec",
    "WangAuthorFamilyTBGHFReceipt",
    "WangCutoffReceipt",
    "WangIKSBoostLabel",
    "WangImplementationReceipt",
    "WangMeshSignedQReceipt",
    "WangModeInventory",
    "WangModeScales",
    "WangPaperPrescription",
    "WangPrimarySourceReceipt",
    "WangQualifiedModeScale",
    "WangRasterOnlyTargets",
    "WangScenarioModes",
    "WangTDHFTransferLabel",
]
