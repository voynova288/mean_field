from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .params import TDBGParameters, delta_from_paper_ud

SPIN_LABELS: tuple[str, str] = ("up", "down")
VALLEY_SEQUENCE: tuple[int, int] = (1, -1)
VALLEY_LABELS: dict[int, str] = {1: "K", -1: "Kprime"}
TDBG_LOCAL_LABELS: tuple[str, str, str, str] = ("A1", "B1", "A2", "B2")
TDBGPaperUdConvention = Literal["same_delta_minus_ud_over3", "minus_xi_ud_over3"]
VALID_PAPER_UD_CONVENTIONS: tuple[str, str] = ("same_delta_minus_ud_over3", "minus_xi_ud_over3")


def tdbg_delta_from_paper_ud_for_valley(
    paper_ud_ev: float,
    valley: int,
    *,
    convention: TDBGPaperUdConvention = "same_delta_minus_ud_over3",
) -> float:
    """Map Liu paper ``U_d`` to the local TDBG ``Delta`` for one valley.

    ``same_delta_minus_ud_over3`` preserves the historical average-zero layer
    potential mapping used by :func:`delta_from_paper_ud`.  ``minus_xi_ud_over3``
    follows the user-identified noninteracting Liu-band convention where K uses
    ``-U_d/3`` and K' uses ``+U_d/3``; this is the convention that keeps the
    trusted pytwist-style valley separation in the paper-band diagnostic.
    """

    valley = int(valley)
    if valley not in VALLEY_SEQUENCE:
        raise ValueError(f"Expected valley in {VALLEY_SEQUENCE}, got {valley}")
    if convention == "same_delta_minus_ud_over3":
        return delta_from_paper_ud(paper_ud_ev)
    if convention == "minus_xi_ud_over3":
        return -float(valley) * float(paper_ud_ev) / 3.0
    raise ValueError(f"Unsupported TDBG paper-Ud convention: {convention!r}")


def tdbg_parameters_from_paper_ud_for_valley(
    paper_ud_ev: float,
    *,
    stacking: str,
    valley: int,
    convention: TDBGPaperUdConvention = "same_delta_minus_ud_over3",
) -> TDBGParameters:
    """Construct TDBG parameters for a valley-resolved Liu paper ``U_d`` convention."""

    return TDBGParameters.full(
        stacking=stacking,
        valley=int(valley),
        Delta=tdbg_delta_from_paper_ud_for_valley(paper_ud_ev, int(valley), convention=convention),
    )


@dataclass(frozen=True)
class TDBGProjectedWindow:
    """Band window for a TDBG projected-HF calculation.

    The default paper-scout path is ``two_flat``: one valence and one
    conduction band around charge neutrality for each spin/valley.  Larger
    windows can be requested as ``central4``/``central6`` or by explicit band
    indices.  The window definition is intentionally separate from the filling
    and reference-density convention.
    """

    name: str = "two_flat"
    band_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TDBGInteractionSettings:
    """Interaction choices used by the reusable TDBG projected-HF adapter."""

    include_intersite: bool = True
    include_onsite: bool = True
    hubbard_u_ev: float = 0.5
    epsilon_r: float = 10.0
    kappa_nm_inv: float = 0.05  # Liu SI kappa = 0.005 Angstrom^-1
    g_shells: int | None = None
    hartree_reference: Literal["none", "charge_neutral"] = "charge_neutral"
    fock_density: Literal["absolute", "reference_subtracted"] = "absolute"
    onsite_valley_policy: Literal["valley_diagonal", "all_overlaps"] = "valley_diagonal"
    drop_g0_hartree: bool = False


@dataclass(frozen=True)
class TDBGProjectedHFConfig:
    """Configuration for Liu-style TDBG projected-HF state searches."""

    theta_deg: float = 1.38
    cut: float = 5.0
    mesh_size: int = 9
    paper_ud_ev: float = 0.09
    paper_ud_convention: TDBGPaperUdConvention = "same_delta_minus_ud_over3"
    stacking: str = "AB-BA"
    window: TDBGProjectedWindow = field(default_factory=TDBGProjectedWindow)
    filling: int = 2
    interaction: TDBGInteractionSettings = field(default_factory=TDBGInteractionSettings)
    precision: float = 1.0e-7
    max_iter: int = 300
    mix_fallback: float | None = None
    frac_shift: tuple[float, float] | None = None
    orbital_zeeman_b_t: float = 0.0
    orbital_zeeman_delta_k_nm_inv: float = 1.0e-5


def validate_tdbg_interaction_settings(settings: TDBGInteractionSettings) -> None:
    """Validate runtime policy strings and positive interaction scales.

    ``Literal`` annotations document the supported choices, but they do not
    protect project scripts at runtime. The HF adapter raises early instead of
    silently falling through to an unintended density/reference convention.
    """

    if settings.hartree_reference not in {"none", "charge_neutral"}:
        raise ValueError(f"Unsupported TDBG Hartree reference policy: {settings.hartree_reference!r}")
    if settings.fock_density not in {"absolute", "reference_subtracted"}:
        raise ValueError(f"Unsupported TDBG Fock density policy: {settings.fock_density!r}")
    if settings.onsite_valley_policy not in {"valley_diagonal", "all_overlaps"}:
        raise ValueError(f"Unsupported TDBG onsite valley policy: {settings.onsite_valley_policy!r}")
    if settings.hubbard_u_ev < 0.0:
        raise ValueError(f"hubbard_u_ev must be non-negative, got {settings.hubbard_u_ev}")
    if settings.epsilon_r <= 0.0:
        raise ValueError(f"epsilon_r must be positive, got {settings.epsilon_r}")
    if settings.kappa_nm_inv <= 0.0:
        raise ValueError(f"kappa_nm_inv must be positive, got {settings.kappa_nm_inv}")
    if settings.g_shells is not None and int(settings.g_shells) < 0:
        raise ValueError(f"g_shells must be non-negative, got {settings.g_shells}")


def validate_tdbg_projected_hf_config(config: TDBGProjectedHFConfig) -> None:
    """Validate non-iterative TDBG projected-HF configuration choices."""

    validate_tdbg_interaction_settings(config.interaction)
    if int(config.filling) != config.filling:
        raise ValueError(f"filling must be an integer charge relative to neutrality, got {config.filling}")
    if config.paper_ud_convention not in VALID_PAPER_UD_CONVENTIONS:
        raise ValueError(f"Unsupported paper_ud_convention={config.paper_ud_convention!r}")
    if config.mesh_size <= 0:
        raise ValueError(f"mesh_size must be positive, got {config.mesh_size}")
    if config.cut <= 0.0:
        raise ValueError(f"cut must be positive, got {config.cut}")
    if config.precision <= 0.0:
        raise ValueError(f"precision must be positive, got {config.precision}")
    if config.max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {config.max_iter}")
    if config.mix_fallback is not None and not (0.0 < float(config.mix_fallback) <= 1.0):
        raise ValueError(f"mix_fallback must lie in (0, 1] when set, got {config.mix_fallback}")
    if config.orbital_zeeman_delta_k_nm_inv <= 0.0:
        raise ValueError(f"orbital_zeeman_delta_k_nm_inv must be positive, got {config.orbital_zeeman_delta_k_nm_inv}")

__all__ = [
    "SPIN_LABELS",
    "TDBGInteractionSettings",
    "TDBG_LOCAL_LABELS",
    "TDBGPaperUdConvention",
    "TDBGProjectedHFConfig",
    "TDBGProjectedWindow",
    "VALID_PAPER_UD_CONVENTIONS",
    "VALLEY_LABELS",
    "VALLEY_SEQUENCE",
    "tdbg_delta_from_paper_ud_for_valley",
    "tdbg_parameters_from_paper_ud_for_valley",
    "validate_tdbg_interaction_settings",
    "validate_tdbg_projected_hf_config",
]
