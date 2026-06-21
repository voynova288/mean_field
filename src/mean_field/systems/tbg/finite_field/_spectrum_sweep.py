from __future__ import annotations

from ._spectrum_shared import *  # noqa: F401,F403
from ._spectrum_params import *  # noqa: F401,F403

@dataclass(frozen=True)
class MagneticSpectrumSweepCase:
    """One flux point in a paper-style Hofstadter spectrum sweep."""

    flux: MagneticFlux
    nq: int
    n_landau: int
    mesh_shift: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "nq", int(self.nq))
        object.__setattr__(self, "n_landau", int(self.n_landau))
        object.__setattr__(self, "mesh_shift", float(self.mesh_shift))
        if self.nq <= 0:
            raise ValueError(f"nq must be positive, got {self.nq}")
        if self.n_landau <= 0:
            raise ValueError(f"n_landau must be positive, got {self.n_landau}")

@dataclass(frozen=True)
class MagneticSpectrumSweepResult:
    """No-I/O result for a paper-style magnetic-spectrum sweep."""

    cases: tuple[MagneticSpectrumSweepCase, ...]
    spectra: tuple[MagneticSpectrumResult, ...]
    red_group_masks: tuple[Array, ...]

    def __post_init__(self) -> None:
        if not (len(self.cases) == len(self.spectra) == len(self.red_group_masks)):
            raise ValueError("cases, spectra, and red_group_masks must have the same length")
        masks = tuple(np.asarray(mask, dtype=bool) for mask in self.red_group_masks)
        for case, result, mask in zip(self.cases, self.spectra, masks, strict=True):
            if int(result.flux.p) != case.flux.p or int(result.flux.q) != case.flux.q:
                raise ValueError("Sweep case/result flux mismatch")
            if mask.shape != (result.n_subbands,):
                raise ValueError(f"Expected red-group mask shape {(result.n_subbands,)}, got {mask.shape}")
        object.__setattr__(self, "red_group_masks", masks)

    def as_point_table(self) -> dict[str, Array]:
        """Return flattened arrays suitable for paper-style scatter plots."""

        p_vals: list[int] = []
        q_vals: list[int] = []
        nq_vals: list[int] = []
        n_landau_vals: list[int] = []
        band_vals: list[int] = []
        i1_vals: list[int] = []
        i2_vals: list[int] = []
        energy_vals: list[float] = []
        red_vals: list[bool] = []
        for case, result, red_mask in zip(self.cases, self.spectra, self.red_group_masks, strict=True):
            for i2 in range(result.nq):
                for i1 in range(result.nq):
                    for band in range(result.n_subbands):
                        p_vals.append(case.flux.p)
                        q_vals.append(case.flux.q)
                        nq_vals.append(case.nq)
                        n_landau_vals.append(case.n_landau)
                        band_vals.append(band + 1)
                        i1_vals.append(i1 + 1)
                        i2_vals.append(i2 + 1)
                        energy_vals.append(float(result.spectrum[band, i1, i2]))
                        red_vals.append(bool(red_mask[band]))
        return {
            "p": np.asarray(p_vals, dtype=int),
            "q": np.asarray(q_vals, dtype=int),
            "phi": np.asarray(p_vals, dtype=float) / np.asarray(q_vals, dtype=float),
            "nq": np.asarray(nq_vals, dtype=int),
            "n_landau": np.asarray(n_landau_vals, dtype=int),
            "band": np.asarray(band_vals, dtype=int),
            "i1": np.asarray(i1_vals, dtype=int),
            "i2": np.asarray(i2_vals, dtype=int),
            "energy_mev": np.asarray(energy_vals, dtype=float),
            "red_group": np.asarray(red_vals, dtype=bool),
        }

def paper_hofstadter_fluxes(*, max_denominator: int = 12, phi_max: float = 0.5) -> tuple[MagneticFlux, ...]:
    """Return the rational fluxes used for paper-style Hofstadter panels."""

    max_denominator = int(max_denominator)
    if max_denominator <= 0:
        raise ValueError(f"max_denominator must be positive, got {max_denominator}")
    fracs = {
        Fraction(p, q)
        for q in range(1, max_denominator + 1)
        for p in range(1, q + 1)
        if float(Fraction(p, q)) <= float(phi_max) + 1e-15
    }
    return tuple(MagneticFlux(frac.numerator, frac.denominator) for frac in sorted(fracs))

def author_landau_cutoff(flux: MagneticFlux, *, cutoff: int = 25) -> int:
    """Return author production cutoff ``nLL = cutoff*q/p``."""

    if flux.p <= 0:
        raise ValueError(f"Author paper spectra expect positive p, got {flux.p}")
    return int(int(cutoff) * flux.q // flux.p)

def red_chern_minus_one_group_mask(flux: MagneticFlux) -> Array:
    """Mask the lower C=-1 magnetic-subband group used in Fig. 3(a).

    For flux ``p/q`` the C=-1 group below charge neutrality contains the
    lower ``q-p`` subbands of the ascending-energy ``2q`` central spectrum.
    The complementary ``p`` subbands immediately below charge neutrality are
    not the red group in the paper panel.
    """

    p_flux, q_flux = int(flux.p), int(flux.q)
    if p_flux <= 0 or p_flux > q_flux:
        raise ValueError(f"Expected 0 < p <= q for red-group mask, got p/q={p_flux}/{q_flux}")
    mask = np.zeros(2 * q_flux, dtype=bool)
    mask[: q_flux - p_flux] = True
    return mask

def _value_for_flux(value, flux: MagneticFlux):
    if callable(value):
        return value(flux)
    if isinstance(value, Mapping):
        if (flux.p, flux.q) in value:
            return value[(flux.p, flux.q)]
        label = f"{flux.p}/{flux.q}"
        if label in value:
            return value[label]
        return 0.0
    return value

def compute_magnetic_spectrum_sweep(
    params: FiniteFieldBMParameters,
    *,
    fluxes: Sequence[MagneticFlux | tuple[int, int] | Fraction | str] | None = None,
    max_denominator: int = 12,
    phi_max: float = 0.5,
    valley: Valley = "K",
    landau_cutoff: int = 25,
    n_landau_by_flux: Callable[[MagneticFlux], int] | Mapping[tuple[int, int] | str, int] | None = None,
    nq_by_flux: Callable[[MagneticFlux], int] | Mapping[tuple[int, int] | str, int] | None = None,
    mesh_shift_by_flux: Callable[[MagneticFlux], float] | Mapping[tuple[int, int] | str, float] | float = 0.0,
    sigma_rotation: bool = False,
    hbn: bool = False,
    include_strain: bool = True,
    q0: complex = 0.0 + 0.0j,
) -> MagneticSpectrumSweepResult:
    """Compute a no-I/O paper-style finite-B Hofstadter spectrum sweep.

    By default this matches the Fig. 3(a)-style flux set ``q<=12`` and
    ``phi<=1/2`` with the author cutoff ``nLL=25*q/p`` and mesh rule
    ``choose_magnetic_nq(q)``. Tests and small diagnostics can override
    ``fluxes``, ``n_landau_by_flux``, and ``nq_by_flux``.
    """

    from ._spectrum_hamiltonian import compute_magnetic_spectrum

    normalized_fluxes = (
        paper_hofstadter_fluxes(max_denominator=max_denominator, phi_max=phi_max)
        if fluxes is None
        else tuple(MagneticFlux.from_value(flux) if not isinstance(flux, MagneticFlux) else flux for flux in fluxes)
    )
    cases: list[MagneticSpectrumSweepCase] = []
    spectra: list[MagneticSpectrumResult] = []
    red_masks: list[Array] = []
    for flux in normalized_fluxes:
        n_landau = int(_value_for_flux(n_landau_by_flux, flux)) if n_landau_by_flux is not None else author_landau_cutoff(flux, cutoff=landau_cutoff)
        nq = int(_value_for_flux(nq_by_flux, flux)) if nq_by_flux is not None else choose_magnetic_nq(flux.q)
        mesh_shift = float(_value_for_flux(mesh_shift_by_flux, flux))
        case = MagneticSpectrumSweepCase(flux=flux, nq=nq, n_landau=n_landau, mesh_shift=mesh_shift)
        result = compute_magnetic_spectrum(
            params,
            flux=flux,
            n_landau=n_landau,
            nq=nq,
            valley=valley,
            sigma_rotation=sigma_rotation,
            hbn=hbn,
            include_strain=include_strain,
            q0=q0,
            mesh_shift=mesh_shift,
        )
        cases.append(case)
        spectra.append(result)
        red_masks.append(red_chern_minus_one_group_mask(flux))
    return MagneticSpectrumSweepResult(cases=tuple(cases), spectra=tuple(spectra), red_group_masks=tuple(red_masks))

__all__ = [name for name in globals() if not name.startswith('__')]
