from __future__ import annotations

from ._spectrum_shared import *  # noqa: F401,F403

def _complex_matrix(values: list[list[complex]]) -> Array:
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True)
class FiniteFieldBMParameters:
    """Author-code finite-B BM parameters.

    Defaults mirror ``libs/params.jl`` and ``initParamsWithStrain`` in the
    original repository rather than the zero-field B0 adapter.  In particular,
    the original finite-B code does not apply the extra inversion-symmetric
    ``K``-point shift used by some B0 workflows.
    """

    dtheta_rad: float
    vf: float = 2482.0
    w0: float = 77.0
    w1: float = 110.0
    delta: float = 0.0
    strain: float = 0.0
    strain_angle_rad: float = 0.0
    poisson: float = 0.16
    beta_g: float = 3.14
    deformation_potential: float = -4100.0

    g1: complex = field(init=False)
    g2: complex = field(init=False)
    a1: complex = field(init=False)
    a2: complex = field(init=False)
    area: float = field(init=False)
    theta12: float = field(init=False)
    kt: complex = field(init=False)
    kb_point: complex = field(init=False)
    omega: complex = field(init=False)
    t0: Array = field(init=False, repr=False)
    t1: Array = field(init=False, repr=False)
    t2: Array = field(init=False, repr=False)
    strain_matrix: Array = field(init=False, repr=False)
    gauge_shift: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dtheta = float(self.dtheta_rad)
        strain = float(self.strain)
        phi = float(self.strain_angle_rad)
        poisson = float(self.poisson)

        kb = 4.0 * np.pi / 3.0 * dtheta
        kt = kb / 2.0 * np.exp(1j * np.pi / 2.0)
        kb_point = -kb / 2.0 * np.exp(1j * np.pi / 2.0)

        omega = np.exp(1j * 2.0 * np.pi / 3.0)
        t0 = _complex_matrix([[self.w0, self.w1], [self.w1, self.w0]])
        # Match original Params.T1/T2, not the B0 neighbor-swapped wrapper.
        t1 = _complex_matrix([[self.w0, self.w1 * omega], [self.w1 * np.conj(omega), self.w0]])
        t2 = _complex_matrix([[self.w0, self.w1 * np.conj(omega)], [self.w1 * omega, self.w0]])

        exx = -strain * np.cos(phi) ** 2 + poisson * strain * np.sin(phi) ** 2
        eyy = poisson * strain * np.cos(phi) ** 2 - strain * np.sin(phi) ** 2
        exy = (1.0 + poisson) * strain * np.cos(phi) * np.sin(phi)
        gauge_shift = (np.sqrt(3.0) * self.beta_g / 2.0) * np.asarray([exx - eyy, -2.0 * exy], dtype=float)
        rotation_phi = np.asarray([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]], dtype=float)
        strain_matrix = rotation_phi.T @ np.asarray([[-strain, 0.0], [0.0, poisson * strain]], dtype=float) @ rotation_phi

        twist_generator = dtheta / 2.0 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float)
        g1_cart = 4.0 * np.pi / np.sqrt(3.0) * np.asarray([0.0, -1.0], dtype=float)
        g2_cart = 4.0 * np.pi / np.sqrt(3.0) * np.asarray([np.sqrt(3.0) / 2.0, 0.5], dtype=float)
        tmp1 = (2.0 * twist_generator - strain_matrix) @ g1_cart
        tmp2 = (2.0 * twist_generator - strain_matrix) @ g2_cart
        g1 = complex(tmp1[0], tmp1[1])
        g2 = complex(tmp2[0], tmp2[1])

        reciprocal_area = abs(g1.real * g2.imag - g1.imag * g2.real)
        a1 = 2.0 * np.pi / reciprocal_area * complex(g2.imag, -g2.real)
        a2 = 2.0 * np.pi / reciprocal_area * complex(-g1.imag, g1.real)
        area = abs((np.conj(a1) * a2).imag)
        theta12 = np.angle(a2) - np.angle(a1)

        kt = kt + complex(gauge_shift[0], gauge_shift[1]) / 2.0 - complex(strain_matrix[0, 0], strain_matrix[1, 0]) * 2.0 * np.pi / 3.0
        kb_point = kb_point - complex(gauge_shift[0], gauge_shift[1]) / 2.0 + complex(strain_matrix[0, 0], strain_matrix[1, 0]) * 2.0 * np.pi / 3.0

        object.__setattr__(self, "g1", complex(g1))
        object.__setattr__(self, "g2", complex(g2))
        object.__setattr__(self, "a1", complex(a1))
        object.__setattr__(self, "a2", complex(a2))
        object.__setattr__(self, "area", float(area))
        object.__setattr__(self, "theta12", float(theta12))
        object.__setattr__(self, "kt", complex(kt))
        object.__setattr__(self, "kb_point", complex(kb_point))
        object.__setattr__(self, "omega", complex(omega))
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)
        object.__setattr__(self, "t2", t2)
        object.__setattr__(self, "strain_matrix", strain_matrix)
        object.__setattr__(self, "gauge_shift", gauge_shift)

    @classmethod
    def from_degrees(
        cls,
        theta_deg: float,
        *,
        w0: float = 77.0,
        w1: float = 110.0,
        strain: float = 0.0,
        strain_angle_deg: float = 0.0,
        deformation_potential: float = -4100.0,
        vf: float = 2482.0,
    ) -> "FiniteFieldBMParameters":
        return cls(
            dtheta_rad=float(theta_deg) * np.pi / 180.0,
            vf=vf,
            w0=w0,
            w1=w1,
            strain=strain,
            strain_angle_rad=float(strain_angle_deg) * np.pi / 180.0,
            deformation_potential=deformation_potential,
        )


@dataclass(frozen=True)
class MagneticSpectrumResult:
    """Central Hofstadter subbands and projected data for one valley."""

    params: FiniteFieldBMParameters
    flux: MagneticFlux
    valley: Valley
    n_landau: int
    n_h: int
    nq: int
    sigma_rotation: bool
    l_b: float
    q_phi: complex
    lattice_k1: Array
    lattice_k2: Array
    hamiltonian_ll: Array
    sigma_z_ll: Array
    spectrum: Array
    vec: Array
    p_sigma_z: Array
    sigma_z_eigenvalues: Array
    sigma_z_energy_diag: Array

    @property
    def q(self) -> int:
        return self.flux.q

    @property
    def p(self) -> int:
        return self.flux.p

    @property
    def n_subbands(self) -> int:
        return 2 * self.q

    @property
    def inner_dim(self) -> int:
        return 2 * self.n_h * self.p

    def overlap_data_for_shifts(self, shifts: tuple[tuple[int, int], ...], *, fast: bool = True) -> MagneticOverlapData:
        """Return 4D overlap data for the requested finite-B shifts."""

        from ._spectrum_overlap import compute_coulomb_overlap, compute_coulomb_overlap_fast

        overlaps: dict[tuple[int, int], Array] = {}
        gvecs: list[complex] = []
        nk_full = self.q * self.nq * self.nq
        overlap_fn = compute_coulomb_overlap_fast if fast else compute_coulomb_overlap
        for m, n in shifts:
            shift = (int(m), int(n))
            flat = overlap_fn(self, shift[0], shift[1])
            overlaps[shift] = flat.reshape((self.n_subbands, nk_full, self.n_subbands, nk_full), order="F")
            gvecs.append(magnetic_reciprocal_vector(shift[0], shift[1], g1=self.params.g1, g2=self.params.g2, q=self.q))
        return MagneticOverlapData(shifts=tuple((int(m), int(n)) for m, n in shifts), gvecs=np.asarray(gvecs, dtype=np.complex128), overlaps=overlaps)

__all__ = [name for name in globals() if not name.startswith('__')]
