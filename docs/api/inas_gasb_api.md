# InAs/GaSb API

The maintained InAs/GaSb surface is API-centered. Historical plane-wave qualification code, source-attestation workflows, publication sealers, and standalone devtools are archived rather than treated as runtime APIs.

## Analytic BHZ model

```python
from mean_field.api import make_model, compute_bands

model = make_model(
    "inas_gasb",
    variant="xue2018_q0_bhz",
    eg_ry=-0.5,
    hybridization_ab_ry=0.2,
    path_extent_ab_inv=0.3,
)
bands = compute_bands(model, points_per_segment=120)
```

`variant` and the paper parameters are mandatory. The API does not invent a material parameter set.

The Hamiltonian is delegated exactly to `zeng2022.build_zeng2022_folded_h0`:

- momentum: `a_B*^-1`;
- energy: `Ry*`;
- basis/order and spin-block signs: owned by `ZengSlabBasis` and `zeng2022_spin_block`;
- `xue2018_q0_bhz`: source Q=0 symmetric-mass policy from `xue2018_standard_parameters`;
- `zeng2022_folded_bhz`: explicit `Zeng2022Parameters` and slab indices.

`path_extent_ab_inv` is a numerical continuum cutoff, not a Brillouin-zone boundary. The convenience Γ–Xcut–Mcut–Γ path and half-open square grid are diagnostics in this declared continuum domain.

## Xue Q=0 mean field

```python
from mean_field.systems.inas_gasb import InAsGaSbHFConfig, run_inas_gasb_hf

result = run_inas_gasb_hf(
    model,
    InAsGaSbHFConfig(
        kmax_ab_inv=0.3,
        points_per_axis=31,
        init_mode="normal",
    ),
)
```

This is a typed façade over the retained `build_xue2018_hf_state` and `run_xue2018_hf` owners. It adds no Hamiltonian, occupation, Hartree/Fock, ODA, or normalization formula. The cutoff, mesh, self-cell policy, kernel backend, initialization, and iteration controls are explicit.

Heavy HF runs remain Slurm/test-node work.

## Saved Kane4 model

```python
from mean_field.systems.inas_gasb import compute_kane4_bands
from mean_field.systems.inas_gasb.kane4_bundle import load_kane4_bundle

bundle = load_kane4_bundle("kane4.npz")
bands = compute_kane4_bands(bundle, return_eigenvectors=True)
```

`compute_kane4_bands` diagonalizes only `bundle.h0_mev[:, :, ik]` in the exact saved `bundle.k_cart_nm_inv` order. It has no off-grid, nearest-grid, interpolation, or path-reconstruction input. Energies remain in meV and momenta in nm⁻¹.

New radial bundles may be constructed only with `load_split_zero_kdotpy_radial_kane4_bundle`. It requires the exact expected XML SHA-256 and wavefunction-manifest SHA-256; source attestation cannot be disabled. Historical nonzero-split and Cartesian artifact migration loaders are archived rather than maintained APIs. Canonical Kane-Poisson archive loading likewise accepts only the current typed schema.

Matrix mean field remains available through a source-bound interaction operator:

```python
from mean_field.systems.inas_gasb.matrix_ei import (
    MatrixEIConfig,
    solve_reference_subtracted_matrix_ei,
)
from mean_field.systems.inas_gasb.projected_fock import (
    ProjectedFockOperator,
    uniform_dielectric_green_on_mesh_mev_nm2,
)

green = uniform_dielectric_green_on_mesh_mev_nm2(
    bundle.k_cart_nm_inv,
    bundle.weights_nm2,
    bundle.z_nm,
    epsilon_r=15.0,
)
interaction = ProjectedFockOperator.from_bundle(
    bundle,
    green,
    self_cell_description="equal-area mesh cell",
)
result = solve_reference_subtracted_matrix_ei(
    bundle,
    interaction,
    config=MatrixEIConfig(reference_policy="normal_ordered_exchange_only"),
)
```

Before iteration, the solver routes the operator through the closed, fail-closed registry in `matrix_ei_interaction.py`. The resulting `MatrixEIInteractionProtocol` fixes the reference policy, Hartree/Fock component split, electrostatic ensemble, diagnostic label, and authority from the registered concrete type; structural duck typing and caller-supplied semantic labels are rejected. The original operator still performs the numerical action and validates against the same bundle fingerprint, so this routing adds no interaction formula.

## Canonical Kane-Poisson

The maintained lower-level runtime constructors are imported from their explicit owner modules rather than the high-level package façade:

```python
from mean_field.systems.inas_gasb.kane_poisson import (
    CanonicalKanePoissonConfig,
    solve_canonical_split_zero_kane_poisson,
)
from mean_field.systems.inas_gasb.kdotpy_window_builder import (
    KdotpyCanonicalWindowBuilder,
    KdotpyCanonicalWindowOnlyBuilder,
)

raw_builder = KdotpyCanonicalWindowBuilder(**builder_kwargs)
builder = KdotpyCanonicalWindowOnlyBuilder(raw_builder)

# Every replay must be a genuinely new, cold builder with the same parent spec.
def replay_factory():
    return KdotpyCanonicalWindowOnlyBuilder(
        KdotpyCanonicalWindowBuilder(**builder_kwargs)
    )

result = solve_canonical_split_zero_kane_poisson(
    builder,
    independent_replay_builder_factory=replay_factory,
    parent_spec=builder.parent_spec,
    initial_potential_mev=initial_potential_mev,
    config=CanonicalKanePoissonConfig(),
)
```

`builder_kwargs` must provide the kdotpy parent, material/source hashes, grids, selected bands, eigensolver policy, and a mandatory `KdotpyE1H1PairSelectionSpec`. The maintained builder has no anchor-only or auto-nearest quartet mode: every call returns a source-bound pair-resolved receipt before `KdotpyCanonicalWindowOnlyBuilder` unwraps the canonical window. Optional previous-potential homotopy uses `KdotpyPreviousUSameKHomotopySpec`.

Typed numerical-owner results and parameter objects such as `Xue2018HFResult`, `Zeng2022Parameters`, `KaneChargeNeutralReference`, `KaneFixedMuState`, and `ChemicalPotentialRootTolerances` remain available from their explicit owner modules.

## Public entrypoints

- Common model/band façade: `mean_field.api.make_model`, `mean_field.api.compute_bands`.
- System-specific high-level API: `mean_field.systems.inas_gasb`.
- The system package façade intentionally exposes exactly six model, band, and HF entrypoints from `api.py`; `tests/test_inas_gasb_api.py` freezes this contract.
- Kane-Poisson, archive-I/O, normal-reference, interaction-operator, and matrix-EI development surfaces are imported from their explicit numerical-owner modules. They are maintained capabilities, not package-root compatibility promises.
- Formula helpers, diagnostics, gauge transforms, and test oracles likewise remain owner-module implementation surfaces rather than façade exports.
- The project root `mean_field` does not duplicate every system-specific symbol.

Archived workflows are provenance only and are not current operational authority.
