from __future__ import annotations

"""Typed RLG/hBN HF archive writer for TDHF-ready saved states."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from mean_field.core.io import write_npz_artifact

from ._hf_interaction_path import (
    RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION,
)
from ._hf_shared import (
    RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
    RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
    RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
)
from ._hf_types import RLGhBNHartreeFockRun


_PROVENANCE_ARCHIVE_FIELDS = frozenset(
    {
        "hf_interaction_convention",
        "hf_quotient_enabled",
        "hf_beta",
        "hf_physical_shifts",
        "zero_literal_q0_fock",
        "hf_basis_periodic_gauge",
        "hf_basis_periodic_gauge_padding",
        "hf_form_factor_convention",
        "hf_remote_h0_policy",
        "hf_remote_h0_sha256",
        "hf_physical_shift_policy",
        "hf_provider_fingerprint",
        "hf_provider_schema_version",
    }
)


def _complex_to_pairs(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    return np.stack((array.real, array.imag), axis=-1)


def _trace_array(
    trace: Mapping[str, Sequence[float] | Sequence[int]],
    key: str,
) -> np.ndarray:
    return np.asarray(trace.get(key, ()), dtype=float)


def save_rlg_hbn_hf_archive(
    path: str | Path,
    run: RLGhBNHartreeFockRun,
    trace: Mapping[str, Sequence[float] | Sequence[int]],
    *,
    cache_metadata: Mapping[str, object] | None = None,
    require_interaction_provenance: bool = True,
) -> Path:
    """Write an RLG/hBN HF state with the provenance required by TDHF replay.

    Provider-schema archives fail closed unless the attached Track-P provider
    matches the run's exact basis, overlap blocks, state, and fingerprint.
    Generic atomic NPZ publication is delegated to :mod:`mean_field.core.io`;
    this module owns only RLG/hBN field and provenance conventions.
    """

    provenance = run.interaction_provenance
    if provenance is None and require_interaction_provenance:
        raise ValueError(
            "Refusing to save an RLG/hBN HF archive without typed interaction provenance"
        )
    if provenance is not None and provenance.provider_schema_version not in (0, 1):
        raise ValueError(
            f"Unsupported HF provider schema version {provenance.provider_schema_version}"
        )
    if provenance is not None and provenance.provider_schema_version == 1:
        if (
            provenance.convention
            != RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION
        ):
            raise ValueError("Provider schema 1 requires the Track-P convention")
        if provenance.quotient_enabled:
            raise ValueError("Provider schema 1 cannot describe a quotient functional")
        provider = run.track_p_provider
        if provider is None:
            raise ValueError(
                "Refusing a provider-schema archive without an attached Track-P provider"
            )
        if provider.basis_data is not run.basis_data:
            raise ValueError("Archive Track-P provider basis identity mismatch")
        if provider.overlap_blocks is not run.overlap_blocks:
            raise ValueError("Archive Track-P provider overlap identity mismatch")
        provider.validate_state(run.state)
        provider.validate_integrity(recompute_hashes=True)
        if not provenance.provider_fingerprint:
            raise ValueError("Provider-schema archive has a blank fingerprint")
        if provenance.provider_fingerprint != provider.fingerprint:
            raise ValueError("Archive Track-P provider fingerprint mismatch")

    state = run.state
    basis_data = run.basis_data
    payload: dict[str, np.ndarray] = {
        "density": np.asarray(state.density, dtype=np.complex128),
        "hamiltonian": np.asarray(state.hamiltonian, dtype=np.complex128),
        "h0": np.asarray(state.h0, dtype=np.complex128),
        "energies_mev": np.asarray(state.energies, dtype=float),
        "reference_density": np.asarray(state.reference_density, dtype=np.complex128),
        "density_convention": np.asarray("stored_delta"),
        "density_axis_order": np.asarray("abk"),
        "reference_density_convention": np.asarray(str(state.scheme)),
        "basis_periodic_gauge": np.asarray(
            RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION
            if provenance is None
            else provenance.basis_periodic_gauge
        ),
        "basis_periodic_gauge_padding": np.asarray(
            [
                RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING
                if provenance is None
                else provenance.basis_periodic_gauge_padding
            ],
            dtype=int,
        ),
        "form_factor_convention": np.asarray(
            RLG_HBN_FORM_FACTOR_CONVENTION_VERSION
            if provenance is None
            else provenance.form_factor_convention
        ),
        "nu": np.asarray([state.nu], dtype=float),
        "active_valence_bands": np.asarray([state.active_valence_bands], dtype=int),
        "scheme": np.asarray(str(state.scheme)),
        "n_spin": np.asarray([state.n_spin], dtype=int),
        "n_eta": np.asarray([state.n_eta], dtype=int),
        "n_band": np.asarray([state.n_band], dtype=int),
        "occupation_counts": np.asarray(
            () if state.occupation_counts is None else state.occupation_counts,
            dtype=int,
        ),
        "mu_mev": np.asarray([state.mu], dtype=float),
        "kvec_nm_inv": _complex_to_pairs(basis_data.kvec),
        "k_grid_frac": np.asarray(basis_data.k_grid_frac, dtype=float),
        "band_energies_mev": np.asarray(basis_data.band_energies, dtype=float),
        "active_band_indices": np.asarray(basis_data.active_band_indices, dtype=int),
        "flat_band_indices": np.asarray(basis_data.flat_band_indices, dtype=int),
        "iter_energy_mev": _trace_array(trace, "energy_mev"),
        "iter_err": _trace_array(trace, "err"),
        "iter_oda": _trace_array(trace, "oda"),
    }
    if provenance is not None:
        payload.update(
            {
                "hf_interaction_convention": np.asarray(provenance.convention),
                "hf_quotient_enabled": np.asarray(
                    [provenance.quotient_enabled], dtype=bool
                ),
                "hf_beta": np.asarray([provenance.beta], dtype=float),
                "hf_physical_shifts": np.asarray(
                    provenance.physical_shifts, dtype=int
                ).reshape(-1, 2),
                "zero_literal_q0_fock": np.asarray(
                    [provenance.zero_literal_q0_fock], dtype=bool
                ),
                "hf_basis_periodic_gauge": np.asarray(
                    provenance.basis_periodic_gauge
                ),
                "hf_basis_periodic_gauge_padding": np.asarray(
                    [provenance.basis_periodic_gauge_padding], dtype=int
                ),
                "hf_form_factor_convention": np.asarray(
                    provenance.form_factor_convention
                ),
                "hf_remote_h0_policy": np.asarray(provenance.remote_h0_policy),
                "hf_remote_h0_sha256": np.asarray(provenance.remote_h0_sha256),
                "hf_physical_shift_policy": np.asarray(
                    provenance.physical_shift_policy
                ),
                "hf_provider_fingerprint": np.asarray(
                    provenance.provider_fingerprint
                ),
                "hf_provider_schema_version": np.asarray(
                    [provenance.provider_schema_version], dtype=int
                ),
            }
        )
        if provenance.basis_cache_key:
            payload["cache_key_basis"] = np.asarray(provenance.basis_cache_key)
        if provenance.overlap_cache_key:
            payload["cache_key_overlap"] = np.asarray(provenance.overlap_cache_key)

    for key, value in (cache_metadata or {}).items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"RLG/hBN archive metadata keys must be non-empty strings, got {key!r}"
            )
        if isinstance(value, (str, Path)):
            array = np.asarray(str(value))
        elif value is None:
            array = np.asarray("")
        else:
            array = np.asarray(value)
        field_is_owned = key in payload or key in _PROVENANCE_ARCHIVE_FIELDS
        if field_is_owned:
            same_cache_key = (
                key in {"cache_key_basis", "cache_key_overlap"}
                and key in payload
                and payload[key].size == 1
                and array.size == 1
                and str(payload[key].reshape(-1)[0]) == str(array.reshape(-1)[0])
            )
            if same_cache_key:
                continue
            raise ValueError(
                f"RLG/hBN archive metadata cannot override owned field {key!r}"
            )
        payload[key] = array

    if require_interaction_provenance:
        missing_cache_keys = [
            key
            for key in ("cache_key_basis", "cache_key_overlap")
            if key not in payload or not str(np.asarray(payload[key]).reshape(-1)[0])
        ]
        if missing_cache_keys:
            raise ValueError(
                "Refusing to save a TDHF-source HF archive without cache keys: "
                f"{missing_cache_keys}"
            )

    return write_npz_artifact(payload, path, compressed=True)


__all__ = ["save_rlg_hbn_hf_archive"]
