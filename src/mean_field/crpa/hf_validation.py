from __future__ import annotations

from typing import Any

import numpy as np

from .workflow import CRPAResult


def _metadata_bool(metadata: dict[str, object], key: str) -> bool:
    if key not in metadata:
        raise ValueError(f"cRPA artifact metadata is missing {key!r}; regenerate it with an HF-compatible cRPA workflow.")
    value = metadata[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"cRPA artifact metadata {key!r} must be boolean, got {value!r}.")


def _metadata_string(metadata: dict[str, object], key: str) -> str:
    if key not in metadata:
        raise ValueError(f"cRPA artifact metadata is missing {key!r}; regenerate it with an HF-compatible cRPA workflow.")
    return str(metadata[key])


def _metadata_float(metadata: dict[str, object], key: str) -> float:
    if key not in metadata:
        raise ValueError(f"cRPA artifact metadata is missing {key!r}; regenerate it with an HF-compatible cRPA workflow.")
    return float(metadata[key])


def has_full_crpa_q_table(crpa_result: CRPAResult) -> bool:
    expected = {(i, j) for i in range(int(crpa_result.lk)) for j in range(int(crpa_result.lk))}
    actual = {tuple(int(v) for v in row) for row in np.asarray(crpa_result.q_indices, dtype=int).tolist()}
    return actual == expected


def validate_hf_compatible_crpa(
    crpa_result: CRPAResult,
    params: Any,
    *,
    theta_deg: float,
    overlap_lg: int,
) -> None:
    if abs(float(crpa_result.theta_deg) - float(theta_deg)) > 1.0e-10:
        raise ValueError(
            f"cRPA theta_deg={crpa_result.theta_deg:.16g} does not match HF theta_deg={theta_deg:.16g}."
        )
    required_q_lg = int(overlap_lg)
    if int(overlap_lg) % 2 == 1:
        # The current B0 HF mesh includes both endpoints, so source-target
        # transfer vectors can add one extra reciprocal shell beyond the
        # explicit overlap shift shell.  For overlap_lg=9 this requires
        # cRPA q_lg>=11 for exact matrix-diagonal lookup.
        required_q_lg = int(overlap_lg) + 2
    if int(crpa_result.q_lg) < required_q_lg:
        raise ValueError(
            f"cRPA q_lg={crpa_result.q_lg} is smaller than required q_lg={required_q_lg} "
            f"for HF overlap_lg={overlap_lg}; "
            "regenerate cRPA with a larger Q cutoff."
        )
    if not has_full_crpa_q_table(crpa_result):
        raise ValueError("cRPA artifact does not contain the full lk x lk q table; regenerate without q stride/max-q truncation.")

    metadata = dict(crpa_result.metadata)
    if not _metadata_bool(metadata, "periodic_g_grid"):
        raise ValueError("This HF code requires cRPA generated with periodic_g_grid=True.")
    if _metadata_string(metadata, "form_factor_mode") != "hf_periodic":
        raise ValueError("This HF code requires cRPA form_factor_mode='hf_periodic'.")
    if _metadata_string(metadata, "occupation_mode") != "cnp_index":
        raise ValueError("This HF code requires cRPA occupation_mode='cnp_index'.")
    if not _metadata_bool(metadata, "sigma_rotation"):
        raise ValueError("This HF code requires cRPA generated with sigma_rotation=True.")
    if _metadata_string(metadata, "flat_band_classifier") != "center":
        raise ValueError("This HF code currently expects cRPA flat_band_classifier='center'.")
    if _metadata_string(metadata, "k_grid_kind") != "uniform_crpa":
        raise ValueError("This HF code currently expects cRPA k_grid_kind='uniform_crpa'.")

    for key, expected in (("vf", params.vf), ("w0", params.w0), ("w1", params.w1)):
        actual = _metadata_float(metadata, key)
        if abs(actual - float(expected)) > 1.0e-9:
            raise ValueError(f"cRPA metadata {key}={actual:.16g} does not match HF {key}={float(expected):.16g}.")
