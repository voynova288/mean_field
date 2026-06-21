from __future__ import annotations

from pathlib import Path

import numpy as np

from mean_field.api.artifacts import ModelRecord, write_contract_artifacts
from mean_field.core.hf import check_single_flavor_simplification

# Historical dense RnG/hBN q=0 TDHF runner was retired from tracked command
# surface and archived under
# local_archive/retired_surface/7606a1a_rlg_hbn_tdhf_q0_devtool/.
# Keep only lightweight compatibility helpers used by schema/adapter tests.


def _flavor_counts(state) -> dict[tuple[int, int], int]:
    if getattr(state, "occupation_counts", None) is None:
        return {}
    counts = np.asarray(state.occupation_counts, dtype=int).reshape((int(state.n_spin), int(state.n_eta)), order="C")
    return {(int(s), int(e)): int(counts[s, e]) for s in range(counts.shape[0]) for e in range(counts.shape[1])}


def _shortcut_decision(state, mode: str, channel: str) -> tuple[bool, str]:
    if channel == "all":
        return False, "single-flavor shortcut is not applied to mixed all-channel blocks"
    if channel == "intraflavor":
        return False, "single-flavor shortcut is not applied to intraflavor blocks"
    if mode == "off":
        return False, "disabled by --single-flavor-shortcut=off"
    counts = _flavor_counts(state)
    if not counts:
        status = check_single_flavor_simplification(active_space_has_valence=bool(int(state.active_valence_bands) > 0), occupied_flavor_counts={}, polarized_flavor=(0, 0))
        if mode == "on":
            raise ValueError("--single-flavor-shortcut=on requires saved occupation_counts metadata")
        return False, status.reason
    polarized_candidates = [flavor for flavor, count in counts.items() if int(count) > 0]
    polarized = polarized_candidates[0] if polarized_candidates else next(iter(counts))
    status = check_single_flavor_simplification(active_space_has_valence=bool(int(state.active_valence_bands) > 0), occupied_flavor_counts=counts, polarized_flavor=polarized)
    if mode == "on" and not status.allowed:
        raise ValueError(f"single-flavor shortcut requested but illegal: {status.reason}")
    return bool(status.allowed and mode in {"auto", "on"}), status.reason


def _write_contract_sidecars(output_dir: Path, *, config_payload: dict[str, object], summary_payload: dict[str, object] | None, spectrum_path: Path | None) -> dict[str, Path]:
    """Write public contract sidecars for a historical RLG/hBN q=0 TDHF result."""

    hf_archive = str(config_payload.get("hf_archive", ""))
    channel = str(config_payload.get("channel", "all"))
    files: dict[str, object] = {"tdhf_config": "tdhf_q0_config.json", "source_hf_archive": hf_archive}
    array_files: tuple[str | Path, ...] = ()
    if summary_payload is None:
        validation_payload = {"status": "dry_run", "source_hf_converged": config_payload.get("summary_converged")}
        observables_payload: dict[str, object] = {}
    else:
        files["tdhf_summary"] = "tdhf_q0_summary.json"
        if spectrum_path is not None:
            files["tdhf_spectrum"] = spectrum_path.name
            array_files = (spectrum_path,)
        structure = dict(summary_payload.get("structure", {})) if isinstance(summary_payload.get("structure"), dict) else {}
        spectrum = dict(summary_payload.get("spectrum", {})) if isinstance(summary_payload.get("spectrum"), dict) else {}
        source_converged = config_payload.get("summary_converged")
        validation_payload = {
            "status": "pass" if bool(structure.get("ok", False)) and source_converged is not False else "warning",
            "source_hf_converged": source_converged,
            "structure": structure,
            "spectrum_residuals": {"max_residual": spectrum.get("max_residual"), "pairing_residual": spectrum.get("pairing_residual"), "selected_count": spectrum.get("selected_count")},
            "dense_memory_guard": {"max_dense_memory_gb": config_payload.get("max_dense_memory_gb"), "estimated_dense_memory_gib": summary_payload.get("estimated_dense_memory_gib"), "max_pairs": config_payload.get("max_pairs"), "n_pairs": summary_payload.get("n_pairs")},
        }
        observables_payload = {
            "channel": summary_payload.get("channel"),
            "channel_counts": summary_payload.get("channel_counts"),
            "n_pairs": summary_payload.get("n_pairs"),
            "liouvillian_dim": summary_payload.get("liouvillian_dim"),
            "first_positive_energies_mev": spectrum.get("first_positive_energies_mev"),
            "single_flavor_shortcut_used": summary_payload.get("single_flavor_shortcut_used"),
            "single_flavor_shortcut_reason": summary_payload.get("single_flavor_shortcut_reason"),
            "hf_summary": summary_payload.get("hf_summary"),
        }
    return write_contract_artifacts(
        output_dir,
        workflow="rlg_hbn.tdhf_q0",
        system_name="rlg_hbn",
        model=ModelRecord(system_name="rlg_hbn", params={"source_hf_archive": hf_archive}),
        config=config_payload,
        conventions={"energy_unit": "meV", "density_convention": "stored_delta", "density_axis_order": "abk", "q_sector": "q0", "tdhf_pair_convention": "core_hf_tdhf", "channel": channel, "source_hf_archive": hf_archive},
        environment=dict(config_payload.get("runtime", {})) if isinstance(config_payload.get("runtime"), dict) else {},
        validation=validation_payload,
        observables=observables_payload,
        files=files,
        metadata={"source_hf_archive": hf_archive, "channel": channel},
        array_files=array_files,
    )


def main() -> None:
    raise SystemExit(
        "run_rlg_hbn_tdhf_q0 was retired from the tracked command surface. "
        "Consult local_archive/retired_surface/7606a1a_rlg_hbn_tdhf_q0_devtool/ or git history if needed."
    )


__all__ = ["_flavor_counts", "_shortcut_decision", "_write_contract_sidecars", "main"]
