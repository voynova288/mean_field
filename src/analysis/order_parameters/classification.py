from __future__ import annotations

from collections.abc import Mapping


def classify_tdbg_flavor_state(numeric: Mapping[str, float], *, ivc_threshold: float = 0.15, polarization_threshold: float = 1.2) -> str:
    """Historical TDBG projected-HF flavor classification thresholds."""

    if abs(float(numeric.get("ivc_amplitude", 0.0))) > float(ivc_threshold):
        return "IVC_or_valley_coherent"
    valley = float(numeric.get("cb_valley_polarization", numeric.get("active_valley_polarization", 0.0)))
    if abs(valley) > float(polarization_threshold):
        return "VP_K" if valley > 0 else "VP_Kprime"
    spin = float(numeric.get("cb_spin_polarization", numeric.get("active_spin_polarization", 0.0)))
    if abs(spin) > float(polarization_threshold):
        return "SP_up" if spin > 0 else "SP_down"
    return "mixed"


__all__ = ["classify_tdbg_flavor_state"]
