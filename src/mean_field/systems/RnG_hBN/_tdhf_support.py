from __future__ import annotations

from ._tdhf_shared import *  # noqa: F401,F403

@dataclass(frozen=True)
class RLGhBNTDHFFiniteQSupport:
    """Introspection record for the currently implemented RLG/hBN finite-q TDHF modes.

    The canonical TDHF boundary only normalizes HF orbitals.  Whether finite-q
    direct terms, B terms, q/-q pair sectors, and the system-specific ``V_hf``
    are valid is decided in this RLG/hBN system layer.
    """

    supported: bool
    channel: str
    canonical_boundary: bool
    shortcut_exchange_only: bool
    supported_terms: tuple[str, ...]
    unsupported_terms: tuple[str, ...]
    runtime_guards: tuple[str, ...]
    blockers: tuple[str, ...]
    evidence: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "supported": bool(self.supported),
            "channel": self.channel,
            "canonical_boundary": bool(self.canonical_boundary),
            "shortcut_exchange_only": bool(self.shortcut_exchange_only),
            "supported_terms": list(self.supported_terms),
            "unsupported_terms": list(self.unsupported_terms),
            "runtime_guards": list(self.runtime_guards),
            "blockers": list(self.blockers),
            "evidence": list(self.evidence),
            "reason": self.reason,
        }

def rlg_hbn_tdhf_finite_q_mode_support(
    channel: str,
    *,
    shortcut_exchange_only: bool = True,
    canonical_boundary: bool = False,
) -> RLGhBNTDHFFiniteQSupport:
    """Describe whether an RLG/hBN finite-q TDHF mode is implemented.

    This helper is intentionally conservative: it reports only the legacy
    system code paths that actually exist.  In particular, a canonical HF input
    does not supply finite-q direct/B-term formulas or construct ``V_hf``; it
    only supplies parity-checked HF orbitals before the system adapter builds
    the already-implemented flavor-flip exchange shortcut.
    """

    channel_key = str(channel)
    blockers: list[str] = []
    unsupported_terms = (
        ()
        if channel_key == "intraflavor"
        else (
            "finite_q_A_direct",
            "finite_q_B_direct",
            "finite_q_B_exchange",
            "finite_q_all_channel",
        )
    )
    runtime_guards = (
        "conduction_only_active_space",
        "saved_occupation_counts",
        "exactly_one_occupied_spin_valley_flavor",
        "flavor_flip_pairs_only",
        "complete_wrapped_umklapp_overlap_shifts",
        "canonical_orbital_legacy_parity" if canonical_boundary else "legacy_orbital_builder",
    )
    evidence = (
        "finite-q flavor-flip RLG/hBN assembly is build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs",
        "that shortcut assembly sets B=0 and includes only one-body plus A-exchange for flavor-flip sectors",
        "finite-q intraflavor assembly is build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs",
        "the intraflavor assembly implements Eq. D19 q/-q X/Y bookkeeping and reduces to q=0 direct/exchange/B assembly at q=0",
        "V_hf construction remains system-specific; the canonical boundary is only an orbital normalizer",
    )

    if channel_key == "all":
        blockers.append(
            "all-channel finite-q blocks mix flavor-flip and intraflavor sectors; use separated finite-q "
            "intraflavor/intervalley/interspin blocks."
        )
    elif channel_key == "intraflavor":
        pass
    elif channel_key in FINITE_Q_SHORTCUT_CHANNELS:
        if not bool(shortcut_exchange_only):
            blockers.append(
                "shortcut_exchange_only=False requests full finite-q direct/B terms for a flavor-flip channel; "
                "for the fully polarized conduction-only RLG/hBN sectors these terms vanish and the implemented "
                "path is the guarded exchange shortcut."
            )
    else:
        blockers.append(
            f"unknown finite-q channel {channel_key!r}; expected one of {FINITE_Q_KNOWN_CHANNELS}."
        )

    supported = not blockers
    boundary = "canonical" if canonical_boundary else "legacy"
    if supported and channel_key == "intraflavor":
        supported_terms = (
            "hf_energy_difference",
            "finite_q_A_direct",
            "finite_q_A_exchange",
            "finite_q_B_direct",
            "finite_q_B_exchange",
        )
        reason = (
            f"RLG/hBN {boundary} finite-q TDHF supports channel='intraflavor' through the full Eq. D19 "
            "q/-q X/Y bookkeeping: X uses d†_{k+q,p} d_{k,h}, while Y uses d†_{k,h} d_{k-q,p}."
        )
    elif supported:
        supported_terms = ("hf_energy_difference", "finite_q_A_exchange")
        reason = (
            f"RLG/hBN {boundary} finite-q TDHF supports channel={channel_key!r} only through the "
            "conduction-only, fully spin-valley-polarized, flavor-flip exchange shortcut; runtime guards still "
            "validate the active space, occupation_counts, pair flavors, and wrapped Umklapp cache coverage."
        )
    else:
        supported_terms = ()
        reason = (
            f"RLG/hBN {boundary} finite-q TDHF mode is not supported for channel={channel_key!r}, "
            f"shortcut_exchange_only={bool(shortcut_exchange_only)}. " + " ".join(blockers)
        )

    return RLGhBNTDHFFiniteQSupport(
        supported=bool(supported),
        channel=channel_key,
        canonical_boundary=bool(canonical_boundary),
        shortcut_exchange_only=bool(shortcut_exchange_only),
        supported_terms=supported_terms,
        unsupported_terms=unsupported_terms,
        runtime_guards=runtime_guards,
        blockers=tuple(blockers),
        evidence=evidence,
        reason=reason,
    )

def _require_rlg_hbn_tdhf_finite_q_mode_supported(
    channel: str,
    *,
    shortcut_exchange_only: bool,
    canonical_boundary: bool,
) -> RLGhBNTDHFFiniteQSupport:
    support = rlg_hbn_tdhf_finite_q_mode_support(
        channel,
        shortcut_exchange_only=shortcut_exchange_only,
        canonical_boundary=canonical_boundary,
    )
    if support.supported:
        return support
    if support.channel not in FINITE_Q_KNOWN_CHANNELS:
        raise ValueError(support.reason)
    raise NotImplementedError(support.reason)


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _reject_zero_literal_q0_fock_env() -> None:
    if _env_flag_enabled("MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK", default=False):
        raise ValueError(
            "RLG/hBN TDHF does not support MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=1; "
            "rerun/load HF with the physical q=0 Fock convention before TDHF."
        )

__all__ = [name for name in globals() if not name.startswith('__')]
