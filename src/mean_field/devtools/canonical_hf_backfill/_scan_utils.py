from __future__ import annotations

from ._shared import *  # noqa: F401,F403

def _json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _rlg_hbn_cache_manifest_blockers(cache_path: Path, *, cache_kind: str) -> tuple[str, ...]:
    """Return loader-compatibility blockers from a RLG/hBN cache manifest."""

    manifest_path = cache_path / "manifest.json"
    try:
        manifest = _json_load(manifest_path)
    except Exception as exc:  # pragma: no cover - exact filesystem failures vary.
        return (
            f"{cache_kind} cache manifest could not be read: {manifest_path}: "
            f"{type(exc).__name__}: {exc}",
        )
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping):
        return (f"{cache_kind} cache manifest has invalid/missing extra metadata: {manifest_path}",)

    blockers: list[str] = []
    basis_periodic_gauge = extra.get("basis_periodic_gauge")
    if basis_periodic_gauge != _RLG_HBN_EXPECTED_BASIS_PERIODIC_GAUGE:
        blockers.append(
            f"{cache_kind} cache {cache_path} uses incompatible basis_periodic_gauge "
            f"{basis_periodic_gauge!r}; expected {_RLG_HBN_EXPECTED_BASIS_PERIODIC_GAUGE!r}"
        )
    form_factor_convention = extra.get("form_factor_convention")
    if form_factor_convention != _RLG_HBN_EXPECTED_FORM_FACTOR_CONVENTION:
        blockers.append(
            f"{cache_kind} cache {cache_path} uses incompatible form_factor_convention "
            f"{form_factor_convention!r}; expected {_RLG_HBN_EXPECTED_FORM_FACTOR_CONVENTION!r}"
        )
    return tuple(blockers)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_or_empty(value: object) -> str:
    return "" if value is None else str(value)


def _normal_system_name(system_name: str, workflow: str = "") -> str:
    text = f"{system_name} {workflow}".lower().replace("-", "_")
    if "tdbg" in text:
        return "tdbg"
    if "htg_supercell" in text or ("htg" in text and "supercell" in text):
        return "htg_supercell"
    if "rlg_hbn" in text or "rng_hbn" in text or "rng/hbn" in text or "rlg/hbn" in text:
        return "rlg_hbn"
    return system_name.lower()


def _manifest_sidecar_path(root: Path, files: Mapping[str, Any], key: str) -> Path | None:
    raw = files.get(key)
    if not isinstance(raw, str):
        return None
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return root / relative


def _artifact_path(root: Path, files: Mapping[str, Any], key: str, default_name: str) -> Path:
    return _manifest_sidecar_path(root, files, key) or (root / default_name)


def _npz_key_set(path: Path) -> tuple[frozenset[str] | None, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return frozenset(str(key) for key in payload.files), None
    except Exception as exc:  # pragma: no cover - depends on malformed local artifacts.
        return None, f"{type(exc).__name__}: {exc}"


def _missing_file_blocker(path: Path, *, role: str) -> str:
    return f"missing raw file for {role}: {path}"


def _missing_key_blockers(path: Path, missing_keys: Iterable[str], *, role: str) -> tuple[str, ...]:
    return tuple(f"{path} missing key `{key}` required for {role}" for key in sorted(missing_keys))


def _path_text_mentions(path: Path, token: str) -> bool:
    return token.lower() in str(path).lower().replace("-", "_")


def _contract_metadata(*, raw_object: str, state_keys: Iterable[str], basis_keys: Iterable[str]) -> dict[str, object]:
    return {
        "raw_object": raw_object,
        "state_npz_required_keys": sorted(str(key) for key in state_keys),
        "projected_basis_npz_required_keys": sorted(str(key) for key in basis_keys),
        "loader_policy": "metadata-only/raw-archive loader must materialize these fields without SCF, diagonalization, topology, cRPA, or fabricated wavefunctions",
    }

__all__ = [name for name in globals() if not name.startswith('__')]
