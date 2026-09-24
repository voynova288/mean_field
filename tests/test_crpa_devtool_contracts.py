from __future__ import annotations

import ast

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.core.io import read_json_artifact, write_json_artifact
from mean_field.crpa.coulomb import CRPACoulombParams
from mean_field.crpa.grid import build_q_shift_table
from mean_field.crpa import chunk_merge
from mean_field.crpa.workflow import CRPAResult, load_crpa_result, write_crpa_outputs
from mean_field.devtools import merge_tbg_crpa_chunks as merge_tool
from mean_field.devtools import run_tbg_crpa_chunk as chunk_tool


def _make_result(
    coords: list[tuple[int, int]],
    *,
    lk: int = 2,
    q_lg: int = 1,
    metadata: dict[str, object] | None = None,
) -> CRPAResult:
    q_indices = np.asarray(coords, dtype=np.int64).reshape((-1, 2))
    nq = q_indices.shape[0]
    q_shifts = build_q_shift_table(q_lg)[1].astype(np.int64)
    nshift = q_shifts.shape[0]
    marker = (q_indices[:, 0] + lk * q_indices[:, 1]).astype(np.float64)
    q_tilde = marker.astype(np.complex128)
    shift_marker = q_shifts[:, 0].astype(np.float64) + 1j * q_shifts[:, 1].astype(np.float64)
    physical_q_vectors = q_tilde[:, None] + shift_marker[None, :]
    matrices = marker[:, None, None] * np.ones((nq, nshift, nshift), dtype=np.complex128)
    effective = (marker[:, None] + 2.0) * np.ones((nq, nshift), dtype=np.float64)
    epsilon = matrices.copy()
    diagonal = np.arange(nshift)
    epsilon[:, diagonal, diagonal] = effective
    return CRPAResult(
        theta_deg=1.05,
        lk=lk,
        lg=3,
        q_lg=q_lg,
        bands_per_valley=None,
        eta_mev=1.0,
        coulomb_params=CRPACoulombParams(),
        q_indices=q_indices,
        q_tilde=q_tilde,
        q_shifts=q_shifts,
        q_vectors=shift_marker.astype(np.complex128),
        physical_q_vectors=physical_q_vectors.astype(np.complex128),
        chi0=matrices.copy(),
        dielectric_matrix=epsilon,
        epsilon_inv=matrices.copy(),
        screened_v=matrices.copy(),
        effective_epsilon=effective,
        metadata={
            "form_factor_mode": "k_periodic_zero_fill",
            **({} if metadata is None else metadata),
        },
    )


def _write_chunk(
    path: Path,
    coords: list[tuple[int, int]],
    *,
    lk: int = 2,
    q_lg: int = 1,
    metadata: dict[str, object] | None = None,
) -> Path:
    result = _make_result(coords, lk=lk, q_lg=q_lg, metadata=metadata)
    return write_crpa_outputs(result, path)


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.array(payload[key], copy=True) for key in payload.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


def _chunk_args(*extra: str):
    return chunk_tool.build_parser().parse_args(
        ["--bm-solution", "bm.npz", "--q-lg", "3", "--output-dir", "out", *extra]
    )


def test_chunk_static_request_and_pure_selection_contracts() -> None:
    args = _chunk_args()
    chunk_tool._validate_crpa_chunk_request(args)
    assert chunk_tool._select_q_flat_range(
        9, q_range=None, chunk_index=None, chunk_count=None
    ) == (0, 9)
    assert chunk_tool._select_q_flat_range(
        9, q_range="2:7", chunk_index=None, chunk_count=None
    ) == (2, 7)
    assert chunk_tool._select_q_flat_range(
        9, q_range=None, chunk_index=1, chunk_count=3
    ) == (3, 6)

    invalid_requests = (
        ("--q-range", "0:2", "--chunk-index", "0", "--chunk-count", "1"),
        ("--chunk-index", "0"),
        ("--chunk-count", "1"),
        ("--q-lg", "2"),
        ("--epsilon-bn", "nan"),
        ("--epsilon-bn", "0"),
        ("--ds-angstrom", "-1"),
        ("--eta-mev", "inf"),
        ("--chi0-eq19-overlap-lg", "3"),
        ("--chi0-energy-mode", "eq19_flat_remote", "--chi0-eq19-overlap-lg", "2"),
    )
    for extra in invalid_requests:
        with pytest.raises(ValueError):
            chunk_tool._validate_crpa_chunk_request(_chunk_args(*extra))

    for selector in (
        {"q_range": "-1:2", "chunk_index": None, "chunk_count": None},
        {"q_range": "0:10", "chunk_index": None, "chunk_count": None},
        {"q_range": None, "chunk_index": 0, "chunk_count": 10},
        {"q_range": None, "chunk_index": 3, "chunk_count": 3},
    ):
        with pytest.raises(ValueError):
            chunk_tool._select_q_flat_range(9, **selector)


def test_chunk_login_guard_precedes_artifacts_and_cache(monkeypatch, tmp_path: Path) -> None:
    class GuardReached(RuntimeError):
        pass

    def stop_at_guard(_name: str) -> None:
        raise GuardReached

    monkeypatch.setattr(chunk_tool, "ensure_not_running_compute_on_login_node", stop_at_guard)
    monkeypatch.setattr(
        chunk_tool,
        "_write_crpa_chunk_workflow_artifacts",
        lambda *_args, **_kwargs: pytest.fail("workflow artifact written before login guard"),
    )
    monkeypatch.setattr(
        chunk_tool,
        "read_all_band_bm_solution",
        lambda *_args, **_kwargs: pytest.fail("BM cache loaded before login guard"),
    )
    output_dir = tmp_path / "must_not_exist"
    with pytest.raises(GuardReached):
        chunk_tool.main(
            [
                "--bm-solution",
                str(tmp_path / "bm.npz"),
                "--q-lg",
                "3",
                "--output-dir",
                str(output_dir),
            ]
        )
    assert not output_dir.exists()


def test_merge_login_guard_precedes_artifacts_and_run(monkeypatch, tmp_path: Path) -> None:
    class GuardReached(RuntimeError):
        pass

    def stop_at_guard(_name: str) -> None:
        raise GuardReached

    monkeypatch.setattr(merge_tool, "ensure_not_running_compute_on_login_node", stop_at_guard)
    monkeypatch.setattr(
        merge_tool,
        "_write_merge_workflow_artifacts",
        lambda *_args, **_kwargs: pytest.fail("workflow artifact written before login guard"),
    )
    monkeypatch.setattr(
        merge_tool,
        "_run_merge",
        lambda *_args, **_kwargs: pytest.fail("merge started before login guard"),
    )
    output_dir = tmp_path / "must_not_exist"
    with pytest.raises(GuardReached):
        merge_tool.main(
            [
                "--output-dir",
                str(output_dir),
                "--chunk",
                str(tmp_path / "chunk"),
            ]
        )
    assert not output_dir.exists()


def test_real_writer_loader_roundtrip(tmp_path: Path) -> None:
    expected = _make_result([(0, 0), (1, 0)], q_lg=3)
    chunk = write_crpa_outputs(expected, tmp_path / "chunk")
    loaded = load_crpa_result(chunk)

    assert loaded.theta_deg == expected.theta_deg
    assert loaded.lk == expected.lk
    assert loaded.lg == expected.lg
    assert loaded.q_lg == expected.q_lg
    assert loaded.bands_per_valley == expected.bands_per_valley
    assert loaded.eta_mev == expected.eta_mev
    assert loaded.coulomb_params == expected.coulomb_params
    assert loaded.metadata == expected.metadata
    for name in (
        "q_indices",
        "q_tilde",
        "q_shifts",
        "q_vectors",
        "physical_q_vectors",
        "chi0",
        "dielectric_matrix",
        "epsilon_inv",
        "screened_v",
        "effective_epsilon",
    ):
        assert np.array_equal(getattr(loaded, name), getattr(expected, name)), name
    assert loaded.q_indices.dtype == np.dtype(np.int64)
    assert loaded.chi0.dtype == np.dtype(np.complex128)
    assert loaded.effective_epsilon.dtype == np.dtype(np.float64)


def test_merge_rejects_duplicate_gap_and_out_of_range_but_allows_partial(tmp_path: Path) -> None:
    duplicate_a = _write_chunk(tmp_path / "duplicate_a", [(0, 0)])
    duplicate_b = _write_chunk(tmp_path / "duplicate_b", [(0, 0)])
    with pytest.raises(ValueError, match="Duplicate q coordinates"):
        chunk_merge.load_and_validate_chunks(
            (duplicate_a, duplicate_b), allow_partial=True
        )

    partial = _write_chunk(tmp_path / "partial", [(0, 0), (1, 0), (0, 1)])
    with pytest.raises(ValueError, match=r"exact lk\^2 coverage"):
        chunk_merge.load_and_validate_chunks((partial,), allow_partial=False)
    _params, arrays, coverage = chunk_merge.load_and_validate_chunks(
        (partial,), allow_partial=True
    )
    assert arrays["q_indices"].shape == (3, 2)
    assert coverage == {
        "allow_partial": True,
        "policy": "nonempty_valid_subset",
        "complete": False,
        "q_point_count": 3,
        "expected_q_point_count": 4,
        "missing_flat_indices": [3],
    }

    out_of_range = _write_chunk(tmp_path / "out_of_range", [(2, 0)])
    with pytest.raises(ValueError, match=r"must lie in \[0, 2\)"):
        chunk_merge.load_and_validate_chunks((out_of_range,), allow_partial=True)


def test_merge_rejects_npz_label_shape_and_coordinate_mismatches(tmp_path: Path) -> None:
    bad_label = _write_chunk(tmp_path / "bad_label", [(0, 0)])
    _rewrite_npz(bad_label / "dielectric_matrix.npz", lambda arrays: arrays.pop("epsilon_inv"))
    with pytest.raises(ValueError, match="NPZ labels mismatch"):
        chunk_merge.load_and_validate_chunks((bad_label,), allow_partial=True)

    bad_shape = _write_chunk(tmp_path / "bad_shape", [(0, 0)])
    _rewrite_npz(
        bad_shape / "chi0_q.npz",
        lambda arrays: arrays.__setitem__("chi0", np.zeros((1, 2, 2), dtype=np.complex128)),
    )
    with pytest.raises(ValueError, match="chi0_q.npz:chi0 has shape"):
        chunk_merge.load_and_validate_chunks((bad_shape,), allow_partial=True)

    bad_coords = _write_chunk(tmp_path / "bad_coords", [(0, 0)])
    _rewrite_npz(
        bad_coords / "effective_epsilon.npz",
        lambda arrays: arrays.__setitem__("q_indices", np.asarray([[1, 0]], dtype=np.int64)),
    )
    with pytest.raises(ValueError, match="inconsistent q_indices"):
        chunk_merge.load_and_validate_chunks((bad_coords,), allow_partial=True)

    bad_dtype = _write_chunk(tmp_path / "bad_dtype", [(0, 0)])
    _rewrite_npz(
        bad_dtype / "screened_coulomb.npz",
        lambda arrays: arrays.__setitem__("q_indices", arrays["q_indices"].astype(float)),
    )
    with pytest.raises(ValueError, match="must have integer dtype"):
        chunk_merge.load_and_validate_chunks((bad_dtype,), allow_partial=True)

    bad_shifts = _write_chunk(tmp_path / "bad_shifts", [(0, 0)])
    _rewrite_npz(
        bad_shifts / "dielectric_matrix.npz",
        lambda arrays: arrays.__setitem__("q_shifts", np.asarray([[1, 0]], dtype=np.int64)),
    )
    with pytest.raises(ValueError, match="inconsistent q_shifts"):
        chunk_merge.load_and_validate_chunks((bad_shifts,), allow_partial=True)


def test_merge_requires_strict_json_and_exact_parameter_types(tmp_path: Path) -> None:
    duplicate = _write_chunk(tmp_path / "duplicate_json", [(0, 0)])
    (duplicate / "crpa_params.json").write_text('{"lk": 2, "lk": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        chunk_merge.load_and_validate_chunks((duplicate,), allow_partial=True)

    nan_json = _write_chunk(tmp_path / "nan_json", [(0, 0)])
    (nan_json / "crpa_params.json").write_text('{"theta_deg": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="nonstandard JSON constant"):
        chunk_merge.load_and_validate_chunks((nan_json,), allow_partial=True)

    bool_int = _write_chunk(tmp_path / "bool_int", [(0, 0)])
    params = read_json_artifact(bool_int / "crpa_params.json")
    assert isinstance(params, dict)
    params["lk"] = True
    write_json_artifact(params, bool_int / "crpa_params.json")
    with pytest.raises(ValueError, match="lk must be an integer"):
        chunk_merge.load_and_validate_chunks((bool_int,), allow_partial=True)

    bad_coulomb = _write_chunk(tmp_path / "bad_coulomb", [(0, 0)])
    params = read_json_artifact(bad_coulomb / "crpa_params.json")
    assert isinstance(params, dict) and isinstance(params["coulomb_params"], dict)
    params["coulomb_params"]["unexpected"] = 1.0
    write_json_artifact(params, bad_coulomb / "crpa_params.json")
    with pytest.raises(ValueError, match="coulomb_params schema mismatch"):
        chunk_merge.load_and_validate_chunks((bad_coulomb,), allow_partial=True)


def test_merge_rejects_nonfinite_and_lossy_physical_array_dtypes(tmp_path: Path) -> None:
    nonfinite = _write_chunk(tmp_path / "nonfinite", [(0, 0)])

    def put_nan(arrays: dict[str, np.ndarray]) -> None:
        arrays["q_real"][0, 0] = np.nan

    _rewrite_npz(nonfinite / "effective_epsilon.npz", put_nan)
    with pytest.raises(ValueError, match="must contain only finite values"):
        chunk_merge.load_and_validate_chunks((nonfinite,), allow_partial=True)

    integer_real = _write_chunk(tmp_path / "integer_real", [(0, 0)])
    _rewrite_npz(
        integer_real / "effective_epsilon.npz",
        lambda arrays: arrays.__setitem__("q_abs", arrays["q_abs"].astype(np.int64)),
    )
    with pytest.raises(ValueError, match="must have floating dtype"):
        chunk_merge.load_and_validate_chunks((integer_real,), allow_partial=True)

    real_complex = _write_chunk(tmp_path / "real_complex", [(0, 0)])
    _rewrite_npz(
        real_complex / "dielectric_matrix.npz",
        lambda arrays: arrays.__setitem__("epsilon", arrays["epsilon"].real.astype(np.float64)),
    )
    with pytest.raises(ValueError, match="must have complex dtype"):
        chunk_merge.load_and_validate_chunks((real_complex,), allow_partial=True)


def test_merge_requires_canonical_q_shift_inventory_even_when_all_npzs_agree(tmp_path: Path) -> None:
    chunk = _write_chunk(tmp_path / "chunk", [(0, 0)], q_lg=3)
    noncanonical = np.roll(build_q_shift_table(3)[1], 1, axis=0).astype(np.int64)
    for filename in (
        "chi0_q.npz",
        "dielectric_matrix.npz",
        "effective_epsilon.npz",
        "screened_coulomb.npz",
    ):
        _rewrite_npz(
            chunk / filename,
            lambda arrays, value=noncanonical: arrays.__setitem__("q_shifts", value),
        )
    with pytest.raises(ValueError, match="does not match canonical build_q_shift_table"):
        chunk_merge.load_and_validate_chunks((chunk,), allow_partial=True)


@pytest.mark.parametrize(
    ("filename", "key", "expected"),
    (
        ("screened_coulomb.npz", "effective_epsilon", "screened effective_epsilon"),
        ("screened_coulomb.npz", "q_abs_nm_inv", "screened q_abs_nm_inv"),
        ("screened_coulomb.npz", "q_vectors_real", "screened q_vectors_real"),
        ("screened_coulomb.npz", "q_vectors_imag", "screened q_vectors_imag"),
        ("effective_epsilon.npz", "epsilon_times_bn", "epsilon_times_bn"),
        ("effective_epsilon.npz", "q_abs_nm_inv", "q_abs_nm_inv"),
        ("effective_epsilon.npz", "q_abs", "q_abs == hypot"),
    ),
)
def test_merge_rejects_redundant_field_tampering(
    tmp_path: Path,
    filename: str,
    key: str,
    expected: str,
) -> None:
    chunk = _write_chunk(tmp_path / f"tamper_{key}", [(1, 0)])

    def tamper(arrays: dict[str, np.ndarray]) -> None:
        arrays[key] = arrays[key].copy()
        arrays[key][0, 0] += 0.25

    _rewrite_npz(chunk / filename, tamper)
    with pytest.raises(ValueError, match=expected):
        chunk_merge.load_and_validate_chunks((chunk,), allow_partial=True)


def test_merge_rejects_epsilon_diagonal_and_zero_shift_q_tilde_tampering(
    tmp_path: Path,
) -> None:
    bad_epsilon = _write_chunk(tmp_path / "bad_epsilon", [(1, 0)], q_lg=3)

    def tamper_epsilon(arrays: dict[str, np.ndarray]) -> None:
        arrays["epsilon"] = arrays["epsilon"].copy()
        arrays["epsilon"][0, 0, 0] += 0.25

    _rewrite_npz(bad_epsilon / "dielectric_matrix.npz", tamper_epsilon)
    with pytest.raises(ValueError, match=r"effective_epsilon == real\(diag\(epsilon\)\)"):
        chunk_merge.load_and_validate_chunks((bad_epsilon,), allow_partial=True)

    bad_q_tilde = _write_chunk(tmp_path / "bad_q_tilde", [(1, 0)], q_lg=3)

    def tamper_q_tilde(arrays: dict[str, np.ndarray]) -> None:
        arrays["q_tilde_real"] = arrays["q_tilde_real"].copy()
        arrays["q_tilde_real"][0] += 0.25

    _rewrite_npz(bad_q_tilde / "chi0_q.npz", tamper_q_tilde)
    with pytest.raises(ValueError, match="q_tilde_real == q_real at canonical zero shift"):
        chunk_merge.load_and_validate_chunks((bad_q_tilde,), allow_partial=True)


def test_merge_rejects_cross_chunk_metadata_mismatch(tmp_path: Path) -> None:
    first = _write_chunk(tmp_path / "metadata_a", [(0, 0)])
    second = _write_chunk(tmp_path / "metadata_b", [(1, 0)])
    params_path = second / "crpa_params.json"
    params = read_json_artifact(params_path)
    assert isinstance(params, dict) and isinstance(params["metadata"], dict)
    params["metadata"]["form_factor_mode"] = "hf_periodic"
    write_json_artifact(params, params_path)

    with pytest.raises(ValueError, match="incompatible metadata"):
        chunk_merge.load_and_validate_chunks((first, second), allow_partial=True)


def test_merge_aggregates_chunk_local_dynamic_metadata(tmp_path: Path) -> None:
    first = _write_chunk(
        tmp_path / "metadata_a",
        [(0, 0)],
        metadata={
            "k_periodic_max_wrap_shell": 1,
            "periodic_roll_required_lg": 3,
            "periodic_roll_no_alias": True,
        },
    )
    second = _write_chunk(
        tmp_path / "metadata_b",
        [(1, 0)],
        metadata={
            "k_periodic_max_wrap_shell": 2,
            "periodic_roll_required_lg": 5,
            "periodic_roll_no_alias": False,
        },
    )

    params, _arrays, _coverage = chunk_merge.load_and_validate_chunks(
        (first, second), allow_partial=True
    )
    assert params["metadata"] == {
        "form_factor_mode": "k_periodic_zero_fill",
        "legacy_zero_fill_test": False,
        "k_periodic_max_wrap_shell": 2,
        "periodic_roll_required_lg": 5,
        "periodic_roll_no_alias": False,
    }


def test_merge_rejects_missing_or_inconsistent_dynamic_metadata_triple(tmp_path: Path) -> None:
    missing = _write_chunk(
        tmp_path / "missing_dynamic",
        [(0, 0)],
        metadata={"k_periodic_max_wrap_shell": 1},
    )
    with pytest.raises(ValueError, match="dynamic metadata triple must be all missing or all present"):
        chunk_merge.load_and_validate_chunks((missing,), allow_partial=True)

    invalid_required = _write_chunk(
        tmp_path / "invalid_required",
        [(0, 0)],
        metadata={
            "k_periodic_max_wrap_shell": 1,
            "periodic_roll_required_lg": 5,
            "periodic_roll_no_alias": False,
        },
    )
    with pytest.raises(ValueError, match="must equal owner formula"):
        chunk_merge.load_and_validate_chunks((invalid_required,), allow_partial=True)

    invalid_alias = _write_chunk(
        tmp_path / "invalid_alias",
        [(0, 0)],
        metadata={
            "k_periodic_max_wrap_shell": 1,
            "periodic_roll_required_lg": 3,
            "periodic_roll_no_alias": False,
        },
    )
    with pytest.raises(ValueError, match=r"must equal \(lg >= periodic_roll_required_lg\)"):
        chunk_merge.load_and_validate_chunks((invalid_alias,), allow_partial=True)

    invalid_wrap_type = _write_chunk(
        tmp_path / "invalid_wrap_type",
        [(0, 0)],
        metadata={
            "k_periodic_max_wrap_shell": True,
            "periodic_roll_required_lg": 3,
            "periodic_roll_no_alias": True,
        },
    )
    with pytest.raises(ValueError, match="max_wrap_shell must be an integer"):
        chunk_merge.load_and_validate_chunks((invalid_wrap_type,), allow_partial=True)


def test_merge_uses_stable_canonical_flat_sort(tmp_path: Path) -> None:
    first = _write_chunk(tmp_path / "first", [(1, 1), (0, 0)])
    second = _write_chunk(tmp_path / "second", [(1, 0), (0, 1)])

    _params, arrays, coverage = chunk_merge.load_and_validate_chunks(
        (first, second), allow_partial=False
    )

    assert arrays["q_indices"].tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert arrays["q_tilde_real"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert coverage["complete"] is True


def test_multi_chunk_q_lg_greater_than_one_merges_successfully(tmp_path: Path) -> None:
    first = _write_chunk(tmp_path / "qlg3_first", [(1, 1), (0, 0)], q_lg=3)
    second = _write_chunk(tmp_path / "qlg3_second", [(1, 0), (0, 1)], q_lg=3)

    params, arrays, coverage = chunk_merge.load_and_validate_chunks(
        (first, second), allow_partial=False
    )

    assert params["q_lg"] == 3
    assert arrays["q_shifts"].shape == (9, 2)
    assert arrays["effective_epsilon"].shape == (4, 9)
    assert arrays["q_indices"].tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert coverage["complete"] is True


def test_package_merge_owner_writes_real_writer_compatible_base_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    chunk = _write_chunk(tmp_path / "chunk", [(0, 0), (1, 0), (0, 1), (1, 1)])
    output = tmp_path / "merged"
    params, arrays, coverage = chunk_merge.load_and_validate_chunks(
        (chunk,), allow_partial=False
    )
    chunk_merge.write_merged_crpa_base_artifacts(params, arrays, coverage, output)

    expected_keys = {
        "crpa_params.json": frozenset(
            {
                "theta_deg",
                "lk",
                "lg",
                "q_lg",
                "bands_per_valley",
                "eta_mev",
                "coulomb_params",
                "q_point_count",
                "q_shift_count",
                "metadata",
            }
        ),
        "chi0_q.npz": frozenset(
            {"chi0", "q_indices", "q_tilde_real", "q_tilde_imag", "q_shifts"}
        ),
        "dielectric_matrix.npz": frozenset(
            {"epsilon", "epsilon_inv", "q_indices", "q_shifts"}
        ),
        "effective_epsilon.npz": frozenset(
            {
                "effective_epsilon",
                "epsilon_times_bn",
                "q_abs",
                "q_abs_nm_inv",
                "q_real",
                "q_imag",
                "q_indices",
                "q_shifts",
            }
        ),
        "screened_coulomb.npz": frozenset(
            {
                "screened_v",
                "effective_epsilon",
                "q_indices",
                "q_shifts",
                "q_abs_nm_inv",
                "q_vectors_real",
                "q_vectors_imag",
            }
        ),
    }
    assert all((output / name).is_file() for name in expected_keys)
    params = read_json_artifact(output / "crpa_params.json")
    assert isinstance(params, dict)
    assert frozenset(params) == expected_keys["crpa_params.json"]
    assert params["metadata"]["merge_coverage"]["complete"] is True
    for filename, keys in expected_keys.items():
        if filename == "crpa_params.json":
            continue
        with np.load(output / filename, allow_pickle=False) as payload:
            assert frozenset(payload.files) == keys
            assert payload["q_indices"].dtype == np.dtype(np.int64)
            assert payload["q_shifts"].dtype == np.dtype(np.int64)
            assert payload["q_shifts"].tolist() == [[0, 0]]
    with np.load(output / "chi0_q.npz", allow_pickle=False) as payload:
        assert payload["chi0"].dtype == np.dtype(np.complex128)
        assert payload["q_tilde_real"].dtype == np.dtype(np.float64)
        assert payload["q_indices"].tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    with np.load(output / "effective_epsilon.npz", allow_pickle=False) as payload:
        assert payload["effective_epsilon"].dtype == np.dtype(np.float64)
        assert payload["effective_epsilon"][:, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
        assert np.array_equal(payload["epsilon_times_bn"], 4.0 * payload["effective_epsilon"])
    with np.load(output / "screened_coulomb.npz", allow_pickle=False) as payload:
        assert payload["screened_v"].dtype == np.dtype(np.complex128)
        assert payload["q_vectors_real"].dtype == np.dtype(np.float64)
    loaded = load_crpa_result(output)
    assert loaded.q_indices.tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert loaded.effective_epsilon[:, 0].tolist() == [2.0, 3.0, 4.0, 5.0]



def test_recursive_partial_merge_rewrites_coverage_and_compares_nested_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata_a = {
        "diagnostics": {
            "residuals": [1.25, {"maximum": 2.5}],
            "count": 2,
            "enabled": True,
            "label": "writer",
        }
    }
    metadata_b = {
        "diagnostics": {
            "residuals": [1.25 + 5.0e-13, {"maximum": 2.5 - 5.0e-13}],
            "count": 2,
            "enabled": True,
            "label": "writer",
        }
    }
    first = _write_chunk(tmp_path / "first", [(0, 0), (1, 0)], metadata=metadata_a)
    second = _write_chunk(tmp_path / "second", [(0, 1), (1, 1)], metadata=metadata_b)
    monkeypatch.setattr(
        merge_tool,
        "write_all_epsilon_diagnostics",
        lambda *_args, **_kwargs: SimpleNamespace(
            q_peak_nm_inv=0.0,
            eps_total_peak=0.0,
            eps_total_q12=0.0,
            eps_diag_imag_max_abs=0.0,
        ),
    )

    partial_output = tmp_path / "partial_output"
    partial_args = merge_tool.build_parser().parse_args(
        ["--output-dir", str(partial_output), "--chunk", str(first), "--allow-partial"]
    )
    merge_tool._run_merge(partial_args)
    partial_params = read_json_artifact(partial_output / "crpa_params.json")
    assert partial_params["metadata"]["merge_coverage"]["complete"] is False

    final_output = tmp_path / "final_output"
    final_args = merge_tool.build_parser().parse_args(
        [
            "--output-dir",
            str(final_output),
            "--chunk",
            str(partial_output),
            "--chunk",
            str(second),
        ]
    )
    merge_tool._run_merge(final_args)
    final_params = read_json_artifact(final_output / "crpa_params.json")
    assert final_params["metadata"]["merge_coverage"]["complete"] is True
    assert final_params["metadata"]["merge_coverage"]["q_point_count"] == 4
    assert final_params["metadata"]["diagnostics"] == metadata_a["diagnostics"]

    incompatible = _write_chunk(
        tmp_path / "incompatible_type",
        [(0, 1)],
        metadata={
            "diagnostics": {
                "residuals": [1, {"maximum": 2.5}],
                "count": 2,
                "enabled": True,
                "label": "writer",
            }
        },
    )
    with pytest.raises(ValueError, match="incompatible metadata"):
        chunk_merge.load_and_validate_chunks((first, incompatible), allow_partial=True)


def test_merge_workflow_records_commands_provenance_and_pending_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    launch_cwd = tmp_path / "outside_repo"
    launch_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)
    chunk = Path("chunk")
    parsed_default = merge_tool.build_parser().parse_args(
        ["--output-dir", "default", "--chunk", str(chunk)]
    )
    assert parsed_default.allow_partial is False

    manifest = merge_tool._merge_workflow_manifest(
        Path("out"), (chunk,), allow_partial=True
    )
    input_job = next(job for job in manifest.jobs if job.name.startswith("input_chunk_"))
    merge_job = next(job for job in manifest.jobs if job.name == "merge")
    pending = merge_tool._merge_workflow_state(manifest, "running")
    pending_input = next(job for job in pending.jobs if job.name == input_job.name)
    validated = merge_tool._merge_workflow_state(
        manifest, "running", inputs_validated=True
    )
    validated_input = next(job for job in validated.jobs if job.name == input_job.name)

    assert manifest.metadata["allow_partial"] is True
    assert manifest.metadata["coverage_policy"] == "nonempty_valid_subset"
    assert merge_job.metadata["coverage_policy"] == "nonempty_valid_subset"
    assert merge_job.command[:3] == (
        sys.executable,
        str(merge_tool._DEVTOOL_DISPATCHER),
        "merge_tbg_crpa_chunks",
    )
    assert Path(merge_job.command[1]).is_absolute()
    assert Path(merge_job.command[merge_job.command.index("--output-dir") + 1]).is_absolute()
    assert Path(merge_job.command[merge_job.command.index("--chunk") + 1]).is_absolute()
    assert merge_job.metadata["working_directory"] == str(launch_cwd.resolve())
    assert merge_job.metadata["launch_cwd"] == str(launch_cwd.resolve())
    assert merge_job.metadata["replay_command_owner"] == str(merge_tool._DEVTOOL_DISPATCHER)
    assert "--allow-partial" in merge_job.command
    assert input_job.command == ("provenance-only", str((launch_cwd / "chunk").resolve()))
    assert input_job.metadata["provenance_only"] is True
    assert input_job.metadata["executable"] is False
    assert pending_input.status == "pending"
    assert pending_input.message == "awaiting input validation"
    assert validated_input.status == "succeeded"
    assert validated_input.metadata["provenance_only"] is True
    assert validated_input.metadata["executable"] is False

    chunk_manifest = chunk_tool._crpa_chunk_workflow_manifest(_chunk_args())
    chunk_job = chunk_manifest.jobs[0]
    assert chunk_job.command[:3] == (
        sys.executable,
        str(chunk_tool._DEVTOOL_DISPATCHER),
        "run_tbg_crpa_chunk",
    )
    assert chunk_job.command[chunk_job.command.index("--epsilon-bn") + 1] == repr(4.0)
    assert Path(chunk_job.command[chunk_job.command.index("--bm-solution") + 1]).is_absolute()
    assert Path(chunk_job.command[chunk_job.command.index("--output-dir") + 1]).is_absolute()
    assert chunk_job.metadata["working_directory"] == str(launch_cwd.resolve())
    assert chunk_job.metadata["launch_cwd"] == str(launch_cwd.resolve())
    assert chunk_job.metadata["replay_command_owner"] == str(chunk_tool._DEVTOOL_DISPATCHER)

def test_chunk_merge_owner_has_exact_public_surface_and_layering() -> None:
    owner_path = Path(chunk_merge.__file__).resolve()
    owner_tree = ast.parse(owner_path.read_text(encoding="utf-8"))
    all_assignments = [
        node
        for node in owner_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    ]
    assert len(all_assignments) == 1
    assert ast.literal_eval(all_assignments[0].value) == [
        "coverage_policy",
        "load_and_validate_chunks",
        "write_merged_crpa_base_artifacts",
    ]
    forbidden_import_roots = {"argparse", "matplotlib", "mean_field.devtools", "mean_field.workflows"}
    imported = set()
    for node in ast.walk(owner_tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not {
        name for name in imported if any(name == root or name.startswith(root + ".") for root in forbidden_import_roots)
    }


def test_merge_devtool_does_not_redefine_package_owner_symbols() -> None:
    devtool_tree = ast.parse(Path(merge_tool.__file__).read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in devtool_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assigned = {
        target.id
        for node in devtool_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ((node.targets if isinstance(node, ast.Assign) else [node.target]))
        if isinstance(target, ast.Name)
    }
    forbidden = {
        "_ValidatedChunk",
        "_REQUIRED_FILES",
        "_REQUIRED_PARAM_KEYS",
        "_COULOMB_PARAM_KEYS",
        "_DYNAMIC_METADATA_KEYS",
        "_MERGE_LOCAL_METADATA_KEYS",
        "_NPZ_KEYS",
        "_coverage_policy",
        "coverage_policy",
        "_load_and_validate_chunks",
        "load_and_validate_chunks",
        "write_merged_crpa_base_artifacts",
        "_save_npz",
    }
    assert not ((defined | assigned) & forbidden)
