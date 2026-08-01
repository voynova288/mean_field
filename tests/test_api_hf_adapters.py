from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import mean_field.api.hf as hf_api
from mean_field.api import HFConfig, HFResult, make_model, run_hf
from mean_field.api.hf import get_hf_adapter_info, list_hf_adapters, resolve_hf_adapter
from mean_field.core.contracts import HFRunResult as ContractHFRunResult
from mean_field.systems import tdbg as tdbg_system
from mean_field.systems.RnG_hBN import RLGhBNInteractionParams, RLGhBNRunHFConfig
from mean_field.systems.htg import HTGRunHFConfig, HTGSupercellRunHFConfig, InteractionParams
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import (
    BMSolution,
    TBGZeroFieldInteractionSpec,
    TBGZeroFieldRunHFConfig,
    solve_bm_model_on_torus,
    tbg_zero_field_hf_run_to_hf_result,
)
from mean_field.systems.tdbg import TDBGInteractionSettings, TDBGProjectedHFConfig, TDBGProjectedWindow


def _tiny_tdbg_config() -> TDBGProjectedHFConfig:
    return TDBGProjectedHFConfig(
        theta_deg=1.38,
        cut=1.0,
        mesh_size=1,
        paper_ud_ev=0.09,
        paper_ud_convention="minus_xi_ud_over3",
        window=TDBGProjectedWindow("two_flat"),
        filling=2,
        interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        precision=1.0e-7,
        max_iter=1,
    )


_TINY_TBG_GRID_CACHE: dict[float, BMSolution] = {}
_TINY_TBG_ARRAY_FIELDS = (
    "lattice_kvec",
    "hamiltonian",
    "sigma_z",
    "uk",
    "spectrum",
    "gvec",
)

def _tiny_tbg_grid_solution(theta_deg: float = 1.05) -> BMSolution:
    key = float(theta_deg)
    if key not in _TINY_TBG_GRID_CACHE:
        base = solve_bm_model_on_torus(
            TBGParameters.from_degrees(theta_deg),
            2,
            lg=7,
            calculate_chern_operator=True,
        )
        for name in _TINY_TBG_ARRAY_FIELDS:
            np.asarray(getattr(base, name)).setflags(write=False)
        _TINY_TBG_GRID_CACHE[key] = base
    copied = deepcopy(_TINY_TBG_GRID_CACHE[key])
    for name in _TINY_TBG_ARRAY_FIELDS:
        object.__setattr__(copied, name, np.array(getattr(copied, name), copy=True))
    return copied


def test_public_hf_adapter_registry_exposes_post_run_converters_without_run_dispatch() -> None:
    adapters = {info.name: info for info in list_hf_adapters()}
    expected = {
        "tdbg_projected_hf_result_to_hf_run_result",
        "tdbg_explicit_projected_run_hf",
        "htg_hf_run_to_hf_run_result",
        "htg_hf_run_to_hf_result",
        "htg_explicit_primitive_run_hf",
        "htg_supercell_hf_run_to_hf_run_result",
        "htg_supercell_hf_run_to_hf_result",
        "htg_explicit_supercell_run_hf",
        "tbg_zero_field_hf_run_to_hf_run_result",
        "tbg_zero_field_hf_run_to_hf_result",
        "tbg_zero_field_explicit_run_hf",
        "b0_hf_benchmark_run_to_hf_run_result",
        "rlg_hbn_hf_run_to_hf_run_result",
        "rlg_hbn_hf_run_to_hf_result",
        "rlg_hbn_explicit_run_hf",
        "polshyn_wang_hf_bundle_to_hf_run_result",
    }
    run_adapters = {
        "tdbg_explicit_projected_run_hf",
        "htg_explicit_primitive_run_hf",
        "htg_explicit_supercell_run_hf",
        "rlg_hbn_explicit_run_hf",
        "tbg_zero_field_explicit_run_hf",
    }

    assert expected <= set(adapters)
    for name in run_adapters:
        assert adapters[name].adapter_type == "run_hf"
        assert adapters[name].supports_run_hf_config is True
        assert adapters[name].requires_explicit_inputs
        assert adapters[name].run_hf_config_reason
    for name in expected - run_adapters:
        assert adapters[name].adapter_type in {"canonical_hf_run_result", "hf_result"}
        assert adapters[name].supports_run_hf_config is False
        assert ":" in adapters[name].import_path
        assert adapters[name].requires_explicit_inputs
        assert adapters[name].run_hf_config_reason


def test_public_hf_adapter_registry_filters_and_resolves_existing_helpers() -> None:
    htg_supercell = {info.name for info in list_hf_adapters(system_name="htg_supercell")}
    assert htg_supercell == {
        "htg_supercell_hf_run_to_hf_run_result",
        "htg_supercell_hf_run_to_hf_result",
        "htg_explicit_supercell_run_hf",
    }
    canonical = {info.name for info in list_hf_adapters(adapter_type="canonical_hf_run_result")}
    assert "tdbg_explicit_projected_run_hf" not in canonical
    assert "polshyn_wang_hf_bundle_to_hf_run_result" in canonical

    adapter = resolve_hf_adapter("htg_supercell_hf_run_to_hf_run_result")
    assert adapter.__name__ == "htg_supercell_hf_run_to_hf_run_result"
    assert adapter.__module__ == "mean_field.systems.htg.supercell_contracts"
    assert get_hf_adapter_info("tdbg_explicit_projected_run_hf").supports_run_hf_config is True
    assert get_hf_adapter_info("htg_explicit_primitive_run_hf").supports_run_hf_config is True
    assert "HTGRunHFConfig" in get_hf_adapter_info("htg_explicit_primitive_run_hf").run_hf_config_reason
    assert "RLGhBNRunHFConfig" in get_hf_adapter_info("rlg_hbn_explicit_run_hf").run_hf_config_reason
    assert "TBGZeroFieldRunHFConfig" in get_hf_adapter_info("tbg_zero_field_explicit_run_hf").run_hf_config_reason
    assert "htg_supercell_hf_run_to_hf_result" in hf_api.__all__
    assert "rlg_hbn_hf_run_to_hf_result" in hf_api.__all__
    assert "tbg_zero_field_hf_run_to_hf_result" in hf_api.__all__

    with pytest.raises(KeyError, match="Unknown HF adapter"):
        get_hf_adapter_info("not_a_registered_hf_adapter")


def test_public_run_hf_tbg_bm_requires_explicit_system_workflow() -> None:
    model = make_model("tbg", variant="zero_field_bm", theta_deg=1.2, lg=1)
    cfg = HFConfig(filling=0, mesh=(1, 1), max_iter=1)

    with pytest.raises(NotImplementedError, match="explicit tbg_zero_field_config"):
        run_hf(model, cfg)


def test_public_run_hf_tbg_zero_field_explicit_config_attaches_canonical_contract_result() -> None:
    grid_solution = _tiny_tbg_grid_solution(theta_deg=1.05)
    model = make_model(
        "tbg",
        variant="zero_field_bm",
        theta_deg=1.05,
        lg=grid_solution.lg,
    )
    interaction_spec = TBGZeroFieldInteractionSpec()
    cfg = HFConfig(
        filling=5.0e-13,
        mesh=(2, 2),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme="average",
        epsilon_r=interaction_spec.epsr,
        dsc_nm=interaction_spec.dsc_nm,
        coulomb_kernel="2d_gate",
        seeds=("5",),
    )
    tbg_cfg = TBGZeroFieldRunHFConfig(
        grid_solution=grid_solution,
        nu=5.0e-13,
        init_mode="bm",
        seed=5,
        max_iter=1,
        overlap_lg=7,
        precision=1.0e-6,
        interaction_spec=interaction_spec,
    )

    result = run_hf(model, cfg, tbg_zero_field_config=tbg_cfg)

    assert isinstance(result, HFResult)
    assert tbg_cfg.nu == 0.0
    assert result.config.filling == 0.0
    assert result.model.system_name == "tbg_zero_field"
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.state.seed == 5
    assert result.config.dsc_nm == interaction_spec.dsc_nm
    assert result.state.interaction_spec == interaction_spec
    assert (
        result.state.hf_source_receipt.interaction_spec_fingerprint
        == interaction_spec.fingerprint
    )
    assert result.observables["public_run_hf_adapter"].endswith("run_tbg_zero_field_hf_config_adapter")
    assert result.observables["hf_mode"] == "restricted"
    assert result.observables["beta"] == tbg_cfg.beta
    assert result.observables["hf_run_provenance"] == result.state.provenance.to_metadata()
    assert result.artifacts.metadata["workflow"] == "tbg.zero_field.restricted_hf.raw_run_result"
    assert result.observables["torus_mesh_fingerprint"] == grid_solution.torus_mesh.fingerprint
    assert result.observables["bm_solution_sha256"] == result.state.screened_block_bundle.bm_solution_sha256
    assert result.observables["lattice_kvec_sha256"] == result.state.hf_source_receipt.lattice_kvec_sha256
    assert result.observables["interaction_spec"] == interaction_spec.to_metadata()
    assert result.observables["hf_source_receipt"] == result.state.hf_source_receipt.to_metadata()
    assert result.observables["screened_block_bundle"] == result.state.screened_block_bundle.to_metadata()
    assert result.canonical_run_result.final_state.observables["grid_mesh_size"] == 2
    assert result.canonical_run_result.final_state.density.reference.metadata["raw_density_convention"] == "stored_delta"
    assert (
        result.canonical_run_result.final_state.density.metadata["density_delta_definition"]
        == "D_stored[a,b]=<c_a† c_b>-0.5*delta_ab"
    )
    assert result.canonical_run_result.final_state.hamiltonian.metadata["supports_crpa"] is False

    for reserved_key in (
        "interaction_spec",
        "hf_source_receipt",
        "screened_block_bundle",
        "hf_mode",
        "beta",
        "hf_run_provenance",
        "solver_provenance",
        "mesh_fingerprint",
        "source_fingerprint",
        "torus_mesh_fingerprint",
        "bm_solution_sha256",
        "lattice_kvec_sha256",
    ):
        with pytest.raises(ValueError, match="reserved verified TBG HF keys"):
            tbg_zero_field_hf_run_to_hf_result(
                result.state,
                grid_solution=grid_solution,
                config=cfg,
                observables={reserved_key: "caller-forged"},
            )


def test_public_run_hf_tbg_zero_field_refuses_fractional_hfconfig_filling() -> None:
    grid_solution = _tiny_tbg_grid_solution(theta_deg=1.05)
    model = make_model(
        "tbg",
        variant="zero_field_bm",
        theta_deg=1.05,
        lg=grid_solution.lg,
    )
    interaction_spec = TBGZeroFieldInteractionSpec()
    config = HFConfig(
        filling=0.25,
        mesh=(2, 2),
        max_iter=0,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme="average",
        epsilon_r=interaction_spec.epsr,
        dsc_nm=interaction_spec.dsc_nm,
        coulomb_kernel="2d_gate",
        seeds=("1",),
    )
    tbg_config = TBGZeroFieldRunHFConfig(
        grid_solution=grid_solution,
        nu=0.0,
        init_mode="bm",
        max_iter=0,
        overlap_lg=7,
        precision=1.0e-6,
        interaction_spec=interaction_spec,
    )

    with pytest.raises(ValueError, match="separate supercell workflow"):
        run_hf(model, config, tbg_zero_field_config=tbg_config)

@pytest.mark.parametrize(
    "mismatch_kind",
    ["w0", "w1", "strain", "sigma_rotation", "periodic_g_grid"],
)
def test_public_run_hf_tbg_rejects_complete_bm_generation_mismatch(
    mismatch_kind: str,
) -> None:
    grid_solution = _tiny_tbg_grid_solution(theta_deg=1.05)
    model_kwargs: dict[str, object] = {
        "variant": "zero_field_bm",
        "theta_deg": 1.05,
        "lg": grid_solution.lg,
        "params": grid_solution.params,
        "sigma_rotation": grid_solution.sigma_rotation,
        "periodic_g_grid": grid_solution.periodic_g_grid,
    }
    if mismatch_kind == "w0":
        model_kwargs["params"] = replace(
            grid_solution.params,
            w0=grid_solution.params.w0 + 1.0,
        )
    elif mismatch_kind == "w1":
        model_kwargs["params"] = replace(
            grid_solution.params,
            w1=grid_solution.params.w1 + 1.0,
        )
    elif mismatch_kind == "strain":
        model_kwargs["params"] = replace(
            grid_solution.params,
            strain=grid_solution.params.strain + 1.0e-4,
        )
    elif mismatch_kind == "sigma_rotation":
        model_kwargs["sigma_rotation"] = not grid_solution.sigma_rotation
    elif mismatch_kind == "periodic_g_grid":
        model_kwargs["periodic_g_grid"] = not grid_solution.periodic_g_grid
    else:
        raise AssertionError(f"Unhandled mismatch_kind={mismatch_kind!r}")
    model = make_model("tbg", **model_kwargs)
    interaction_spec = TBGZeroFieldInteractionSpec()
    config = HFConfig(
        filling=0.0,
        mesh=(2, 2),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme="average",
        epsilon_r=interaction_spec.epsr,
        dsc_nm=interaction_spec.dsc_nm,
        coulomb_kernel="2d_gate",
        seeds=("1",),
    )
    tbg_config = TBGZeroFieldRunHFConfig(
        grid_solution=grid_solution,
        nu=0.0,
        init_mode="bm",
        max_iter=1,
        overlap_lg=7,
        precision=1.0e-6,
        interaction_spec=interaction_spec,
    )

    with pytest.raises(ValueError, match="BM generation fingerprint"):
        run_hf(model, config, tbg_zero_field_config=tbg_config)


def test_public_post_run_tbg_config_rejects_interaction_kernel_and_window_mislabels() -> None:
    grid_solution = _tiny_tbg_grid_solution(theta_deg=1.05)
    model = make_model(
        "tbg",
        variant="zero_field_bm",
        theta_deg=1.05,
        lg=grid_solution.lg,
    )
    interaction_spec = TBGZeroFieldInteractionSpec()
    cfg = HFConfig(
        filling=0.0,
        mesh=(2, 2),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme="average",
        epsilon_r=interaction_spec.epsr,
        dsc_nm=interaction_spec.dsc_nm,
        coulomb_kernel="2d_gate",
        seeds=("1",),
    )
    run_config = TBGZeroFieldRunHFConfig(
        grid_solution=grid_solution,
        nu=0.0,
        init_mode="bm",
        max_iter=1,
        overlap_lg=7,
        precision=1.0e-6,
        interaction_spec=interaction_spec,
    )
    result = run_hf(model, cfg, tbg_zero_field_config=run_config)
    for mislabeled in (
        replace(cfg, interaction_scheme="cn"),
        replace(cfg, coulomb_kernel="crpa"),
        replace(cfg, active_window=(1, 1)),
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            tbg_zero_field_hf_run_to_hf_result(
                result.state,
                grid_solution=grid_solution,
                config=mislabeled,
            )


def test_public_run_hf_tbg_zero_field_rejects_missing_grid_contract() -> None:
    grid_solution = _tiny_tbg_grid_solution(theta_deg=1.05)
    model = make_model(
        "tbg",
        variant="zero_field_bm",
        theta_deg=1.05,
        lg=grid_solution.lg,
    )
    interaction_spec = TBGZeroFieldInteractionSpec()
    cfg = HFConfig(
        filling=0.0,
        mesh=(1, 1),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme="average",
        epsilon_r=interaction_spec.epsr,
        dsc_nm=interaction_spec.dsc_nm,
        coulomb_kernel="2d_gate",
        seeds=("1",),
    )
    tbg_cfg = TBGZeroFieldRunHFConfig(
        grid_solution=grid_solution,
        nu=0.0,
        init_mode="bm",
        max_iter=1,
        overlap_lg=7,
        precision=1.0e-6,
        interaction_spec=interaction_spec,
    )

    with pytest.raises(ValueError, match="carried half-open torus mesh"):
        run_hf(model, cfg, tbg_zero_field_config=tbg_cfg)


def test_public_run_hf_tbg_zero_field_refuses_untyped_and_dimensionless_dsc_contracts() -> None:
    grid_solution = _tiny_tbg_grid_solution(theta_deg=1.05)
    model = make_model(
        "tbg",
        variant="zero_field_bm",
        theta_deg=1.05,
        lg=grid_solution.lg,
    )
    with pytest.raises(ValueError, match="requires a typed"):
        TBGZeroFieldRunHFConfig(grid_solution=grid_solution, nu=0.0)

    interaction_spec = TBGZeroFieldInteractionSpec()
    with pytest.raises(ValueError, match="Raw screening_lm is rejected"):
        TBGZeroFieldRunHFConfig(
            grid_solution=grid_solution,
            nu=0.0,
            overlap_lg=7,
            interaction_spec=interaction_spec,
            screening_lm=interaction_spec.screening_lm,
        )
    tbg_cfg = TBGZeroFieldRunHFConfig(
        grid_solution=grid_solution,
        nu=0.0,
        init_mode="bm",
        max_iter=1,
        overlap_lg=7,
        precision=1.0e-6,
        interaction_spec=interaction_spec,
    )
    config_with_dimensionless_lm = HFConfig(
        filling=0.0,
        mesh=(2, 2),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme="average",
        epsilon_r=interaction_spec.epsr,
        dsc_nm=interaction_spec.screening_lm,
        coulomb_kernel="2d_gate",
        seeds=("1",),
    )
    with pytest.raises(ValueError, match="physical gate distance in nm"):
        run_hf(
            model,
            config_with_dimensionless_lm,
            tbg_zero_field_config=tbg_cfg,
        )


@pytest.mark.parametrize(
    "bad_mesh",
    [(2.9, 2), (2, 2.9), (True, 2), (2, np.bool_(False))],
)
def test_hf_config_rejects_float_and_bool_mesh_components(
    bad_mesh: tuple[object, object],
) -> None:
    with pytest.raises(ValueError, match="positive pair of non-bool integers"):
        HFConfig(filling=0.0, mesh=bad_mesh)  # type: ignore[arg-type]

def test_hf_config_normalizes_numpy_integer_mesh_to_exact_tuple() -> None:
    config = HFConfig(filling=0.0, mesh=(np.int64(2), np.int32(3)))
    assert config.mesh == (2, 3)
    assert type(config.mesh) is tuple
    assert all(type(value) is int for value in config.mesh)

@pytest.mark.parametrize("bad_max_iter", [2.9, True, np.bool_(False)])
def test_hf_config_rejects_non_integer_and_bool_max_iter(bad_max_iter: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        HFConfig(filling=0.0, mesh=(1, 1), max_iter=bad_max_iter)  # type: ignore[arg-type]

def test_hf_config_accepts_zero_and_numpy_integer_max_iter() -> None:
    config = HFConfig(filling=0.0, mesh=(1, 1), max_iter=np.int64(0))
    assert config.max_iter == 0
    assert isinstance(config.max_iter, int)

@pytest.mark.parametrize(
    "bad_nu",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "inf", "neg_inf"],
)
def test_tbg_typed_config_rejects_nonfinite_primitive_cell_nu(
    bad_nu: float,
) -> None:
    with pytest.raises(ValueError, match="finite real integer"):
        TBGZeroFieldRunHFConfig(
            grid_solution=_tiny_tbg_grid_solution(),
            nu=bad_nu,
            overlap_lg=7,
            interaction_spec=TBGZeroFieldInteractionSpec(),
        )

@pytest.mark.parametrize(
    "bad_nu",
    [0.25, 2.0 + 2.0e-12],
    ids=["fractional", "outside_integer_tolerance"],
)
def test_tbg_typed_config_refuses_fractional_primitive_cell_nu(
    bad_nu: float,
) -> None:
    with pytest.raises(ValueError, match="separate supercell workflow"):
        TBGZeroFieldRunHFConfig(
            grid_solution=_tiny_tbg_grid_solution(),
            nu=bad_nu,
            overlap_lg=7,
            interaction_spec=TBGZeroFieldInteractionSpec(),
        )

def test_tbg_typed_config_normalizes_float_integer_nu_with_tight_tolerance() -> None:
    exact = TBGZeroFieldRunHFConfig(
        grid_solution=_tiny_tbg_grid_solution(),
        nu=2.0,
        overlap_lg=7,
        interaction_spec=TBGZeroFieldInteractionSpec(),
    )
    near = TBGZeroFieldRunHFConfig(
        grid_solution=_tiny_tbg_grid_solution(),
        nu=2.0 + 5.0e-13,
        overlap_lg=7,
        interaction_spec=TBGZeroFieldInteractionSpec(),
    )
    assert exact.nu == 2.0
    assert near.nu == 2.0
    assert type(exact.nu) is float
    assert type(near.nu) is float

@pytest.mark.parametrize("bad_max_iter", [2.9, True, np.bool_(True)])
def test_tbg_typed_config_rejects_non_integer_and_bool_max_iter(
    bad_max_iter: object,
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        TBGZeroFieldRunHFConfig(
            grid_solution=_tiny_tbg_grid_solution(),
            nu=0.0,
            max_iter=bad_max_iter,  # type: ignore[arg-type]
            overlap_lg=7,
            interaction_spec=TBGZeroFieldInteractionSpec(),
        )

@pytest.mark.parametrize("bad_seed", [2.9, True, np.bool_(False)])
def test_tbg_typed_config_rejects_float_and_bool_seed(bad_seed: object) -> None:
    with pytest.raises(ValueError, match="seed must be a non-bool integer"):
        TBGZeroFieldRunHFConfig(
            grid_solution=_tiny_tbg_grid_solution(),
            nu=0.0,
            seed=bad_seed,  # type: ignore[arg-type]
            overlap_lg=7,
            interaction_spec=TBGZeroFieldInteractionSpec(),
        )

@pytest.mark.parametrize(
    ("bad_overlap_lg", "message"),
    [
        (7.9, "positive odd integer"),
        (True, "positive odd integer"),
        (np.bool_(False), "positive odd integer"),
        (6, "positive odd integer"),
        (5, "insufficient"),
    ],
)
def test_tbg_typed_config_rejects_invalid_or_insufficient_overlap_lg(
    bad_overlap_lg: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TBGZeroFieldRunHFConfig(
            grid_solution=_tiny_tbg_grid_solution(),
            nu=0.0,
            max_iter=0,
            overlap_lg=bad_overlap_lg,  # type: ignore[arg-type]
            interaction_spec=TBGZeroFieldInteractionSpec(),
        )

def test_tbg_typed_config_accepts_numpy_integer_limits() -> None:
    config = TBGZeroFieldRunHFConfig(
        grid_solution=_tiny_tbg_grid_solution(),
        nu=0.0,
        seed=np.int64(17),
        max_iter=np.int64(0),
        overlap_lg=np.int64(7),
        interaction_spec=TBGZeroFieldInteractionSpec(),
    )
    assert config.seed == 17
    assert config.max_iter == 0
    assert config.overlap_lg == 7
    assert type(config.seed) is int
    assert type(config.max_iter) is int
    assert type(config.overlap_lg) is int

def test_public_run_hf_tdbg_requires_explicit_projected_config() -> None:
    model = make_model("tdbg", theta_deg=1.38, cut=1.0)
    cfg = HFConfig(filling=2, mesh=(1, 1), max_iter=1, precision=1.0e-7, density_convention="projector")

    with pytest.raises(NotImplementedError, match="explicit tdbg_config"):
        run_hf(model, cfg)


def test_public_run_hf_htg_requires_explicit_system_config() -> None:
    model = make_model("htg", theta_deg=1.8, n_shells=0)
    cfg = HFConfig(
        filling=3.0,
        mesh=(1, 1),
        max_iter=1,
        density_convention="stored_delta",
        epsilon_r=8.0,
        dsc_nm=25.0,
    )

    with pytest.raises(NotImplementedError, match="explicit htg_config"):
        run_hf(model, cfg)


def test_public_run_hf_rlg_hbn_requires_explicit_system_config() -> None:
    model = make_model("rlg_hbn", layer_count=3, xi=1, theta_deg=0.77, displacement_field_mev=24.0, shell_count=1)
    interaction = RLGhBNInteractionParams(
        active_valence_bands=1,
        active_conduction_bands=1,
        k_mesh_size=1,
        interaction_cutoff_q1=1.0,
        interaction_dimension="2d_diagnostic",
        use_screened_basis=False,
    )
    cfg = HFConfig(
        filling=1.0,
        mesh=(1, 1),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme=interaction.scheme,  # type: ignore[arg-type]
        epsilon_r=interaction.epsilon_r,
        dsc_nm=interaction.gate_distance_nm,
        coulomb_kernel="2d_gate",
    )

    with pytest.raises(NotImplementedError, match="explicit rlg_hbn_config"):
        run_hf(model, cfg)


def test_public_run_hf_rlg_hbn_explicit_config_attaches_canonical_contract_result() -> None:
    model = make_model("rlg_hbn", layer_count=3, xi=1, theta_deg=0.77, displacement_field_mev=24.0, shell_count=1)
    interaction = RLGhBNInteractionParams(
        active_valence_bands=1,
        active_conduction_bands=1,
        k_mesh_size=1,
        interaction_cutoff_q1=1.0,
        interaction_dimension="2d_diagnostic",
        use_screened_basis=False,
    )
    cfg = HFConfig(
        filling=1.0,
        mesh=(1, 1),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme=interaction.scheme,  # type: ignore[arg-type]
        epsilon_r=interaction.epsilon_r,
        dsc_nm=interaction.gate_distance_nm,
        coulomb_kernel="2d_gate",
    )
    rlg_cfg = RLGhBNRunHFConfig(
        nu=1.0,
        interaction=interaction,
        mesh_size=1,
        init_mode="flavor",
        seed=4,
        max_iter=1,
        precision=1.0e-6,
    )

    result = run_hf(model, cfg, rlg_hbn_config=rlg_cfg)

    assert isinstance(result, HFResult)
    assert result.model.system_name == "rlg_hbn"
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.state.seed == 4
    assert result.observables["public_run_hf_adapter"].endswith("run_rlg_hbn_hf_config_adapter")
    assert result.canonical_run_result.final_state.density.reference.metadata["raw_density_convention"] == "stored_delta"
    assert result.canonical_run_result.final_state.hamiltonian.metadata["supports_crpa"] is False


def test_public_run_hf_rlg_hbn_rejects_mismatched_generic_config() -> None:
    model = make_model("rlg_hbn", layer_count=3, xi=1, theta_deg=0.77, displacement_field_mev=24.0, shell_count=1)
    interaction = RLGhBNInteractionParams(
        active_valence_bands=1,
        active_conduction_bands=1,
        k_mesh_size=1,
        interaction_cutoff_q1=1.0,
        interaction_dimension="2d_diagnostic",
        use_screened_basis=False,
    )
    cfg = HFConfig(
        filling=1.0,
        mesh=(1, 1),
        max_iter=1,
        precision=1.0e-6,
        density_convention="projector",
        interaction_scheme=interaction.scheme,  # type: ignore[arg-type]
        epsilon_r=interaction.epsilon_r,
        dsc_nm=interaction.gate_distance_nm,
        coulomb_kernel="2d_gate",
    )
    rlg_cfg = RLGhBNRunHFConfig(nu=1.0, interaction=interaction, mesh_size=1, max_iter=1, precision=1.0e-6)

    with pytest.raises(ValueError, match="density_convention='stored_delta'"):
        run_hf(model, cfg, rlg_hbn_config=rlg_cfg)


def test_public_run_hf_htg_primitive_explicit_config_attaches_canonical_contract_result() -> None:
    model = make_model("htg", theta_deg=1.8, n_shells=0)
    interaction = InteractionParams(n_k=1, g_shells=0)
    cfg = HFConfig(
        filling=3.0,
        mesh=(1, 1),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        epsilon_r=interaction.epsilon_r,
        dsc_nm=interaction.d_sc_nm,
    )
    htg_cfg = HTGRunHFConfig(
        nu=3.0,
        mesh_size=1,
        interaction=interaction,
        init_mode="bm",
        seed=2,
        max_iter=1,
        precision=1.0e-6,
        g_shells=0,
        use_numba=False,
    )

    result = run_hf(model, cfg, htg_config=htg_cfg)

    assert isinstance(result, HFResult)
    assert result.model.system_name == "htg"
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.state.seed == 2
    assert result.observables["public_run_hf_adapter"].endswith("run_htg_hf_config_adapter")
    assert result.canonical_run_result.final_state.density.reference.metadata["raw_density_convention"] == "stored_delta"
    assert result.canonical_run_result.final_state.hamiltonian.metadata["supports_crpa"] is False


def test_public_run_hf_htg_supercell_explicit_config_attaches_canonical_contract_result() -> None:
    model = make_model("htg", theta_deg=1.8, n_shells=0)
    interaction = InteractionParams(n_k=1, g_shells=0)
    cfg = HFConfig(
        filling=3.5,
        mesh=(1, 1),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        epsilon_r=interaction.epsilon_r,
        dsc_nm=interaction.d_sc_nm,
    )
    htg_supercell_cfg = HTGSupercellRunHFConfig(
        primitive_nu=3.5,
        mesh_size=1,
        interaction=interaction,
        init_mode="bm",
        seed=1,
        max_iter=1,
        precision=1.0e-6,
        g_shells=0,
        use_numba=False,
    )

    result = run_hf(model, cfg, htg_supercell_config=htg_supercell_cfg)

    assert isinstance(result, HFResult)
    assert result.model.system_name == "htg_supercell"
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.state.seed == 1
    assert result.observables["supercell_area_ratio"] == 2
    assert result.observables["public_run_hf_adapter"].endswith("run_htg_supercell_hf_config_adapter")
    assert result.canonical_run_result.final_state.density.reference.metadata["raw_density_convention"] == "stored_delta"
    assert result.canonical_run_result.final_state.hamiltonian.metadata["supports_crpa"] is False


def test_public_run_hf_tdbg_explicit_config_dispatches_without_guessing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model("tdbg", theta_deg=1.38, cut=1.0)
    cfg = HFConfig(filling=2, mesh=(1, 1), max_iter=1, precision=1.0e-7, density_convention="projector")
    tdbg_cfg = _tiny_tdbg_config()
    calls: dict[str, object] = {}

    def fake_build_data(config: TDBGProjectedHFConfig) -> SimpleNamespace:
        calls["build_config"] = config
        return SimpleNamespace(config=config)

    class FakeTDBGResult:
        def to_summary_dict(self) -> dict[str, object]:
            return {
                "init_mode": "sp",
                "seed": 7,
                "converged": False,
                "exit_reason": "max_iter",
                "iterations": 1,
            }

    def fake_run(data: SimpleNamespace, *, init_mode: str, seed: int = 1) -> FakeTDBGResult:
        calls["run_data"] = data
        calls["init_mode"] = init_mode
        calls["seed"] = seed
        return FakeTDBGResult()

    monkeypatch.setattr(tdbg_system, "build_tdbg_projected_hf_data", fake_build_data)
    monkeypatch.setattr(tdbg_system, "run_tdbg_projected_hf", fake_run)

    result = run_hf(model, cfg, tdbg_config=tdbg_cfg, init_mode="sp", seed=7)

    assert isinstance(result, HFResult)
    assert result.model.system_name == "tdbg"
    assert result.state.to_summary_dict()["iterations"] == 1
    assert result.observables["init_mode"] == "sp"
    assert calls["build_config"] is tdbg_cfg
    assert calls["init_mode"] == "sp"
    assert calls["seed"] == 7
    assert result.artifacts is not None
    assert result.artifacts.metadata["workflow"] == "tdbg.projected_hf.explicit_config"
    assert result.artifacts.conventions.to_dict()["energy_unit"] == "eV"  # type: ignore[union-attr]
    assert result.artifacts.conventions.to_dict()["density_convention"] == "projector"  # type: ignore[union-attr]
    assert result.canonical_run_result is None


def test_public_run_hf_tdbg_explicit_config_attaches_canonical_contract_result() -> None:
    model = make_model("tdbg", theta_deg=1.38, cut=1.0)
    cfg = HFConfig(filling=2, mesh=(1, 1), max_iter=1, precision=1.0e-7, density_convention="projector")

    result = run_hf(model, cfg, tdbg_config=_tiny_tdbg_config(), init_mode="sp", seed=7)

    assert isinstance(result, HFResult)
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.canonical_run_result.final_state.density.reference.scheme == "CN"
    np.testing.assert_allclose(
        result.canonical_run_result.final_state.density.projector,
        result.state.run.state.density,
    )
    assert result.canonical_run_result.final_state.hamiltonian.metadata["supports_crpa"] is False


def test_public_run_hf_tdbg_rejects_mismatched_generic_config() -> None:
    model = make_model("tdbg", theta_deg=1.38, cut=1.0)
    cfg = HFConfig(filling=2, mesh=(1, 1), max_iter=1, precision=1.0e-7)

    with pytest.raises(ValueError, match="density_convention='projector'"):
        run_hf(model, cfg, tdbg_config=_tiny_tdbg_config(), init_mode="sp")
