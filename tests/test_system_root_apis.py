from __future__ import annotations

import importlib.util

import mean_field.systems.RnG_hBN as rlg_hbn_api
import mean_field.systems.atmg as atmg_api
import mean_field.systems.htg as htg_api
import mean_field.systems.htqg as htqg_api
import mean_field.systems.tbg as tbg_api
import mean_field.systems.tbg.finite_field as tbg_finite_api
import mean_field.systems.tbg.zero_field as tbg_zero_api
import mean_field.systems.tdbg as tdbg_api
import mean_field.systems.tmbg as tmbg_api
from mean_field.systems.RnG_hBN.hf_contracts import RLGhBNRunHFConfig
from mean_field.systems.RnG_hBN.interaction import RLGhBNInteractionParams
from mean_field.systems.RnG_hBN.model import RLGhBNModel
from mean_field.systems.RnG_hBN.params import RLGhBNParams
from mean_field.systems.atmg.model import ATMGModel
from mean_field.systems.atmg.params import ATMGParameters
from mean_field.systems.htg._hf_contracts import HTGRunHFConfig
from mean_field.systems.htg.model import HTGModel
from mean_field.systems.htg.params import HTGParams, InteractionParams
from mean_field.systems.htg.supercell_contracts import HTGSupercellRunHFConfig
from mean_field.systems.htqg.domains import HTQGDomain, HTQGExplicitDisplacements
from mean_field.systems.htqg.model import HTQGModel
from mean_field.systems.htqg.params import HTQGParams
from mean_field.core.hf.finite_field import (
    FiniteFieldHartreeFockInputBundle,
    FiniteFieldHartreeFockInputs,
    FiniteFieldHartreeFockState,
    FiniteFieldHartreeFockSummary,
    FiniteFieldTLSymmetricHartreeFockInputs,
    MagneticOverlapData,
    run_finite_field_hartree_fock_from_inputs,
    summarize_finite_field_hartree_fock,
)
from mean_field.core.magnetic_field import MagneticFlux
from mean_field.systems.tbg.finite_field.hf import (
    build_finite_field_hf_inputs_from_parameters,
    build_finite_field_hf_inputs_from_spectra,
    build_finite_field_hf_state_from_spectra,
    build_full_flavor_overlap_data_from_spectra,
)
from mean_field.systems.tbg.finite_field.spectrum import (
    FiniteFieldBMParameters,
    MagneticSpectrumResult,
    MagneticSpectrumSweepCase,
    MagneticSpectrumSweepResult,
    compute_magnetic_spectrum,
    compute_magnetic_spectrum_sweep,
)
from mean_field.systems.tbg.zero_field._hf_basis_overlap import (
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
)
from mean_field.systems.tbg.zero_field.hf_contracts import TBGZeroFieldRunHFConfig
from mean_field.systems.tbg.zero_field.interaction import TBGZeroFieldInteractionSpec
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.model import (
    BMSolution,
    TBGZeroFieldBMModel,
    TBGZeroFieldTorusMesh,
    build_tbg_zero_field_half_open_torus_mesh,
    solve_bm_model_on_torus,
)
from mean_field.systems.tdbg.model import TDBGModel
from mean_field.systems.tdbg.params import TDBGParameters
from mean_field.systems.tdbg.projected_hf_config import (
    TDBGInteractionSettings,
    TDBGProjectedHFConfig,
    TDBGProjectedWindow,
)
from mean_field.systems.tmbg.model import TMBGModel
from mean_field.systems.tmbg.params import TMBGParameters


def test_retired_internal_hf_facades_are_absent() -> None:
    assert importlib.util.find_spec("mean_field.systems.htg.mean_field_adapter") is None
    assert importlib.util.find_spec("mean_field.systems.tdbg.projected_hf") is None
    assert importlib.util.find_spec("mean_field.systems.tbg.zero_field.hf") is None
    assert importlib.util.find_spec("mean_field.systems.tmbg._polshyn_shared") is None
    assert importlib.util.find_spec("mean_field.systems.tmbg.polshyn_supercell") is None


def _assert_exact_api(module, expected: tuple[tuple[str, object], ...]) -> None:
    assert module.__all__ == [name for name, _ in expected]
    for name, owner in expected:
        assert getattr(module, name) is owner


def test_system_roots_are_narrow_typed_apis() -> None:
    _assert_exact_api(
        rlg_hbn_api,
        (
            ("RLGhBNInteractionParams", RLGhBNInteractionParams),
            ("RLGhBNModel", RLGhBNModel),
            ("RLGhBNParams", RLGhBNParams),
            ("RLGhBNRunHFConfig", RLGhBNRunHFConfig),
        ),
    )
    _assert_exact_api(
        htg_api,
        (
            ("HTGModel", HTGModel),
            ("HTGParams", HTGParams),
            ("HTGRunHFConfig", HTGRunHFConfig),
            ("HTGSupercellRunHFConfig", HTGSupercellRunHFConfig),
            ("InteractionParams", InteractionParams),
        ),
    )
    _assert_exact_api(
        tbg_api,
        (("TBGParameters", TBGParameters), ("TBGZeroFieldBMModel", TBGZeroFieldBMModel)),
    )
    _assert_exact_api(
        tbg_zero_api,
        (
            ("BMSolution", BMSolution),
            ("RestrictedHartreeFockRun", RestrictedHartreeFockRun),
            ("RestrictedHartreeFockState", RestrictedHartreeFockState),
            ("TBGZeroFieldBMModel", TBGZeroFieldBMModel),
            ("TBGZeroFieldInteractionSpec", TBGZeroFieldInteractionSpec),
            ("TBGZeroFieldRunHFConfig", TBGZeroFieldRunHFConfig),
            ("TBGZeroFieldTorusMesh", TBGZeroFieldTorusMesh),
            ("build_tbg_zero_field_half_open_torus_mesh", build_tbg_zero_field_half_open_torus_mesh),
            ("solve_bm_model_on_torus", solve_bm_model_on_torus),
        ),
    )
    _assert_exact_api(
        tbg_finite_api,
        (
            ("FiniteFieldBMParameters", FiniteFieldBMParameters),
            ("MagneticFlux", MagneticFlux),
            ("MagneticSpectrumResult", MagneticSpectrumResult),
            ("MagneticSpectrumSweepCase", MagneticSpectrumSweepCase),
            ("MagneticSpectrumSweepResult", MagneticSpectrumSweepResult),
            ("FiniteFieldHartreeFockInputBundle", FiniteFieldHartreeFockInputBundle),
            ("FiniteFieldHartreeFockInputs", FiniteFieldHartreeFockInputs),
            ("FiniteFieldTLSymmetricHartreeFockInputs", FiniteFieldTLSymmetricHartreeFockInputs),
            ("FiniteFieldHartreeFockState", FiniteFieldHartreeFockState),
            ("FiniteFieldHartreeFockSummary", FiniteFieldHartreeFockSummary),
            ("MagneticOverlapData", MagneticOverlapData),
            ("compute_magnetic_spectrum", compute_magnetic_spectrum),
            ("compute_magnetic_spectrum_sweep", compute_magnetic_spectrum_sweep),
            ("build_finite_field_hf_state_from_spectra", build_finite_field_hf_state_from_spectra),
            ("build_full_flavor_overlap_data_from_spectra", build_full_flavor_overlap_data_from_spectra),
            ("build_finite_field_hf_inputs_from_spectra", build_finite_field_hf_inputs_from_spectra),
            ("build_finite_field_hf_inputs_from_parameters", build_finite_field_hf_inputs_from_parameters),
            ("run_finite_field_hartree_fock_from_inputs", run_finite_field_hartree_fock_from_inputs),
            ("summarize_finite_field_hartree_fock", summarize_finite_field_hartree_fock),
        ),
    )
    _assert_exact_api(
        tdbg_api,
        (
            ("TDBGInteractionSettings", TDBGInteractionSettings),
            ("TDBGModel", TDBGModel),
            ("TDBGParameters", TDBGParameters),
            ("TDBGProjectedHFConfig", TDBGProjectedHFConfig),
            ("TDBGProjectedWindow", TDBGProjectedWindow),
        ),
    )
    _assert_exact_api(
        tmbg_api,
        (("TMBGModel", TMBGModel), ("TMBGParameters", TMBGParameters)),
    )
    _assert_exact_api(
        atmg_api,
        (("ATMGModel", ATMGModel), ("ATMGParameters", ATMGParameters)),
    )
    _assert_exact_api(
        htqg_api,
        (
            ("HTQGDomain", HTQGDomain),
            ("HTQGExplicitDisplacements", HTQGExplicitDisplacements),
            ("HTQGModel", HTQGModel),
            ("HTQGParams", HTQGParams),
        ),
    )
