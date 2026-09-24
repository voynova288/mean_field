from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType


_PACKAGE_OWNERS = {
    "analysis.shift_current": (
        (
            "analysis.shift_current.core",
            (
                "Axis",
                "Component",
                "E_CHARGE_C",
                "ExactDegenerateGroupTransition",
                "HBAR_EV_S",
                "HBAR_J_S",
                "HTG_LEGACY_CONVENTION",
                "JOYA_EQ7_GEOMETRIC_CONVENTION",
                "KB_EV_PER_K",
                "OpticalSymmetrization",
                "PairTransitionKernel",
                "PairTransitionWeight",
                "SHIFT_CURRENT_PREFAC_UA_NM_PER_V2",
                "ShiftCurrentComponent",
                "ShiftCurrentConvention",
                "ShiftCurrentTensors",
                "WANNIERBERRI_INTERNAL_IMN_CONVENTION",
                "WANNIERBERRI_POSITIVE_TRANSITION_PREFAC_UA_NM_PER_V2",
                "accumulate_fermi_omega_heatmap",
                "add_transitions_to_integral",
                "axis_index",
                "berry_connection_matrix",
                "component_amplitude_from_pair",
                "component_from_any",
                "component_group_trace_amplitude",
                "component_group_trace_kernel",
                "component_kernel_from_gauge_pair",
                "component_kernel_from_pair",
                "component_label",
                "component_transition_weight",
                "component_transition_weight_from_gauge_pair",
                "conductivity_from_integral",
                "exact_degenerate_group_transition_weight",
                "fermi_occupation",
                "fermi_window_indices",
                "generalized_derivative_from_velocity",
                "lorentzian_delta",
                "parse_component",
                "positive_transition_pairs",
                "positive_transition_terms",
                "precompute_shift_current_tensors",
                "second_derivative_matrices",
                "spectra_from_transition_table",
                "velocity_matrices",
                "wannierberri_positive_transition_conductivity_from_imn",
            ),
        ),
    ),
    "analysis.injection_current": (
        (
            "analysis.injection_current.core",
            (
                "InjectionCurrentTensors",
                "accumulate_injection_spectrum",
                "component_label",
                "cpge_part",
                "injection_component_kernel",
                "injection_component_kernel_from_gauge_pair",
                "injection_spectra_from_transition_table",
                "injection_transition_weight",
                "linear_injection_part",
                "positive_injection_transition_terms",
                "precompute_injection_current_tensors",
            ),
        ),
    ),
    "analysis.optical": (
        (
            "analysis.optical.core",
            (
                "OpticalKPointResponseKind",
                "OpticalKPointResponseKindLike",
                "OpticalResponseKind",
                "OpticalResponseKindLike",
                "OpticalTransitionResponseKind",
                "OpticalTransitionResponseKindLike",
                "OpticalResponseTensors",
                "OpticalTensors",
                "accumulate_optical_spectrum",
                "canonical_response_kind",
                "cpge_part",
                "linear_injection_part",
                "optical_spectra_from_transition_table",
                "optical_transition_weight",
                "positive_optical_transition_terms",
                "precompute_optical_tensors",
            ),
        ),
        (
            "analysis.optical.kmesh",
            (
                "OpticalKPointData",
                "linear_conductivity_from_kpoint_data",
                "optical_kpoint_data_from_eigensystem",
                "shift_current_velocity_gauge_from_kpoint_data",
                "shg_conductivity_from_kpoint_data",
            ),
        ),
        (
            "analysis.optical.linear",
            (
                "E2_OVER_H_S",
                "E2_OVER_HBAR_S",
                "E_CHARGE_C",
                "H_PLANCK_J_S",
                "LinearConductivityResult",
                "linear_conductivity_tensor_from_gauge_data",
            ),
        ),
        (
            "analysis.optical.magneto",
            (
                "ComplexPolarization",
                "MagnetoOpticalAmplitudes",
                "NormalIncidenceSheetScattering",
                "PHASOR_CONVENTION",
                "REFLECTED_FIELD_CONVENTION",
                "SAMPLE_AXES_CONVENTION",
                "SmallConductivityAngles",
                "VACUUM_ADMITTANCE_S",
                "VACUUM_IMPEDANCE_OHM",
                "VACUUM_PERMEABILITY_H_PER_M",
                "VACUUM_PERMITTIVITY_F_PER_M",
                "complex_polarization_from_jones",
                "faraday_kerr_from_sheet_conductivity",
                "faraday_kerr_from_sheet_conductivity_liu_dai_2020_printed",
                "faraday_kerr_from_sheet_conductivity_on_substrate",
                "faraday_kerr_small_conductivity",
                "sheet_conductivity_from_normalized_transmission_matrix",
                "sheet_conductivity_from_scalar_normalized_transmission",
                "sheet_scattering_at_normal_incidence",
                "unwrap_polarization_angle",
            ),
        ),
        (
            "analysis.optical.passos",
            (
                "DerivativeProvenance",
                "OccupationKind",
                "PassosHarmonicConductivityResult",
                "PassosKPointData",
                "PassosSHGSusceptibilityResult",
                "PassosModelCertificate",
                "PassosOccupationSpec",
                "PassosPrimitiveBZGrid",
                "passos_harmonic_conductivity_from_kpoint_data",
                "passos_harmonic_kernel",
                "passos_kpoint_data_from_fixed_basis_derivatives",
                "passos_shg_conductivity_from_kpoint_data",
                "passos_shg_susceptibility",
                "passos_si_prefactor",
                "passos_thg_conductivity_from_kpoint_data",
            ),
        ),
        (
            "analysis.optical.harmonic",
            (
                "HarmonicFormulation",
                "HarmonicGenerationKind",
                "IndependentInputAdiabaticSwitching",
                "PassosFiniteBandVelocityGauge",
                "harmonic_response_from_kpoint_data",
                "shg_susceptibility_from_response",
            ),
        ),
        (
            "analysis.optical.kerr",
            (
                "KerrFaradayGeometry",
                "KerrFaradayResponse",
                "NormalIncidenceSameSideGeometry",
                "Okada2016MixedPulseGeometry",
                "OpticalKPointResponse",
                "SpectralRefractiveIndex",
                "kerr_faraday_from_kpoint_data",
                "kerr_faraday_from_kpoint_data_liu_dai_2020_printed",
                "kerr_faraday_from_kpoint_data_on_substrate",
                "kerr_faraday_from_linear_conductivity",
                "kerr_faraday_from_linear_conductivity_liu_dai_2020_printed",
                "kerr_faraday_from_linear_conductivity_on_substrate",
                "optical_response_from_kpoint_data",
            ),
        ),
        (
            "analysis.optical.shg",
            (
                "SHGConductivityResult",
                "ShiftCurrentVelocityGaugeResult",
                "LIU_DAI_2020_EQ4_PREFAC_A_M_PER_V2",
                "LIU_DAI_2020_EQ4_PREFAC_UA_NM_PER_V2",
                "LIU_DAI_2020_EQ4_SHG_PREFAC_S",
                "VELOCITY_GAUGE_2D_PREFAC_A_M_PER_V2",
                "VELOCITY_GAUGE_2D_PREFAC_UA_NM_PER_V2",
                "shift_current_conductivity_velocity_gauge",
                "shg_conductivity_velocity_gauge",
                "shg_susceptibility_from_conductivity",
            ),
        ),
    ),
    "analysis.optical.benchmarks": (
        (
            "analysis.optical.benchmarks.liu_dai_2020",
            (
                "C3TensorDecomposition",
                "c3_inplane_second_order_tensor",
                "decompose_c3_inplane_second_order_tensor",
                "qah_faraday_kerr_benchmark",
                "qah_faraday_kerr_small_benchmark",
                "qah_sheet_conductivity_tensor",
            ),
        ),
        (
            "analysis.optical.benchmarks.mikhailov_2016",
            (
                "IntegralMethod",
                "Mikhailov2016THGResult",
                "mikhailov2016_thg_xxxx_scattering",
            ),
        ),
        (
            "analysis.optical.benchmarks.okada_2016",
            (
                "OKADA_2016_INP_REFRACTIVE_INDEX",
                "OKADA_2016_DC_ESTIMATED_FARADAY_RAD",
                "OKADA_2016_DC_ESTIMATED_KERR_RAD",
                "OKADA_2016_MEASURED_FARADAY_RAD",
                "OKADA_2016_MEASURED_KERR_RAD",
                "Okada2016QAHBenchmark",
                "Okada2016ScalingUncertainty",
                "okada2016_qah_benchmark",
                "okada2016_qah_sheet_conductivity",
                "okada2016_scaling_from_dc_conductivity",
                "okada2016_scaling_function",
                "okada2016_scaling_uncertainty",
            ),
        ),
        (
            "analysis.optical.benchmarks.passos_2018",
            (
                "Passos2018GrapheneParameters",
                "RiceMeleParameters",
                "graphene_direct_lattice_vectors",
                "graphene_reciprocal_vectors",
                "iter_passos2018_graphene_kpoints",
                "iter_rice_mele_kpoints",
                "passos2018_graphene_grid",
                "passos2018_graphene_refined_grid",
                "passos2018_graphene_hamiltonian_and_derivatives",
                "passos2018_sigma0_si",
                "rice_mele_grid",
                "rice_mele_hamiltonian_and_derivatives",
            ),
        ),
        (
            "analysis.optical.benchmarks.qwz",
            (
                "qwz_hamiltonian_and_derivatives",
                "qwz_lower_band_chern_dvector",
                "qwz_lower_band_chern_number",
                "qwz_optical_kpoint_data",
            ),
        ),
    ),
}

_EXPECTED_COUNTS = {
    "analysis.shift_current": 45,
    "analysis.injection_current": 11,
    "analysis.optical": 91,
    "analysis.optical.benchmarks": 37,
}

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _expected_exports(package_name: str) -> tuple[str, ...]:
    return tuple(
        symbol
        for _owner_name, symbols in _PACKAGE_OWNERS[package_name]
        for symbol in symbols
    )


def _initializer_path(package_name: str) -> Path:
    return _REPOSITORY_ROOT / "src" / Path(*package_name.split(".")) / "__init__.py"


def test_analysis_namespaces_have_exact_ordered_leaf_exports_and_identity():
    for package_name, owners in _PACKAGE_OWNERS.items():
        package = importlib.import_module(package_name)
        expected = _expected_exports(package_name)
        assert len(expected) == _EXPECTED_COUNTS[package_name]
        assert package.__all__ == list(expected)
        assert len(set(package.__all__)) == len(package.__all__)

        for owner_name, symbols in owners:
            owner = importlib.import_module(owner_name)
            for symbol in symbols:
                exported = getattr(package, symbol)
                assert exported is getattr(owner, symbol)
                assert not isinstance(exported, ModuleType)


def test_analysis_namespace_star_imports_match_only_literal_exports():
    for package_name in _PACKAGE_OWNERS:
        expected = _expected_exports(package_name)
        namespace: dict[str, object] = {}
        exec(f"from {package_name} import *", namespace)
        namespace.pop("__builtins__")
        assert tuple(namespace) == expected

        package = importlib.import_module(package_name)
        for symbol, exported in namespace.items():
            assert exported is getattr(package, symbol)


def test_analysis_namespace_initializers_use_explicit_imports_and_literal_all():
    for package_name, owners in _PACKAGE_OWNERS.items():
        tree = ast.parse(_initializer_path(package_name).read_bytes())
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        expected_imports = [
            (owner_name.rsplit(".", 1)[1], symbols)
            for owner_name, symbols in owners
        ]
        actual_imports = [
            (
                node.module,
                tuple(alias.name for alias in node.names),
            )
            for node in imports
        ]
        assert actual_imports == expected_imports
        assert all(node.level == 1 for node in imports)
        assert all(alias.name != "*" and alias.asname is None for node in imports for alias in node.names)

        all_assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        ]
        assert len(all_assignments) == 1
        value = all_assignments[0].value
        assert isinstance(value, ast.List)
        assert ast.literal_eval(value) == list(_expected_exports(package_name))
