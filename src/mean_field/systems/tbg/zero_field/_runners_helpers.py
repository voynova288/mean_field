from __future__ import annotations

from ._runners_shared import *  # noqa: F401,F403

def _bm_unstrained_reference_map() -> dict[float, BMUnstrainedReference]:
    return {round(ref.theta_deg, 2): ref for ref in load_bm_unstrained_references()}


@lru_cache(maxsize=1)
def _bm_unstrained_runtime_reference_map() -> dict[float, BMRuntimeBenchmarkRecord]:
    return {round(ref.theta_deg, 2): ref for ref in load_bm_unstrained_runtime_benchmarks()}


@lru_cache(maxsize=1)
def _b0_parameter_reference_map() -> dict[float, ParameterReference]:
    return {round(ref.theta_deg, 2): ref for ref in load_b0_parameter_references()}


@lru_cache(maxsize=1)
def _b0_hf_runtime_reference_map() -> dict[str, RuntimeBenchmarkRecord]:
    return {row.benchmark_id: row for row in load_b0_runtime_benchmarks()}


def _ensure_tbg_zero_field_contract_sidecars_writable(
    root: Path | str,
    *,
    overwrite_contract_sidecars: bool,
) -> None:
    if overwrite_contract_sidecars:
        return
    result_root = Path(root)
    existing = [name for name in required_artifact_files() if (result_root / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing TBG zero-field contract sidecars in {result_root}: {existing}. "
            "Pass overwrite_contract_sidecars=True only when intentionally replacing these sidecars."
        )


def _full_init_density_override_paths(case: BenchmarkCase, init_mode: str, seed: int, *, lk: int | None = None) -> tuple[Path, ...]:
    normalized = normalize_full_init_mode(init_mode)
    candidates: list[Path] = []
    if lk is not None:
        candidates.append(case.case_dir / f"initial_density_{normalized}_seed_{int(seed):03d}_lk{int(lk)}.tsv")
    candidates.append(case.case_dir / f"initial_density_{normalized}_seed_{int(seed):03d}.tsv")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _full_init_mode_uses_stochastic_initialization(init_mode: str) -> bool:
    normalized = normalize_full_init_mode(init_mode)
    return normalized in {"random", "diag_random", "flavor"}


def _load_full_init_density_override(path: Path, *, nt: int, nk: int) -> np.ndarray:
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ik_s, row_s, col_s, real_s, imag_s = stripped.split("\t")
            density[int(row_s), int(col_s), int(ik_s)] = complex(float(real_s), float(imag_s))
    return density


def _inspect_full_init_density_override(path: Path) -> tuple[int, int, int] | None:
    max_ik = -1
    max_row = -1
    max_col = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ik_s, row_s, col_s, _real_s, _imag_s = stripped.split("\t")
            max_ik = max(max_ik, int(ik_s))
            max_row = max(max_row, int(row_s))
            max_col = max(max_col, int(col_s))
    if max_ik < 0:
        return None
    return max_row + 1, max_col + 1, max_ik + 1


def _load_required_full_init_density_override(
    case: BenchmarkCase,
    *,
    init_mode: str,
    seed: int,
    nt: int,
    nk: int,
    lk: int | None = None,
) -> np.ndarray | None:
    if _full_init_mode_uses_stochastic_initialization(init_mode):
        return None

    override_candidates = _full_init_density_override_paths(case, init_mode, seed, lk=lk)
    override_path = next((path for path in override_candidates if path.is_file()), None)
    if override_path is None:
        raise FileNotFoundError(
            f"Full benchmark case {case.benchmark_id} requires Julia initial density "
            f"{', '.join(path.name for path in override_candidates)}, but none of them exist."
        )

    override_dims = _inspect_full_init_density_override(override_path)
    expected_dims = (nt, nt, nk)
    if override_dims is None:
        raise ValueError(
            f"Full benchmark case {case.benchmark_id} has an empty initial density override: {override_path}"
        )
    if override_dims != expected_dims:
        raise ValueError(
            f"Full benchmark case {case.benchmark_id} expects initial density shape {expected_dims}, "
            f"but {override_path.name} contains {override_dims}."
        )

    return _load_full_init_density_override(
        override_path,
        nt=nt,
        nk=nk,
    )


def _load_explicit_full_init_density_override(path: Path | str, *, nt: int, nk: int) -> np.ndarray:
    override_path = Path(path)
    if not override_path.is_file():
        raise FileNotFoundError(f"Explicit initial density override does not exist: {override_path}")

    override_dims = _inspect_full_init_density_override(override_path)
    expected_dims = (nt, nt, nk)
    if override_dims is None:
        raise ValueError(f"Explicit initial density override is empty: {override_path}")
    if override_dims != expected_dims:
        raise ValueError(
            f"Explicit initial density override expects shape {expected_dims}, "
            f"but {override_path} contains {override_dims}."
        )

    return _load_full_init_density_override(override_path, nt=nt, nk=nk)


def _benchmark_grid_reference_uk_path(case: BenchmarkCase, *, lk: int, lg: int) -> Path:
    reference_path_getter = getattr(case, "bm_grid_reference_uk_path", None)
    if reference_path_getter is None:
        return Path()
    return reference_path_getter(lk=lk, lg=lg)


def _apply_benchmark_grid_reference_uk(case: BenchmarkCase, solution: BMSolution, *, lk: int, lg: int) -> BMSolution:
    reference_path = _benchmark_grid_reference_uk_path(case, lk=lk, lg=lg)
    if not reference_path.is_file():
        return solution
    reference_uk = load_complex_tensor4_tsv(reference_path, shape=solution.uk.shape)
    return solution.with_reference_uk(reference_uk)


def _should_apply_benchmark_grid_reference_uk(case: BenchmarkCase, *, lk: int, lg: int) -> bool:
    return _benchmark_grid_reference_uk_path(case, lk=lk, lg=lg).is_file()


def _build_benchmark_grid_solution(case: BenchmarkCase, params: TBGParameters, *, lk: int, lg: int) -> BMSolution:
    grid = build_b0_uniform_lattice(params, lk)
    solution = solve_bm_model(params, grid.kvec, lg=lg, sigma_rotation=True)
    if not _should_apply_benchmark_grid_reference_uk(case, lk=lk, lg=lg):
        return solution
    return _apply_benchmark_grid_reference_uk(case, solution, lk=lk, lg=lg)


def _benchmark_grid_cache_key(case: BenchmarkCase, *, lk: int, lg: int) -> tuple[float, int, int, str]:
    reference_key = ""
    if _should_apply_benchmark_grid_reference_uk(case, lk=lk, lg=lg):
        reference_key = str(_benchmark_grid_reference_uk_path(case, lk=lk, lg=lg))
    return (round(case.theta_deg, 2), int(lk), int(lg), reference_key)


def _write_key_value_summary(path: Path, entries: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")
    return path


def _format_ratio(value: float | None) -> str:
    return "" if value is None else str(value)


def _compare_bm_runtime_to_reference(runtime: BMUnstrainedRuntime, reference: BMRuntimeBenchmarkRecord) -> BMUnstrainedRuntimeParity:
    return BMUnstrainedRuntimeParity(
        path_elapsed_sec_delta=runtime.path_elapsed_sec - reference.path_elapsed_sec,
        path_elapsed_sec_ratio=safe_ratio(runtime.path_elapsed_sec, reference.path_elapsed_sec),
        grid_elapsed_sec_delta=runtime.grid_elapsed_sec - reference.grid_elapsed_sec,
        grid_elapsed_sec_ratio=safe_ratio(runtime.grid_elapsed_sec, reference.grid_elapsed_sec),
        total_elapsed_sec_delta=runtime.total_elapsed_sec - reference.total_elapsed_sec,
        total_elapsed_sec_ratio=safe_ratio(runtime.total_elapsed_sec, reference.total_elapsed_sec),
    )


def _compare_hf_runtime_to_reference(runtime: B0HFBenchmarkRuntime, reference: RuntimeBenchmarkRecord) -> B0HFBenchmarkRuntimeParity:
    return B0HFBenchmarkRuntimeParity(
        bm_elapsed_sec_delta=runtime.bm_elapsed_sec - reference.bm_elapsed_sec,
        bm_elapsed_sec_ratio=safe_ratio(runtime.bm_elapsed_sec, reference.bm_elapsed_sec),
        hf_elapsed_sec_delta=runtime.hf_elapsed_sec - reference.hf_elapsed_sec,
        hf_elapsed_sec_ratio=safe_ratio(runtime.hf_elapsed_sec, reference.hf_elapsed_sec),
        path_elapsed_sec_delta=runtime.path_elapsed_sec - reference.path_elapsed_sec,
        path_elapsed_sec_ratio=safe_ratio(runtime.path_elapsed_sec, reference.path_elapsed_sec),
        total_elapsed_sec_delta=runtime.total_elapsed_sec - reference.total_elapsed_sec,
        total_elapsed_sec_ratio=safe_ratio(runtime.total_elapsed_sec, reference.total_elapsed_sec),
    )


def _benchmark_runner_kind(case: BenchmarkCase) -> str:
    return case.load_runtime_summary().entries.get("runner_kind", "restricted").strip().lower()


def _run_benchmark_hf_from_grid_solution(
    case: BenchmarkCase,
    grid_solution: BMSolution,
    overlap_blocks: HFOverlapBlockSet,
    *,
    grid_lk: int,
    requested_init_mode: str,
    requested_seed: int,
    beta: float,
    max_iter: int,
    precision: float,
    overlap_lg: int | None = None,
    runner_kind: str | None = None,
    initial_density_override: np.ndarray | None = None,
) -> RestrictedHartreeFockRun:
    runner_kind = _benchmark_runner_kind(case) if runner_kind is None else str(runner_kind)
    if runner_kind == "full":
        initial_density = initial_density_override
        if initial_density is None and not _full_init_mode_uses_stochastic_initialization(requested_init_mode):
            nt = int(grid_solution.n_spin * grid_solution.n_eta * grid_solution.nb)
            initial_density = _load_required_full_init_density_override(
                case,
                init_mode=requested_init_mode,
                seed=requested_seed,
                nt=nt,
                nk=int(grid_solution.nk),
                lk=grid_lk,
            )
        state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=case.nu, precision=precision)
        if overlap_lg is not None:
            state.diagnostics["overlap_lg"] = float(overlap_lg)
        return run_full_hartree_fock(
            state,
            overlap_blocks,
            grid_solution.lattice_kvec,
            grid_solution.params,
            init_mode=requested_init_mode,
            seed=requested_seed,
            beta=beta,
            max_iter=max_iter,
            initial_density=initial_density,
        )
    if runner_kind == "restricted":
        state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=case.nu, precision=precision)
        if overlap_lg is not None:
            state.diagnostics["overlap_lg"] = float(overlap_lg)
        return run_restricted_hartree_fock(
            state,
            overlap_blocks,
            grid_solution.lattice_kvec,
            grid_solution.params,
            init_mode=requested_init_mode,
            seed=requested_seed,
            beta=beta,
            max_iter=max_iter,
        )
    raise ValueError(f"Unsupported benchmark runner_kind={runner_kind!r} for case {case.benchmark_id}")


def build_b0_reference_parameters(theta_deg: float) -> TBGParameters:
    reference = _b0_parameter_reference_map().get(round(theta_deg, 2))
    if reference is None:
        return TBGParameters.from_degrees(
            theta_deg,
            vf=2482.0,
            w0=77.0,
            w1=110.0,
            strain=0.0,
            alpha=0.5,
            deformation_potential=0.0,
        )

    return TBGParameters(
        dtheta_rad=reference.dtheta_rad,
        convention="b0",
        vf=reference.vf,
        w0=reference.w0,
        w1=reference.w1,
        strain=reference.strain,
        alpha=reference.alpha,
        deformation_potential=0.0,
    )


def _build_packaged_or_native_benchmark_kpath(params: TBGParameters, theta_deg: float, points_per_segment: int) -> KPath:
    reference = _bm_unstrained_reference_map().get(round(theta_deg, 2))
    if reference is None:
        return build_b0_benchmark_kpath(params, points_per_segment)

    summary = reference.load_summary()
    if int(summary["points_per_segment"]) != points_per_segment:
        return build_b0_benchmark_kpath(params, points_per_segment)

    return build_kpath_from_reference_nodes(reference.load_path_nodes())


def _build_packaged_or_native_hf_benchmark_kpath(case: BenchmarkCase, params: TBGParameters, points_per_segment: int) -> KPath:
    if points_per_segment != case.points_per_segment:
        return build_b0_benchmark_kpath(params, points_per_segment)
    return build_kpath_from_reference_nodes(case.load_reference_nodes())


def _build_advisor_hf_benchmark_kpath(
    benchmark_id: str,
    params: TBGParameters,
    *,
    lk: int,
    points_per_segment: int,
) -> tuple[KPath, object]:
    ranked = rank_kpath_candidates_for_lk(
        params,
        lk=lk,
        points_per_segment=points_per_segment,
    )
    if not ranked:
        raise ValueError(f"No advisor path candidates available for {benchmark_id}.")
    compatibility = ranked[0]
    return compatibility.candidate.path, compatibility


def _write_advisor_path_selection(path: Path, *, compatibility: object) -> Path:
    candidate = compatibility.candidate
    return _write_key_value_summary(
        path,
        [
            ("path_source", "advisor_ranked_candidate"),
            ("candidate_rank", "1"),
            ("candidate_name", str(candidate.name)),
            ("candidate_family", str(candidate.family)),
            ("lk", str(int(compatibility.lk))),
            ("exact_count", str(int(compatibility.exact_count))),
            ("exact_node_hit_count", str(int(compatibility.exact_node_hit_count))),
            ("exact_segment_counts", ",".join(str(value) for value in compatibility.exact_segment_counts)),
            ("mean_nearest_distance", f"{float(compatibility.mean_nearest_distance):.16e}"),
            ("max_nearest_distance", f"{float(compatibility.max_nearest_distance):.16e}"),
            ("m_real", f"{float(candidate.m_point.real):.16f}"),
            ("m_imag", f"{float(candidate.m_point.imag):.16f}"),
            ("k_real", f"{float(candidate.k_point.real):.16f}"),
            ("k_imag", f"{float(candidate.k_point.imag):.16f}"),
        ],
    )

__all__ = [name for name in globals() if not name.startswith('__')]
