from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import B0_BENCHMARK_ROOT
from .paths import BM_UNSTRAINED_BENCHMARK_ROOT


@dataclass(frozen=True)
class PathNodeReference:
    label: str
    index: int
    k_dist: float
    kx: float
    ky: float

    @property
    def kvec(self) -> complex:
        return complex(self.kx, self.ky)


@dataclass(frozen=True)
class ParameterReference:
    theta_deg: float
    dtheta_rad: float
    vf: float
    w0: float
    w1: float
    strain: float
    alpha: float
    kb: float
    g1: complex
    g2: complex
    a1: complex
    a2: complex
    theta12: float
    kt: complex
    kb_point: complex


def _load_key_value_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key] = value
    return data


def _load_path_data(path: Path) -> tuple[list[float], list[list[float]]]:
    kdist: list[float] = []
    energies: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = [float(item) for item in line.strip().split("\t")]
            kdist.append(fields[0])
            energies.append(fields[1:])
    return kdist, energies


@dataclass(frozen=True)
class HFPathReference:
    band_labels: tuple[str, ...]
    kdist: tuple[float, ...]
    energies: np.ndarray


@dataclass(frozen=True)
class HFParitySummary:
    reference_path_tsv: str
    kdist_max_abs_diff: float
    max_abs_band_diff_mev: float
    rms_band_diff_mev: float
    mean_abs_band_diff_mev: float
    energy_sorting: str


@dataclass(frozen=True)
class RuntimeSummary:
    entries: dict[str, str]


@dataclass(frozen=True)
class RuntimeBenchmarkRecord:
    benchmark_id: str
    theta_deg: float
    nu: int
    init_mode: str
    lk: int
    lg: int
    bm_elapsed_sec: float
    hf_elapsed_sec: float
    path_elapsed_sec: float
    total_elapsed_sec: float
    hostname: str
    cpu_model: str
    slurm_partition: str
    slurm_nodelist: str
    slurm_cpus_per_task: int
    blas_threads: int
    sys_cpu_threads: int
    julia_version: str


@dataclass(frozen=True)
class BMRuntimeBenchmarkRecord:
    theta_deg: float
    points_per_segment: int
    lg: int
    grid_lk: int
    path_elapsed_sec: float
    grid_elapsed_sec: float
    total_elapsed_sec: float
    hostname: str
    cpu_model: str
    slurm_partition: str
    slurm_nodelist: str
    slurm_cpus_per_task: int
    blas_threads: int
    sys_cpu_threads: int
    julia_version: str


@dataclass(frozen=True)
class BMUnstrainedReference:
    theta_deg: float
    root: Path
    summary_path: Path
    path_nodes_path: Path
    path_tsv_path: Path
    grid_kvec_path: Path

    @property
    def runtime_summary_path(self) -> Path:
        theta_code = int(round(self.theta_deg * 100.0))
        return self.root / f"theta_{theta_code:03d}_unstrained_runtime_summary.txt"

    def load_summary(self) -> dict[str, str]:
        return _load_key_value_file(self.summary_path)

    def load_path_nodes(self) -> tuple[PathNodeReference, ...]:
        with self.path_nodes_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return tuple(
                PathNodeReference(
                    label=row["label"],
                    index=int(row["index"]),
                    k_dist=float(row["k_dist"]),
                    kx=float(row["kx"]),
                    ky=float(row["ky"]),
                )
                for row in reader
            )

    def load_path_data(self) -> tuple[list[float], list[list[float]]]:
        return _load_path_data(self.path_tsv_path)

    def load_grid_kvec(self) -> tuple[complex, ...]:
        values: list[complex] = []
        with self.grid_kvec_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                kx, ky = [float(item) for item in line.strip().split("\t")]
                values.append(complex(kx, ky))
        return tuple(values)

    def load_runtime_summary(self) -> RuntimeSummary:
        return RuntimeSummary(entries=_load_key_value_file(self.runtime_summary_path))


@dataclass(frozen=True)
class OverlapReference:
    theta_deg: float
    lattice_kind: str
    valley_label: str
    m: int
    n: int
    fro_norm: float
    max_abs: float
    trace_real: float
    trace_imag: float
    entry_11_real: float
    entry_11_imag: float
    entry_mid_real: float
    entry_mid_imag: float


@dataclass(frozen=True)
class BenchmarkCase:
    benchmark_id: str
    theta_deg: float
    nu: int
    state_label: str
    description: str
    source_group: str
    source_path_tsv: str
    source_nodes_tsv: str
    source_summary_txt: str
    source_hf_jld2: str
    init_mode: str
    seed: int
    lk: int
    lg: int
    points_per_segment: int
    mu_mev: float
    exit_reason: str
    benchmark_case_dir: str

    @property
    def case_dir(self) -> Path:
        return B0_BENCHMARK_ROOT / "cases" / self.benchmark_id

    @property
    def reference_nodes_path(self) -> Path:
        return self.case_dir / "reference_nodes.tsv"

    @property
    def reference_summary_path(self) -> Path:
        return self.case_dir / "reference_summary.txt"

    @property
    def reference_path_tsv_path(self) -> Path:
        return self.case_dir / "reference_hf_path.tsv"

    @property
    def parity_summary_path(self) -> Path:
        return self.case_dir / "parity_to_reference_summary.txt"

    @property
    def runtime_summary_path(self) -> Path:
        return self.case_dir / "runtime_summary.txt"

    def initial_density_override_path(self, init_mode: str | None = None, seed: int | None = None) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"initial_density_{mode}_seed_{seed_value:03d}.tsv"

    def reference_first_iteration_interaction_path(self, init_mode: str | None = None, seed: int | None = None) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_first_iteration_interaction_{mode}_seed_{seed_value:03d}.tsv"

    def reference_first_iteration_hamiltonian_path(self, init_mode: str | None = None, seed: int | None = None) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_first_iteration_hamiltonian_{mode}_seed_{seed_value:03d}.tsv"

    def reference_first_iteration_density_path(self, init_mode: str | None = None, seed: int | None = None) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_first_iteration_density_{mode}_seed_{seed_value:03d}.tsv"

    def reference_first_iteration_summary_path(self, init_mode: str | None = None, seed: int | None = None) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_first_iteration_summary_{mode}_seed_{seed_value:03d}.txt"

    def reference_iteration_input_density_path(
        self,
        iteration: int,
        init_mode: str | None = None,
        seed: int | None = None,
    ) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_iteration_{iteration:03d}_input_density_{mode}_seed_{seed_value:03d}.tsv"

    def reference_iteration_interaction_path(
        self,
        iteration: int,
        init_mode: str | None = None,
        seed: int | None = None,
    ) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_iteration_{iteration:03d}_interaction_{mode}_seed_{seed_value:03d}.tsv"

    def reference_iteration_hamiltonian_path(
        self,
        iteration: int,
        init_mode: str | None = None,
        seed: int | None = None,
    ) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_iteration_{iteration:03d}_hamiltonian_{mode}_seed_{seed_value:03d}.tsv"

    def reference_iteration_updated_density_path(
        self,
        iteration: int,
        init_mode: str | None = None,
        seed: int | None = None,
    ) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_iteration_{iteration:03d}_updated_density_{mode}_seed_{seed_value:03d}.tsv"

    def reference_iteration_summary_path(
        self,
        iteration: int,
        init_mode: str | None = None,
        seed: int | None = None,
    ) -> Path:
        mode = self.init_mode if init_mode is None else str(init_mode)
        seed_value = self.seed if seed is None else int(seed)
        return self.case_dir / f"reference_iteration_{iteration:03d}_summary_{mode}_seed_{seed_value:03d}.txt"

    def bm_grid_reference_uk_path(self, *, lk: int | None = None, lg: int | None = None) -> Path:
        theta_tag = _theta_tag(self.theta_deg)
        lk_value = self.lk if lk is None else int(lk)
        lg_value = self.lg if lg is None else int(lg)
        return B0_BENCHMARK_ROOT / "bm_inputs" / f"bm_theta_{theta_tag}_lk{lk_value}_lg{lg_value}_uk_reference.tsv"

    def bm_grid_reference_uk_summary_path(self, *, lk: int | None = None, lg: int | None = None) -> Path:
        theta_tag = _theta_tag(self.theta_deg)
        lk_value = self.lk if lk is None else int(lk)
        lg_value = self.lg if lg is None else int(lg)
        return B0_BENCHMARK_ROOT / "bm_inputs" / f"bm_theta_{theta_tag}_lk{lk_value}_lg{lg_value}_uk_reference_summary.txt"

    def load_reference_nodes(self) -> tuple[PathNodeReference, ...]:
        with self.reference_nodes_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return tuple(
                PathNodeReference(
                    label=row["label"],
                    index=int(row["index"]),
                    k_dist=float(row["k_dist"]),
                    kx=float(row["kx"]),
                    ky=float(row["ky"]),
                )
                for row in reader
            )

    def load_reference_summary(self) -> dict[str, str]:
        return _load_key_value_file(self.reference_summary_path)

    def load_reference_path(self) -> HFPathReference:
        with self.reference_path_tsv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
            rows = [row for row in reader if row]
        kdist = [float(row[0]) for row in rows]
        energies = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=float)
        return HFPathReference(
            band_labels=tuple(header[1:]),
            kdist=tuple(kdist),
            energies=energies,
        )

    def load_parity_summary(self) -> HFParitySummary:
        data = _load_key_value_file(self.parity_summary_path)
        return HFParitySummary(
            reference_path_tsv=data["reference_path_tsv"],
            kdist_max_abs_diff=float(data["kdist_max_abs_diff"]),
            max_abs_band_diff_mev=float(data["max_abs_band_diff_mev"]),
            rms_band_diff_mev=float(data["rms_band_diff_mev"]),
            mean_abs_band_diff_mev=float(data["mean_abs_band_diff_mev"]),
            energy_sorting=data["energy_sorting"],
        )

    def load_runtime_summary(self) -> RuntimeSummary:
        return RuntimeSummary(entries=_load_key_value_file(self.runtime_summary_path))


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    root: Path
    manifest_path: Path
    cases: tuple[BenchmarkCase, ...]

    @property
    def runtime_benchmark_path(self) -> Path:
        return self.root / "runtime_benchmark.tsv"

    def get(self, benchmark_id: str) -> BenchmarkCase:
        for case in self.cases:
            if case.benchmark_id == benchmark_id:
                return case
        raise KeyError(f"Unknown benchmark id: {benchmark_id}")


def _as_case(row: dict[str, str]) -> BenchmarkCase:
    return BenchmarkCase(
        benchmark_id=row["benchmark_id"],
        theta_deg=float(row["theta_deg"]),
        nu=int(row["nu"]),
        state_label=row["state_label"],
        description=row["description"],
        source_group=row["source_group"],
        source_path_tsv=row["source_path_tsv"],
        source_nodes_tsv=row["source_nodes_tsv"],
        source_summary_txt=row["source_summary_txt"],
        source_hf_jld2=row["source_hf_jld2"],
        init_mode=row["init_mode"],
        seed=int(row["seed"]),
        lk=int(row["lk"]),
        lg=int(row["lg"]),
        points_per_segment=int(row["points_per_segment"]),
        mu_mev=float(row["mu_mev"]),
        exit_reason=row["exit_reason"],
        benchmark_case_dir=row["benchmark_case_dir"],
    )


def load_b0_suite(root: Path | None = None) -> BenchmarkSuite:
    root = B0_BENCHMARK_ROOT if root is None else Path(root)
    manifest_path = root / "benchmark_manifest.tsv"
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        cases = tuple(_as_case(row) for row in reader)
    return BenchmarkSuite(name="b0", root=root, manifest_path=manifest_path, cases=cases)


def _theta_tag(theta_deg: float) -> str:
    return f"{int(round(theta_deg * 100.0)):03d}"


def load_b0_parameter_references(root: Path | None = None) -> tuple[ParameterReference, ...]:
    root = B0_BENCHMARK_ROOT if root is None else Path(root)
    path = root / "parameter_reference.tsv"
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return tuple(
            ParameterReference(
                theta_deg=float(row["theta_deg"]),
                dtheta_rad=float(row["dtheta_rad"]),
                vf=float(row["vf"]),
                w0=float(row["w0"]),
                w1=float(row["w1"]),
                strain=float(row["strain"]),
                alpha=float(row["alpha"]),
                kb=float(row["kb"]),
                g1=complex(float(row["g1_re"]), float(row["g1_im"])),
                g2=complex(float(row["g2_re"]), float(row["g2_im"])),
                a1=complex(float(row["a1_re"]), float(row["a1_im"])),
                a2=complex(float(row["a2_re"]), float(row["a2_im"])),
                theta12=float(row["theta12"]),
                kt=complex(float(row["kt_re"]), float(row["kt_im"])),
                kb_point=complex(float(row["kb_re"]), float(row["kb_im"])),
            )
            for row in reader
        )


def load_b0_runtime_benchmarks(root: Path | None = None) -> tuple[RuntimeBenchmarkRecord, ...]:
    root = B0_BENCHMARK_ROOT if root is None else Path(root)
    path = root / "runtime_benchmark.tsv"
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return tuple(
            RuntimeBenchmarkRecord(
                benchmark_id=row["benchmark_id"],
                theta_deg=float(row["theta_deg"]),
                nu=int(row["nu"]),
                init_mode=row["init_mode"],
                lk=int(row["lk"]),
                lg=int(row["lg"]),
                bm_elapsed_sec=float(row["bm_elapsed_sec"]),
                hf_elapsed_sec=float(row["hf_elapsed_sec"]),
                path_elapsed_sec=float(row["path_elapsed_sec"]),
                total_elapsed_sec=float(row["total_elapsed_sec"]),
                hostname=row["hostname"],
                cpu_model=row["cpu_model"],
                slurm_partition=row["slurm_partition"],
                slurm_nodelist=row["slurm_nodelist"],
                slurm_cpus_per_task=int(row["slurm_cpus_per_task"]),
                blas_threads=int(row["blas_threads"]),
                sys_cpu_threads=int(row["sys_cpu_threads"]),
                julia_version=row["julia_version"],
            )
            for row in reader
        )


def load_bm_unstrained_references(root: Path | None = None) -> tuple[BMUnstrainedReference, ...]:
    root = BM_UNSTRAINED_BENCHMARK_ROOT if root is None else Path(root)
    refs: list[BMUnstrainedReference] = []
    for theta_code in (120, 128):
        refs.append(
            BMUnstrainedReference(
                theta_deg=theta_code / 100.0,
                root=root,
                summary_path=root / f"theta_{theta_code:03d}_unstrained_summary.txt",
                path_nodes_path=root / f"theta_{theta_code:03d}_unstrained_path_nodes.tsv",
                path_tsv_path=root / f"theta_{theta_code:03d}_unstrained_path.tsv",
                grid_kvec_path=root / f"theta_{theta_code:03d}_unstrained_grid_lk33_kvec.tsv",
            )
        )
    return tuple(refs)


def load_bm_unstrained_overlap_references(root: Path | None = None) -> tuple[OverlapReference, ...]:
    root = BM_UNSTRAINED_BENCHMARK_ROOT if root is None else Path(root)
    path = root / "overlap_reference_path.tsv"
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return tuple(
            OverlapReference(
                theta_deg=float(row["theta_deg"]),
                lattice_kind=row["lattice_kind"],
                valley_label=row["valley_label"],
                m=int(row["m"]),
                n=int(row["n"]),
                fro_norm=float(row["fro_norm"]),
                max_abs=float(row["max_abs"]),
                trace_real=float(row["trace_real"]),
                trace_imag=float(row["trace_imag"]),
                entry_11_real=float(row["entry_11_real"]),
                entry_11_imag=float(row["entry_11_imag"]),
                entry_mid_real=float(row["entry_mid_real"]),
                entry_mid_imag=float(row["entry_mid_imag"]),
            )
            for row in reader
        )


def load_bm_unstrained_runtime_benchmarks(root: Path | None = None) -> tuple[BMRuntimeBenchmarkRecord, ...]:
    root = BM_UNSTRAINED_BENCHMARK_ROOT if root is None else Path(root)
    path = root / "runtime_benchmark.tsv"
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return tuple(
            BMRuntimeBenchmarkRecord(
                theta_deg=float(row["theta_deg"]),
                points_per_segment=int(row["points_per_segment"]),
                lg=int(row["lg"]),
                grid_lk=int(row["grid_lk"]),
                path_elapsed_sec=float(row["path_elapsed_sec"]),
                grid_elapsed_sec=float(row["grid_elapsed_sec"]),
                total_elapsed_sec=float(row["total_elapsed_sec"]),
                hostname=row["hostname"],
                cpu_model=row["cpu_model"],
                slurm_partition=row["slurm_partition"],
                slurm_nodelist=row["slurm_nodelist"],
                slurm_cpus_per_task=int(row["slurm_cpus_per_task"]),
                blas_threads=int(row["blas_threads"]),
                sys_cpu_threads=int(row["sys_cpu_threads"]),
                julia_version=row["julia_version"],
            )
            for row in reader
        )


def load_complex_stack_tsv(path: Path | str, *, shape: tuple[int, int, int] | None = None) -> np.ndarray:
    path = Path(path)
    entries: list[tuple[int, int, int, complex]] = []
    max_row = -1
    max_col = -1
    max_k = -1

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ik_s, row_s, col_s, real_s, imag_s = stripped.split("\t")
            ik = int(ik_s)
            row = int(row_s)
            col = int(col_s)
            value = complex(float(real_s), float(imag_s))
            entries.append((ik, row, col, value))
            max_k = max(max_k, ik)
            max_row = max(max_row, row)
            max_col = max(max_col, col)

    if shape is None:
        if max_row < 0 or max_col < 0 or max_k < 0:
            raise ValueError(f"No matrix entries found in {path}")
        shape = (max_row + 1, max_col + 1, max_k + 1)

    stack = np.zeros(shape, dtype=np.complex128)
    for ik, row, col, value in entries:
        if ik >= shape[2] or row >= shape[0] or col >= shape[1]:
            raise ValueError(f"Entry {(ik, row, col)} exceeds declared shape {shape} in {path}")
        stack[row, col, ik] = value
    return stack


def load_complex_tensor4_tsv(path: Path | str, *, shape: tuple[int, int, int, int] | None = None) -> np.ndarray:
    path = Path(path)
    entries: list[tuple[int, int, int, int, complex]] = []
    max_i0 = -1
    max_i1 = -1
    max_i2 = -1
    max_i3 = -1

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            i0_s, i1_s, i2_s, i3_s, real_s, imag_s = stripped.split("\t")
            i0 = int(i0_s)
            i1 = int(i1_s)
            i2 = int(i2_s)
            i3 = int(i3_s)
            value = complex(float(real_s), float(imag_s))
            entries.append((i0, i1, i2, i3, value))
            max_i0 = max(max_i0, i0)
            max_i1 = max(max_i1, i1)
            max_i2 = max(max_i2, i2)
            max_i3 = max(max_i3, i3)

    if shape is None:
        if max_i0 < 0 or max_i1 < 0 or max_i2 < 0 or max_i3 < 0:
            raise ValueError(f"No tensor entries found in {path}")
        shape = (max_i0 + 1, max_i1 + 1, max_i2 + 1, max_i3 + 1)

    tensor = np.zeros(shape, dtype=np.complex128)
    for i0, i1, i2, i3, value in entries:
        if i0 >= shape[0] or i1 >= shape[1] or i2 >= shape[2] or i3 >= shape[3]:
            raise ValueError(f"Entry {(i0, i1, i2, i3)} exceeds declared shape {shape} in {path}")
        tensor[i0, i1, i2, i3] = value
    return tensor
