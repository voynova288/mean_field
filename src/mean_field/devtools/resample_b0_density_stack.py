from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mean_field.benchmarks import load_complex_stack_tsv


def _infer_lk_from_nk(nk: int) -> int:
    side = int(round(np.sqrt(nk)))
    if side * side != nk or side < 2:
        raise ValueError(f"Cannot infer a B0 square-grid lk from nk={nk}")
    return side - 1


def _density_to_grid(density: np.ndarray, *, lk: int) -> np.ndarray:
    side = lk + 1
    if density.ndim != 3:
        raise ValueError(f"Expected density stack with 3 axes, got shape {density.shape}")
    if density.shape[2] != side * side:
        raise ValueError(f"Expected nk={(side * side)} for lk={lk}, got {density.shape[2]}")
    grid = np.empty((density.shape[0], density.shape[1], side, side), dtype=np.complex128)
    for j in range(side):
        for i in range(side):
            grid[:, :, i, j] = density[:, :, i + j * side]
    return grid


def _grid_to_density(grid: np.ndarray) -> np.ndarray:
    side = grid.shape[2]
    density = np.empty((grid.shape[0], grid.shape[1], side * side), dtype=np.complex128)
    for j in range(side):
        for i in range(side):
            density[:, :, i + j * side] = grid[:, :, i, j]
    return density


def resample_density_stack(
    density: np.ndarray,
    *,
    source_lk: int,
    target_lk: int,
    method: str = "bilinear",
    hermitize: bool = True,
) -> np.ndarray:
    """Resample a B0 full-HF density stack between inclusive square k-grids."""

    source_lk = int(source_lk)
    target_lk = int(target_lk)
    method = str(method).lower()
    if method not in {"nearest", "bilinear"}:
        raise ValueError(f"Unsupported resampling method: {method}")
    if source_lk < 1 or target_lk < 1:
        raise ValueError(f"Expected positive lk values, got source_lk={source_lk}, target_lk={target_lk}")

    source_grid = _density_to_grid(np.asarray(density, dtype=np.complex128), lk=source_lk)
    target_side = target_lk + 1
    target_grid = np.empty((density.shape[0], density.shape[1], target_side, target_side), dtype=np.complex128)

    for j_target in range(target_side):
        y = (j_target / target_lk) * source_lk
        j0 = int(np.floor(y))
        wy = float(y - j0)
        if j0 >= source_lk:
            j0 = source_lk
            j1 = source_lk
            wy = 0.0
        else:
            j1 = j0 + 1

        for i_target in range(target_side):
            x = (i_target / target_lk) * source_lk
            i0 = int(np.floor(x))
            wx = float(x - i0)
            if i0 >= source_lk:
                i0 = source_lk
                i1 = source_lk
                wx = 0.0
            else:
                i1 = i0 + 1

            if method == "nearest":
                source_i = i1 if wx >= 0.5 else i0
                source_j = j1 if wy >= 0.5 else j0
                block = source_grid[:, :, source_i, source_j]
            else:
                block = (
                    (1.0 - wx) * (1.0 - wy) * source_grid[:, :, i0, j0]
                    + wx * (1.0 - wy) * source_grid[:, :, i1, j0]
                    + (1.0 - wx) * wy * source_grid[:, :, i0, j1]
                    + wx * wy * source_grid[:, :, i1, j1]
                )
            if hermitize:
                block = 0.5 * (block + block.conj().T)
            target_grid[:, :, i_target, j_target] = block

    return _grid_to_density(target_grid)


def write_complex_stack_tsv(path: Path, stack: np.ndarray, *, metadata: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in metadata.items():
            handle.write(f"# {key}={value}\n")
        handle.write(f"# nrow={stack.shape[0]}\n")
        handle.write(f"# ncol={stack.shape[1]}\n")
        handle.write(f"# nk={stack.shape[2]}\n")
        for ik in range(stack.shape[2]):
            for row in range(stack.shape[0]):
                for col in range(stack.shape[1]):
                    value = stack[row, col, ik]
                    handle.write(f"{ik}\t{row}\t{col}\t{value.real:.17e}\t{value.imag:.17e}\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resample a B0 full-HF density TSV between inclusive lk grids.")
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument("output_tsv", type=Path)
    parser.add_argument("--source-lk", type=int, default=None, help="Source lk. Inferred from nk if omitted.")
    parser.add_argument("--target-lk", type=int, required=True)
    parser.add_argument("--method", choices=("bilinear", "nearest"), default="bilinear")
    parser.add_argument("--no-hermitize", action="store_true", help="Do not symmetrize each resampled density block.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    density = load_complex_stack_tsv(args.input_tsv)
    source_lk = _infer_lk_from_nk(density.shape[2]) if args.source_lk is None else int(args.source_lk)
    resampled = resample_density_stack(
        density,
        source_lk=source_lk,
        target_lk=args.target_lk,
        method=args.method,
        hermitize=not args.no_hermitize,
    )
    write_complex_stack_tsv(
        args.output_tsv,
        resampled,
        metadata={
            "source": "resample_b0_density_stack",
            "input_tsv": str(args.input_tsv),
            "source_lk": str(source_lk),
            "target_lk": str(args.target_lk),
            "method": args.method,
            "hermitize": str(not args.no_hermitize).lower(),
        },
    )
    print(f"input_tsv={args.input_tsv}")
    print(f"output_tsv={args.output_tsv}")
    print(f"source_lk={source_lk}")
    print(f"target_lk={args.target_lk}")
    print(f"source_nk={density.shape[2]}")
    print(f"target_nk={resampled.shape[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
