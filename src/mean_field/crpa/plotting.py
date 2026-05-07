from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .workflow import CRPAResult


def write_epsilon_vs_q_plot(result: CRPAResult, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q_abs = np.abs(result.physical_q_vectors.reshape(-1))
    q_abs_nm_inv = q_abs / (float(result.coulomb_params.graphene_lattice_angstrom) / 10.0)
    eps = np.real(result.effective_epsilon.reshape(-1)) * float(result.coulomb_params.epsilon_bn)
    order = np.argsort(q_abs_nm_inv)

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.scatter(q_abs_nm_inv[order], eps[order], s=12, linewidths=0.0, alpha=0.8)
    ax.set_xlabel(r"$|\mathbf{q}|$ (nm$^{-1}$)")
    ax.set_ylabel(r"$\epsilon(\mathbf{q})\,\epsilon_{\rm BN}$")
    ax.set_title("cRPA effective dielectric constant")
    ax.grid(alpha=0.25)
    fig.savefig(path)
    plt.close(fig)
    return path
