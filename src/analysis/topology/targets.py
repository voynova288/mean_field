from __future__ import annotations

"""Static inventory of reusable quantum-geometry/topology reproduction targets.

This inventory is deliberately lightweight: it records which saved Mean_Field
artifacts are useful starting points for Chern/Berry-curvature/quantum-geometry
work.  It does not run Hamiltonian solves or assert that every paper panel is
fully reproduced.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ReuseLevel = Literal["direct_saved", "saved_state", "requires_recompute", "diagnostic_only"]


@dataclass(frozen=True)
class GeometryReproductionTarget:
    """A paper/case whose saved artifacts can seed common topology analysis."""

    system: str
    paper_target: str
    artifact_root: str
    quantities: tuple[str, ...]
    reuse_level: ReuseLevel
    evidence: str
    caveat: str = ""

    def root_path(self, repo_root: str | Path) -> Path:
        return Path(repo_root) / self.artifact_root

    def to_dict(self) -> dict[str, object]:
        return {
            "system": self.system,
            "paper_target": self.paper_target,
            "artifact_root": self.artifact_root,
            "quantities": list(self.quantities),
            "reuse_level": self.reuse_level,
            "evidence": self.evidence,
            "caveat": self.caveat,
        }


def known_geometry_reproduction_targets() -> tuple[GeometryReproductionTarget, ...]:
    """Return known reusable targets from the local paper/case inventory."""

    return (
        GeometryReproductionTarget(
            system="tMBG",
            paper_target="Fig. 2 Chern bands and saved Berry curvature",
            artifact_root="results/TMBG/tmbg_fig2_chern_paperpath_bothvalleys_sewn_final_20260427_105114",
            quantities=("Chern number", "FHS Berry curvature", "Berry-curvature maps"),
            reuse_level="direct_saved",
            evidence="Saved chern_numbers.json status=pass for Delta=0,-40,+60 meV; berry_curvature.npz exists for panels.",
            caveat="Full quantum-metric/QGT maps require regenerating topology-grid eigenvectors because old artifacts mostly saved flux/path data, not all wavefunctions.",
        ),
        GeometryReproductionTarget(
            system="TDBG",
            paper_target="Fig. 3 Chern band panels",
            artifact_root="results/TDBG/tdbg_fig3_chern_20260425_theta133_mesh21_open_valleypath",
            quantities=("Chern number", "band/path reproduction"),
            reuse_level="direct_saved",
            evidence="fig3_chern_report.md and summary.json record AB-AB/AB-BA central-pair and isolated-band Chern values.",
            caveat="Exact new-framework metric/QGT maps require regenerating wavefunctions from saved model parameters.",
        ),
        GeometryReproductionTarget(
            system="HTG",
            paper_target="Fig. 2b / Fig. 3b Chern-basis bands",
            artifact_root="results/HTG/htg_fig2b_fig3b_alpha2_1p197_paper_axes_20260429_165908",
            quantities=("Chern-basis Chern numbers", "central-subspace topology"),
            reuse_level="direct_saved",
            evidence="validation_report.md and chern_numbers.json give chern_a=-1, chern_b=-2, total=-3 with machine-precision residuals.",
            caveat="Quantum metric can be computed with compute_quantum_geometry after rebuilding the same central Chern-basis wavefunction mesh.",
        ),
        GeometryReproductionTarget(
            system="tMBG projected HF",
            paper_target="Polshyn 2021 Fig. S1 split C=2 band and Berry-curvature panels",
            artifact_root="results/TMBG_Polshyn2021_figS1/final/chern_checks",
            quantities=("HF Chern numbers", "Berry-curvature maps", "target two-band subspace C=+2"),
            reuse_level="saved_state",
            evidence="README.md records S1b/S1c split target HF bands C=+1,+1 and target subspace C=+2; S1c berry_curvature_panels.npz exists.",
            caveat="Individual remote HF band Chern numbers on coarse k9 are diagnostic because remote bands are mixed/near-degenerate.",
        ),
        GeometryReproductionTarget(
            system="TBG shift current",
            paper_target="Chaudhary 2021 quantum-geometry integrand maps",
            artifact_root="results/shift_current_tbg/chaudhary2021_quantum_geometry_audit_lg7_m55",
            quantities=("shift vector", "|A|^2 S integrand", "gauge-safe Wilson-link check"),
            reuse_level="diagnostic_only",
            evidence="summary.json/quantum_geometry_audit.npz validate the local gauge-free response integrand against a Wilson-link finite-difference check.",
            caveat="This is response/transition quantum geometry, not a completed Fig. 2/Fig. 4 paper reproduction; Hartree convention gaps remain.",
        ),
        GeometryReproductionTarget(
            system="tMoTe2",
            paper_target="Zhang et al. 2025 Fig. 3 local Berry curvature and quantum-metric trace at 2.00 deg",
            artifact_root="/data/home/ziyuzhu/VMC_AI/NQS_FQHE/results/zhang2025_tmote2_fig3_qg_checkpoint_20260605",
            quantities=(
                "paper-scaled Berry curvature A_mBZ*Omega/(2pi)",
                "paper-scaled quantum-metric trace A_mBZ*Tr[g]/(2pi)",
                "full FS/quantum-metric matrix components g_xx/g_xy/g_yy diagnostics",
                "Chern (+1,+1,+1)",
            ),
            reuse_level="direct_saved",
            evidence=(
                "summary.json reports paper convention paper Omega=-common Omega, averaged Chern "
                "(0.9997969,0.9994884,0.9987717), and integrated Tr[g] "
                "(1.0432151,3.1293262,5.2261800) versus paper model target (1.04,3.13,5.26)."
            ),
            caveat=(
                "This validates the Zhang2025 continuum-model column, not raw DFT/Wannier difference maps; "
                "the paper publishes Tr[g] rather than separate g_xx/g_xy/g_yy panels."
            ),
        ),
        GeometryReproductionTarget(
            system="RLG/hBN HF",
            paper_target="Kwan et al. 2312.11617 Fig. 6 Berry curvature + Fubini-Study metric trace",
            artifact_root="results/RnG_hBN",
            quantities=(
                "HF saved states",
                "occupied-conduction Berry curvature f(k)",
                "Fubini-Study metric trace tr[g_FS(k)]",
                "trace-condition violation",
                "real-space density candidate input",
            ),
            reuse_level="saved_state",
            evidence=(
                "The work plan identifies Fig. 6 as HF bands + Berry curvature + Fubini-Study + density; "
                "saved RLG/hBN Fig. 6 HF/Chern artifacts and the user's VP mean-field Fig. 6 result can seed the metric maps."
            ),
            caveat=(
                "For the paper metric, reconstruct the full microscopic HF wavefunction from active-HF eigenvectors and screened-basis "
                "Bloch functions before calling compute_quantum_geometry; do not compute FS metric from active W(k) alone."
            ),
        ),
        GeometryReproductionTarget(
            system="RLG/hBN single-particle",
            paper_target="MFCI-II 2311.12920 Fig. 11/13 Berry curvature and integrated Fubini-Study metric G",
            artifact_root="reference/2311.12920v1.pdf",
            quantities=(
                "single-band Berry curvature",
                "Fubini-Study metric maps",
                "integrated Fubini-Study metric G",
                "C=0 but nontrivial quantum geometry diagnostics",
            ),
            reuse_level="requires_recompute",
            evidence=(
                "Paper text records Fig. 11 Fubini-Study maps and Fig. 13 integrated G targets "
                "including G=2.2 for xi=1 and G=2.38 for xi=0 at V=-20 meV."
            ),
            caveat="Requires regenerating the appropriate single-particle RLG/hBN wavefunction mesh from the continuum model.",
        ),
    )


__all__ = ["GeometryReproductionTarget", "known_geometry_reproduction_targets"]
