from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    ReferenceDensity as ContractReferenceDensity,
    SingleParticleModel as ContractSingleParticleModel,
)
from mean_field.core.io import write_json_artifact, write_npz_artifact

from .artifacts import ArtifactManifest, ConventionBundle, ModelRecord, ResultDirectory, load_result, write_contract_artifacts

__all__ = [name for name in globals() if not name.startswith('__')]
