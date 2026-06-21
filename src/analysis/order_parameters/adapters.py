from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .classification import classify_tdbg_flavor_state
from .flavor import flavor_order_parameters
from .schema import OrderParameterResult, StateLabel


def analyze_order_parameters(
    projector_kab: np.ndarray,
    labels: Sequence[StateLabel],
    *,
    classifier: Callable[[dict[str, float]], str] | None = None,
) -> OrderParameterResult:
    """Analyze common flavor order parameters with an optional classifier."""

    result = flavor_order_parameters(projector_kab, labels)
    classification = classifier(dict(result.scalars)) if classifier is not None else None
    return OrderParameterResult(
        scalars=dict(result.scalars),
        arrays=dict(result.arrays),
        tables=dict(result.tables),
        classification=classification,
        metadata=dict(result.metadata),
    )


def analyze_tdbg_order_parameters(projector_kab: np.ndarray, labels: Sequence[StateLabel]) -> OrderParameterResult:
    """TDBG preset preserving historical active-band flavor classification."""

    return analyze_order_parameters(projector_kab, labels, classifier=classify_tdbg_flavor_state)


__all__ = ["analyze_order_parameters", "analyze_tdbg_order_parameters"]
