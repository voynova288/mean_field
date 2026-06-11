"""Common shift-current API for system adapters.

System-specific code should provide Hamiltonian eigenpairs and derivatives at a
k point, then call the generic helpers in :mod:`analysis.shift_current.core`.
Derivative calculations are delegated to :mod:`analysis.response_derivative_gauge`.
"""

from .core import *
