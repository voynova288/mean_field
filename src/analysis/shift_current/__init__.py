"""Compatibility path for the common optical-response shift-current API.

System-specific code should provide Hamiltonian eigenpairs and derivatives at a
k point, then call the generic helpers in :mod:`analysis.optical_response`.
Derivative calculations are implemented under :mod:`analysis.optical_response.gauge`.
"""

from analysis.optical_response.shift_current import *  # noqa: F401,F403
