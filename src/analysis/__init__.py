"""Reusable and local analysis helpers.

Common optical-response workflows should enter through ``analysis.optical``,
which dispatches transition-table shift/injection responses and full-k-point
linear, Kerr/Faraday, and typed finite-band harmonic tensors. Hamiltonians,
derivative provenance, model multiplicity, and paper conventions remain in
system adapters; typed Passos payloads also carry their BZ and SI-unit contract.
"""
