from __future__ import annotations

"""Toy/reference models for optical-response checks."""

from .slg_toy import GappedSLGParams, d2hdk, dhdk, diagonalize, hamiltonian, nn_vectors

__all__ = ["GappedSLGParams", "d2hdk", "dhdk", "diagonalize", "hamiltonian", "nn_vectors"]
