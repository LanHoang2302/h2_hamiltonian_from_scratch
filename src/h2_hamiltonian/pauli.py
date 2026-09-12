"""Decompose a small qubit Hamiltonian into Pauli strings."""
from __future__ import annotations

import itertools
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}


def kron_little_endian(labels):
    """Qubit 0 is the least-significant Fock bit."""
    out = np.array([[1.0 + 0j]])
    for label in reversed(labels):
        out = np.kron(out, PAULI[label])
    return out


def decompose_pauli(H: np.ndarray, tol=1e-10):
    n = int(round(np.log2(H.shape[0])))
    terms = []
    for labels in itertools.product("IXYZ", repeat=n):
        P = kron_little_endian(labels)
        coeff = np.trace(P.conj().T @ H) / (2**n)
        coeff = np.real_if_close(coeff).item()
        if abs(coeff) > tol:
            terms.append(("".join(labels), float(np.real(coeff))))
    terms.sort(key=lambda t: (-abs(t[1]), t[0]))
    return terms
