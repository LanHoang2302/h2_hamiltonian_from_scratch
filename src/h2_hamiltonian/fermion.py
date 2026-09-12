"""Build the second-quantized electronic Hamiltonian as a Fock-space matrix."""
from __future__ import annotations

import numpy as np


def _annihilate(state: int, orbital: int):
    if ((state >> orbital) & 1) == 0:
        return None
    parity = (state & ((1 << orbital) - 1)).bit_count()
    sign = -1.0 if parity % 2 else 1.0
    return state ^ (1 << orbital), sign


def _create(state: int, orbital: int):
    if ((state >> orbital) & 1) == 1:
        return None
    parity = (state & ((1 << orbital) - 1)).bit_count()
    sign = -1.0 if parity % 2 else 1.0
    return state ^ (1 << orbital), sign


def _apply_ops(state: int, ops):
    """Apply operators right-to-left. ops items are ('a'|'ad', orbital)."""
    amp = 1.0
    current = state
    for kind, orb in reversed(ops):
        out = _annihilate(current, orb) if kind == "a" else _create(current, orb)
        if out is None:
            return None
        current, sign = out
        amp *= sign
    return current, amp


def build_hamiltonian(h_mo: np.ndarray, eri_mo: np.ndarray, nuclear_repulsion: float):
    n_spatial = h_mo.shape[0]
    n_spin = 2 * n_spatial
    dim = 1 << n_spin
    H = np.eye(dim, dtype=float) * nuclear_repulsion

    # One-electron term: sum_pq,sigma h_pq a^†_{pσ} a_{qσ}
    for p in range(n_spatial):
        for q in range(n_spatial):
            for spin in range(2):
                P = 2 * p + spin
                Q = 2 * q + spin
                coeff = h_mo[p, q]
                for state in range(dim):
                    out = _apply_ops(state, [("ad", P), ("a", Q)])
                    if out is not None:
                        target, amp = out
                        H[target, state] += coeff * amp

    # Two-electron term:
    # 1/2 sum_pqrs,στ (pq|rs) a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    coeff = 0.5 * eri_mo[p, q, r, s]
                    if abs(coeff) < 1e-14:
                        continue
                    for sigma in range(2):
                        for tau in range(2):
                            P, Q = 2 * p + sigma, 2 * q + sigma
                            R, S = 2 * r + tau, 2 * s + tau
                            for state in range(dim):
                                out = _apply_ops(
                                    state,
                                    [("ad", P), ("ad", R), ("a", S), ("a", Q)],
                                )
                                if out is not None:
                                    target, amp = out
                                    H[target, state] += coeff * amp
    return H


def particle_number_indices(n_spin_orbitals: int, n_electrons: int):
    return [s for s in range(1 << n_spin_orbitals) if s.bit_count() == n_electrons]


def exact_ground_energy(H: np.ndarray, n_electrons: int):
    n_spin = int(round(np.log2(H.shape[0])))
    idx = particle_number_indices(n_spin, n_electrons)
    block = H[np.ix_(idx, idx)]
    vals, vecs = np.linalg.eigh(block)
    return float(vals[0]), idx, vecs[:, 0]
