"""Restricted Hartree-Fock for closed-shell H2 in a tiny basis."""
from __future__ import annotations

import numpy as np


def symmetric_orthogonalizer(S: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(S)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def build_fock(Hcore: np.ndarray, eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    n = Hcore.shape[0]
    G = np.zeros_like(Hcore)
    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                for sig in range(n):
                    G[mu, nu] += P[lam, sig] * (
                        eri[mu, nu, lam, sig] - 0.5 * eri[mu, lam, nu, sig]
                    )
    return Hcore + G


def rhf(Hcore, S, eri, n_electrons=2, max_iter=100, tol=1e-12):
    n_occ = n_electrons // 2
    X = symmetric_orthogonalizer(S)
    P = np.zeros_like(Hcore)
    e_old = None

    for iteration in range(1, max_iter + 1):
        F = build_fock(Hcore, eri, P)
        eps, Cprime = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cprime
        Cocc = C[:, :n_occ]
        P_new = 2.0 * Cocc @ Cocc.T
        e_elec = 0.5 * np.sum(P_new * (Hcore + F))

        if e_old is not None and abs(e_elec - e_old) < tol and np.linalg.norm(P_new - P) < tol:
            return {
                "C": C,
                "orbital_energies": eps,
                "density": P_new,
                "fock": F,
                "electronic_energy": e_elec,
                "iterations": iteration,
            }
        P = P_new
        e_old = e_elec

    raise RuntimeError("RHF did not converge")


def transform_one_body(h_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    return C.T @ h_ao @ C


def transform_two_body(eri_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.einsum(
        "mp,nq,lr,ks,mnlk->pqrs", C, C, C, C, eri_ao, optimize=True
    )
