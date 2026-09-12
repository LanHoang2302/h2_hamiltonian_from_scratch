"""One- and two-electron Gaussian integrals for contracted 1s functions."""
from __future__ import annotations

import math
import numpy as np
from .gaussian import Contracted1s


def boys0(t: float) -> float:
    if t < 1e-10:
        return 1.0 - t / 3.0
    return 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t))


def _product_center(alpha: float, A: np.ndarray, beta: float, B: np.ndarray):
    p = alpha + beta
    return (alpha * A + beta * B) / p


def primitive_overlap(alpha, A, beta, B):
    p = alpha + beta
    mu = alpha * beta / p
    r2 = float(np.dot(A - B, A - B))
    return (math.pi / p) ** 1.5 * math.exp(-mu * r2)


def primitive_kinetic(alpha, A, beta, B):
    p = alpha + beta
    mu = alpha * beta / p
    r2 = float(np.dot(A - B, A - B))
    return mu * (3.0 - 2.0 * mu * r2) * primitive_overlap(alpha, A, beta, B)


def primitive_nuclear_attraction(alpha, A, beta, B, C, charge=1.0):
    p = alpha + beta
    mu = alpha * beta / p
    rAB2 = float(np.dot(A - B, A - B))
    P = _product_center(alpha, A, beta, B)
    rPC2 = float(np.dot(P - C, P - C))
    return -charge * 2.0 * math.pi / p * math.exp(-mu * rAB2) * boys0(p * rPC2)


def primitive_eri(alpha, A, beta, B, gamma, C, delta, D):
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    P = _product_center(alpha, A, beta, B)
    Q = _product_center(gamma, C, delta, D)
    rAB2 = float(np.dot(A - B, A - B))
    rCD2 = float(np.dot(C - D, C - D))
    rPQ2 = float(np.dot(P - Q, P - Q))
    pref = 2.0 * math.pi ** 2.5 / (p * q * math.sqrt(p + q))
    return pref * math.exp(-mu * rAB2 - nu * rCD2) * boys0(p * q / (p + q) * rPQ2)


def contracted_overlap(a: Contracted1s, b: Contracted1s):
    val = 0.0
    for aa, ca, Na in a.primitives():
        for bb, cb, Nb in b.primitives():
            val += ca * cb * Na * Nb * primitive_overlap(aa, a.center, bb, b.center)
    return val


def contracted_kinetic(a: Contracted1s, b: Contracted1s):
    val = 0.0
    for aa, ca, Na in a.primitives():
        for bb, cb, Nb in b.primitives():
            val += ca * cb * Na * Nb * primitive_kinetic(aa, a.center, bb, b.center)
    return val


def contracted_nuclear_attraction(a: Contracted1s, b: Contracted1s, nuclei):
    val = 0.0
    for aa, ca, Na in a.primitives():
        for bb, cb, Nb in b.primitives():
            for charge, center in nuclei:
                val += ca * cb * Na * Nb * primitive_nuclear_attraction(
                    aa, a.center, bb, b.center, np.asarray(center), charge
                )
    return val


def contracted_eri(a: Contracted1s, b: Contracted1s, c: Contracted1s, d: Contracted1s):
    val = 0.0
    for aa, ca, Na in a.primitives():
        for bb, cb, Nb in b.primitives():
            for cc, ccoef, Nc in c.primitives():
                for dd, dcoef, Nd in d.primitives():
                    val += (
                        ca * cb * ccoef * dcoef * Na * Nb * Nc * Nd
                        * primitive_eri(aa, a.center, bb, b.center, cc, c.center, dd, d.center)
                    )
    return val


def build_ao_integrals(basis, nuclei):
    n = len(basis)
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    eri = np.zeros((n, n, n, n))
    for mu in range(n):
        for nu in range(n):
            S[mu, nu] = contracted_overlap(basis[mu], basis[nu])
            T[mu, nu] = contracted_kinetic(basis[mu], basis[nu])
            V[mu, nu] = contracted_nuclear_attraction(basis[mu], basis[nu], nuclei)
            for lam in range(n):
                for sig in range(n):
                    eri[mu, nu, lam, sig] = contracted_eri(
                        basis[mu], basis[nu], basis[lam], basis[sig]
                    )
    return S, T, V, eri
