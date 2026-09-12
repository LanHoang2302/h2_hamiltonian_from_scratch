"""End-to-end construction of the H2 qubit Hamiltonian."""
from __future__ import annotations

import numpy as np
from .gaussian import Contracted1s
from .integrals import build_ao_integrals
from .rhf import rhf, transform_one_body, transform_two_body
from .fermion import build_hamiltonian, exact_ground_energy
from .pauli import decompose_pauli


def build_h2(bond_length_bohr: float = 1.4):
    R = float(bond_length_bohr)
    A = np.array([0.0, 0.0, -R / 2.0])
    B = np.array([0.0, 0.0, +R / 2.0])
    basis = [Contracted1s(A), Contracted1s(B)]
    nuclei = [(1.0, A), (1.0, B)]

    S, T, V, eri_ao = build_ao_integrals(basis, nuclei)
    Hcore = T + V
    Enuc = 1.0 / R

    hf = rhf(Hcore, S, eri_ao, n_electrons=2)
    C = hf["C"]
    h_mo = transform_one_body(Hcore, C)
    eri_mo = transform_two_body(eri_ao, C)

    H_qubit = build_hamiltonian(h_mo, eri_mo, Enuc)
    e_exact, sector_indices, vec = exact_ground_energy(H_qubit, n_electrons=2)
    pauli_terms = decompose_pauli(H_qubit)

    return {
        "bond_length_bohr": R,
        "S": S,
        "T": T,
        "V": V,
        "Hcore": Hcore,
        "eri_ao": eri_ao,
        "hf": hf,
        "hf_total_energy": hf["electronic_energy"] + Enuc,
        "h_mo": h_mo,
        "eri_mo": eri_mo,
        "nuclear_repulsion": Enuc,
        "H_qubit": H_qubit,
        "exact_ground_energy": e_exact,
        "sector_indices": sector_indices,
        "ground_vector_sector": vec,
        "pauli_terms": pauli_terms,
    }
