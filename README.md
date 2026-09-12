# H₂ Hamiltonian From Scratch

A quantum-chemistry project that shows **where the Hamiltonian used by VQE actually comes from**.

The goal is not to hide the chemistry behind `Qiskit Nature` or `PySCF`. The code explicitly constructs the minimal-basis H₂ Hamiltonian from Gaussian integrals, solves restricted Hartree–Fock, converts the result to second quantization, builds the fermionic Fock-space Hamiltonian, and decomposes it into Pauli strings.

## Why this project?

Many VQE tutorials begin with a ready-made qubit Hamiltonian such as

```text
H = c0 I + c1 Z0 + c2 Z1 + c3 Z0 Z1 + ...
```

This project answers the missing question:

> **Where do the coefficients and Pauli operators come from?**

## Pipeline

```text
H2 geometry
   ↓
STO-3G contracted 1s basis
   ↓
AO integrals: S, T, V, (μν|λσ)
   ↓
Core Hamiltonian: H_core = T + V
   ↓
Restricted Hartree-Fock
   ↓
AO → MO integral transformation
   ↓
Second-quantized electronic Hamiltonian
   ↓
Fermionic Fock-space matrix
   ↓
4-qubit Pauli decomposition
   ↓
Exact N=2 ground-state benchmark
```

## Electronic Hamiltonian

In atomic units,

```text
H = Σ_i[-1/2 ∇²_i - Σ_A Z_A/r_iA]
    + Σ_{i<j} 1/r_ij
    + Σ_{A<B} Z_A Z_B/R_AB
```

After choosing a one-particle basis, the electronic part is written as

```text
H = Σ_pq h_pq a†_p a_q
    + 1/2 Σ_pqrs (pq|rs) a†_p a†_r a_s a_q
    + E_nuc
```

For H₂ in STO-3G there are two spatial molecular orbitals and therefore four spin orbitals → four qubits before symmetry reduction.

## Project structure

```text
src/h2_hamiltonian/
├── gaussian.py    # STO-3G contracted 1s basis
├── integrals.py   # overlap, kinetic, nuclear attraction, ERI
├── rhf.py         # Restricted Hartree-Fock + AO→MO transform
├── fermion.py     # creation/annihilation algebra and Fock-space H
├── pauli.py       # Pauli decomposition
└── pipeline.py    # end-to-end H2 construction

main.py            # runnable demonstration
tests/test_h2.py   # energy and Hermiticity checks
```

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

Run tests:

```bash
pytest -q
```

At a bond length of roughly `R = 1.4 bohr`, you should obtain values close to

```text
RHF energy   ≈ -1.1167 Ha
Exact energy ≈ -1.1373 Ha
```

Small differences can come from conventions and numerical details.

## What I implemented

- STO-3G contracted Gaussian basis for hydrogen 1s orbitals
- primitive and contracted overlap integrals
- kinetic-energy integrals
- electron–nuclear attraction integrals
- two-electron repulsion integrals (ERI)
- Boys function F0
- closed-shell Restricted Hartree-Fock
- AO → MO integral transformation
- fermionic creation/annihilation operations
- second-quantized Hamiltonian construction
- fixed-particle-number exact diagonalization
- four-qubit Pauli decomposition

## Learning focus

The main purpose is to understand the bridge

```text
molecular physics → quantum chemistry → second quantization → qubits
```

rather than treating the qubit Hamiltonian as a black box.

## Reference / inspiration

This repository is an independent educational implementation based on standard quantum-chemistry formulas. While studying the topic, I also used the following open-source project as a reference for the overall learning direction:

- MetaDarko / QuantumA-Core — H₂ quantum chemistry example
  https://github.com/ShinRalexis/QuantumA-Core

No claim is made that the referenced project was authored here. If you copy any source code directly from that MIT-licensed repository, retain its original copyright and MIT license notice as required by its license.

## Next steps

- implement Jordan–Wigner symbolically instead of matrix-first decomposition
- add a one-parameter H₂ VQE ansatz
- compare SPSA vs COBYLA
- plot the H₂ dissociation curve
- compare against PySCF/Qiskit Nature only as a verification backend

## Author

Nguyen Hoang Lan
