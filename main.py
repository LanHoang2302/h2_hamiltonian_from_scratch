from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from h2_hamiltonian import build_h2


def main():
    result = build_h2(1.4)

    print("=== H2 Hamiltonian from scratch ===")
    print(f"Bond length       : {result['bond_length_bohr']:.6f} bohr")
    print(f"RHF total energy  : {result['hf_total_energy']:.12f} Ha")
    print(f"Exact ground (N=2): {result['exact_ground_energy']:.12f} Ha")

    print("\nOverlap matrix S:")
    print(result["S"])

    print("\nCore Hamiltonian H_core = T + V:")
    print(result["Hcore"])

    print("\nLargest Pauli terms of the 4-qubit Hamiltonian:")
    for label, coeff in result["pauli_terms"][:20]:
        print(f"{coeff:+.10f}  {label}")


if __name__ == "__main__":
    main()
