from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h2_hamiltonian import build_h2


def test_h2_reference_energies():
    out = build_h2(1.4)
    # STO-3G H2 reference values at R=1.4 bohr, loose tolerance for educational implementation.
    assert abs(out["hf_total_energy"] - (-1.1167)) < 5e-3
    assert abs(out["exact_ground_energy"] - (-1.1373)) < 5e-3


def test_qubit_hamiltonian_is_hermitian():
    out = build_h2(1.4)
    H = out["H_qubit"]
    assert np.allclose(H, H.T.conj(), atol=1e-10)
