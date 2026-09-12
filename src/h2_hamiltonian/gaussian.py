"""Minimal STO-3G utilities for 1s orbitals.

All formulas are implemented explicitly for educational use.
Atomic units are used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

STO3G_H_EXPONENTS = np.array([3.42525091, 0.62391373, 0.16885540], dtype=float)
STO3G_H_COEFFICIENTS = np.array([0.15432897, 0.53532814, 0.44463454], dtype=float)


def primitive_s_normalization(alpha: float) -> float:
    """Normalization constant of exp(-alpha r^2) for an s Gaussian."""
    return (2.0 * alpha / math.pi) ** 0.75


@dataclass(frozen=True)
class Contracted1s:
    center: np.ndarray
    exponents: np.ndarray = None
    coefficients: np.ndarray = None

    def __post_init__(self):
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))
        if self.exponents is None:
            object.__setattr__(self, "exponents", STO3G_H_EXPONENTS.copy())
        if self.coefficients is None:
            object.__setattr__(self, "coefficients", STO3G_H_COEFFICIENTS.copy())

    def primitives(self):
        for alpha, coeff in zip(self.exponents, self.coefficients):
            yield float(alpha), float(coeff), primitive_s_normalization(float(alpha))
