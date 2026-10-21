"""
two_port.py — Análise de Redes de Duas Portas / Quadripolos (Two-Port Networks).

Fundamentado nas equações canônicas de teoria de circuitos:
1. Parâmetros de Impedância (Z):
     V1 = z11 * I1 + z12 * I2
     V2 = z21 * I1 + z22 * I2
2. Parâmetros de Admitância (Y):
     I1 = y11 * V1 + y12 * V2
     I2 = y21 * V1 + y22 * V2
3. Parâmetros Híbridos (H):
     V1 = h11 * I1 + h12 * V2
     I2 = h21 * I1 + h22 * V2
4. Parâmetros de Transmissão / ABCD (T):
     V1 = A * V2 - B * I2
     I1 = C * V2 - D * I2
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    CurrentSource,
    VoltageSource,
    Resistor,
    _Element,
)
from circuit_solver.engine.mna import solve
from circuit_solver.engine.ac import solve_ac

__all__ = ["TwoPortResult", "analyze_two_port"]


@dataclass
class TwoPortResult:
    """Encapsulates the mathematical parameter matrices of a Two-Port Network.

    Attributes
    ----------
    Z : np.ndarray
        2x2 Impedance matrix in Ohms.
    Y : np.ndarray
        2x2 Admittance matrix in Siemens.
    H : np.ndarray
        2x2 Hybrid matrix (h11 in Ω, h12 dimensionless, h21 dimensionless, h22 in S).
    T : np.ndarray
        2x2 Transmission (ABCD) matrix (A dimensionless, B in Ω, C in S, D dimensionless).
    frequency : float | None
        Analysis frequency in Hz (None for pure DC).
    is_reciprocal : bool
        True if z12 == z21 (or AD - BC == 1).
    is_symmetric : bool
        True if z11 == z22 and reciprocal (or A == D).
    """

    Z: np.ndarray
    Y: np.ndarray
    H: np.ndarray
    T: np.ndarray
    frequency: float | None = None
    is_reciprocal: bool = True
    is_symmetric: bool = False

    @property
    def z11(self) -> complex | float:
        return self.Z[0, 0]

    @property
    def z12(self) -> complex | float:
        return self.Z[0, 1]

    @property
    def z21(self) -> complex | float:
        return self.Z[1, 0]

    @property
    def z22(self) -> complex | float:
        return self.Z[1, 1]

    @property
    def y11(self) -> complex | float:
        return self.Y[0, 0]

    @property
    def y12(self) -> complex | float:
        return self.Y[0, 1]

    @property
    def y21(self) -> complex | float:
        return self.Y[1, 0]

    @property
    def y22(self) -> complex | float:
        return self.Y[1, 1]

    @property
    def h11(self) -> complex | float:
        return self.H[0, 0]

    @property
    def h12(self) -> complex | float:
        return self.H[0, 1]

    @property
    def h21(self) -> complex | float:
        return self.H[1, 0]

    @property
    def h22(self) -> complex | float:
        return self.H[1, 1]

    @property
    def A(self) -> complex | float:
        return self.T[0, 0]

    @property
    def B(self) -> complex | float:
        return self.T[0, 1]

    @property
    def C(self) -> complex | float:
        return self.T[1, 0]

    @property
    def D(self) -> complex | float:
        return self.T[1, 1]

    @property
    def ABCD(self) -> np.ndarray:
        return self.T

    def input_impedance(self, Z_load: complex | float) -> complex | float:
        """Calculate input driving-point impedance Zin seen at Port 1 terminated in Z_load at Port 2."""
        num = self.A * Z_load + self.B
        den = self.C * Z_load + self.D
        return num / den

    def output_impedance(self, Z_source: complex | float) -> complex | float:
        """Calculate output Thevenin impedance Zout seen at Port 2 with generator Z_source at Port 1."""
        num = self.D * Z_source + self.B
        den = self.C * Z_source + self.A
        return num / den

    def voltage_gain(self, Z_load: complex | float) -> complex | float:
        """Calculate closed-loop voltage transfer ratio Av = V2 / V1 terminated in Z_load."""
        return Z_load / (self.A * Z_load + self.B)

    def cascade(self, other: TwoPortResult) -> TwoPortResult:
        """Compute the equivalent TwoPortResult of cascading this network with another (T_total = T1 * T2)."""
        T_casc = self.T @ other.T
        A, B, C, D = T_casc[0, 0], T_casc[0, 1], T_casc[1, 0], T_casc[1, 1]
        # Reconstruir matrizes Z, Y, H a partir de T
        Z_casc = np.array([
            [A / C, (A * D - B * C) / C],
            [1.0 / C, D / C]
        ], dtype=T_casc.dtype)
        Y_casc = np.linalg.inv(Z_casc) if abs(np.linalg.det(Z_casc)) > 1e-14 else np.zeros((2, 2))
        H_casc = np.array([
            [B / D, (A * D - B * C) / D],
            [-1.0 / D, C / D]
        ], dtype=T_casc.dtype)
        is_recip = bool(abs((A * D - B * C) - 1.0) < 1e-4)
        is_sym = bool(is_recip and abs(A - D) < 1e-4)
        return TwoPortResult(
            Z=Z_casc, Y=Y_casc, H=H_casc, T=T_casc,
            frequency=self.frequency,
            is_reciprocal=is_recip,
            is_symmetric=is_sym
        )


def _kill_independent_sources(elements: Sequence[_Element]) -> list[_Element]:
    """Return a list of elements with all independent sources deactivated."""
    killed: list[_Element] = []
    for el in elements:
        if isinstance(el, VoltageSource):
            # Substituir por resistor de valor infinitesimal (quase curto) para manter continuidade nodal
            killed.append(Resistor(f"_short_{el.name}", el.node_p, el.node_n, 1e-12))
        elif isinstance(el, CurrentSource):
            # Circuito aberto (omitir elemento)
            continue
        else:
            killed.append(copy.deepcopy(el))
    return killed


def analyze_two_port(
    circuit: Circuit,
    port1: tuple[str, str],
    port2: tuple[str, str],
    frequency: float | None = None,
) -> TwoPortResult:
    """Extract Z, Y, H, and ABCD (Transmission) parameter matrices of a Two-Port Network.

    Parameters
    ----------
    circuit : Circuit
        Linear electrical network to analyze.
    port1 : tuple[str, str]
        (node_p, node_n) of Port 1 (Input).
    port2 : tuple[str, str]
        (node_p, node_n) of Port 2 (Output).
    frequency : float | None, optional
        Analysis frequency in Hz for AC phasor analysis. If None, DC analysis is used.

    Returns
    -------
    TwoPortResult
        Extracted parameter matrices with reciprocal/symmetric audit.
    """
    p1_p, p1_n = port1
    p2_p, p2_n = port2

    base_elements = _kill_independent_sources(circuit.elements)

    # Função auxiliar de simulação com injeção de corrente teste
    def run_ac_or_dc(test_elements, allowed_open):
        c = Circuit(test_elements, allowed_open_nodes=allowed_open)
        if frequency is None:
            return solve(c)
        else:
            return solve_ac(c, frequency=frequency)

    dtype = complex if frequency is not None else float

    # Teste 1: Injetar I1 = 1 A na Porta 1, com Porta 2 em circuito aberto (I2 = 0)
    open_nodes_test1 = [p1_p, p1_n, p2_p, p2_n]
    if frequency is None:
        src1 = CurrentSource("_Itest1", p1_p, p1_n, 1.0)
    else:
        src1 = CurrentSource("_Itest1", p1_p, p1_n, 0.0, ac_mag=1.0, ac_phase=0.0)

    sol1 = run_ac_or_dc(base_elements + [src1], open_nodes_test1)
    v1_1 = sol1.V(p1_p) - sol1.V(p1_n)
    v2_1 = sol1.V(p2_p) - sol1.V(p2_n)

    z11 = v1_1
    z21 = v2_1

    # Teste 2: Injetar I2 = 1 A na Porta 2, com Porta 1 em circuito aberto (I1 = 0)
    open_nodes_test2 = [p1_p, p1_n, p2_p, p2_n]
    if frequency is None:
        src2 = CurrentSource("_Itest2", p2_p, p2_n, 1.0)
    else:
        src2 = CurrentSource("_Itest2", p2_p, p2_n, 0.0, ac_mag=1.0, ac_phase=0.0)

    sol2 = run_ac_or_dc(base_elements + [src2], open_nodes_test2)
    v1_2 = sol2.V(p1_p) - sol2.V(p1_n)
    v2_2 = sol2.V(p2_p) - sol2.V(p2_n)

    z12 = v1_2
    z22 = v2_2

    Z_mat = np.array([[z11, z12], [z21, z22]], dtype=dtype)

    # Matriz Y (Admitância) por inversão direta de Z ou solução regularizada
    det_z = z11 * z22 - z12 * z21
    if abs(det_z) > 1e-14:
        Y_mat = np.linalg.inv(Z_mat)
    else:
        # Rede não possui representação Y finita (ou Z singular)
        Y_mat = np.full((2, 2), np.nan, dtype=dtype)

    # Parâmetros H (Híbridos)
    if abs(z22) > 1e-14:
        h11 = det_z / z22
        h12 = z12 / z22
        h21 = -z21 / z22
        h22 = 1.0 / z22
        H_mat = np.array([[h11, h12], [h21, h22]], dtype=dtype)
    else:
        H_mat = np.full((2, 2), np.nan, dtype=dtype)

    # Parâmetros de Transmissão ABCD (T)
    if abs(z21) > 1e-14:
        A = z11 / z21
        B = det_z / z21
        C = 1.0 / z21
        D = z22 / z21
        T_mat = np.array([[A, B], [C, D]], dtype=dtype)
    else:
        T_mat = np.full((2, 2), np.nan, dtype=dtype)

    # Verificação de propriedades analíticas
    is_reciprocal = bool(abs(z12 - z21) < 1e-4)
    is_symmetric = bool(is_reciprocal and abs(z11 - z22) < 1e-4)

    return TwoPortResult(
        Z=Z_mat,
        Y=Y_mat,
        H=H_mat,
        T=T_mat,
        frequency=frequency,
        is_reciprocal=is_reciprocal,
        is_symmetric=is_symmetric,
    )
