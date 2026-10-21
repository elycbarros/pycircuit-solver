"""
mna.py — Modified Nodal Analysis (MNA) stamp engine.

Theory
------
The MNA system is built as the block matrix equation:

    ⎡ G  B ⎤ ⎡ v ⎤   ⎡ i ⎤
    ⎢      ⎥ ⎢   ⎥ = ⎢   ⎥
    ⎣ C  D ⎦ ⎣ j ⎦   ⎣ e ⎦

where
    G  — conductance sub-matrix (n_nodes × n_nodes)
    B  — voltage-source incidence (n_nodes × n_aux)
    C  — constraint rows (n_aux × n_nodes)
    D  — usually zeros for independent sources / VCVS (n_aux × n_aux)
    v  — unknown node voltages
    j  — unknown auxiliary currents (through V-sources, VCVS, Op-Amp outputs)
    i  — current injection vector (from current sources into each node)
    e  — voltage constraint RHS (source voltages / VCVS / op-amp virtual short)

Stamp rules (Alexander & Sadiku §10.6, Vlach & Singhal §3):
    Resistor  R between p, n:  G[p,p] += 1/R; G[n,n] += 1/R;
                               G[p,n] -= 1/R; G[n,p] -= 1/R
    VSource   V between p, n:  B[p,k] += 1; B[n,k] -= 1
                               C[k,p] += 1; C[k,n] -= 1; e[k] = V
    ISource   I from n to p:   i[p] += I; i[n] -= I
    VCVS      E, gain μ:       (same B/C as V-source) + C[k,cp] -= μ; C[k,cn] += μ
    VCCS      G, gm:           G[p,cp] += gm; G[p,cn] -= gm;
                               G[n,cp] -= gm; G[n,cn] += gm
    OpAmp ideal:               B[out,k] += 1; B[gnd,k] -= 1   (output branch)
                               C[k,in_p] += 1; C[k,in_n] -= 1  (virtual short)
                               (virtual short → V(in_p) = V(in_n))

All indices are shifted by n_nodes for auxiliary rows/cols.
Ground ('0') rows/cols are skipped (not added to matrix).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.linalg import LinAlgError

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    CCCS,
    CCVS,
    OpAmp,
    Inductor,
    MutualCoupling,
    Transformer,
    _Element,
)
from circuit_solver.core.solution import Solution

__all__ = ["build_mna", "solve"]


def build_mna(circuit: Circuit) -> tuple[np.ndarray, np.ndarray]:
    """Stamp all elements into the MNA matrix A and RHS vector z.

    Parameters
    ----------
    circuit : Circuit
        Pre-validated circuit with node and auxiliary mappings.

    Returns
    -------
    A : np.ndarray, shape (size, size)
        The MNA coefficient matrix.
    z : np.ndarray, shape (size,)
        The right-hand side (current injections + voltage constraints).
    """
    n = circuit.size
    A = np.zeros((n, n), dtype=float)
    z = np.zeros(n, dtype=float)

    def idx(node: str) -> int | None:
        return circuit.node_index(node)

    def aux(name: str) -> int:
        return circuit.aux_index(name)

    for el in circuit.elements:

        # ------------------------------------------------------------------
        # Resistor: conductance stamp
        # ------------------------------------------------------------------
        if isinstance(el, Resistor):
            g = 1.0 / el.resistance
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                A[p, p] += g
            if n_node is not None:
                A[n_node, n_node] += g
            if p is not None and n_node is not None:
                A[p, n_node] -= g
                A[n_node, p] -= g

        # ------------------------------------------------------------------
        # Independent Voltage Source
        # ------------------------------------------------------------------
        elif isinstance(el, VoltageSource):
            k = aux(el.name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            # B columns
            if p is not None:
                A[p, k] += 1.0
            if n_node is not None:
                A[n_node, k] -= 1.0
            # C rows
            if p is not None:
                A[k, p] += 1.0
            if n_node is not None:
                A[k, n_node] -= 1.0
            # RHS
            z[k] = el.voltage

        # ------------------------------------------------------------------
        # Independent Current Source
        # ------------------------------------------------------------------
        elif isinstance(el, CurrentSource):
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                z[p] += el.current
            if n_node is not None:
                z[n_node] -= el.current

        # ------------------------------------------------------------------
        # VCVS (E element)
        # ------------------------------------------------------------------
        elif isinstance(el, VCVS):
            k = aux(el.name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            cp, cn = idx(el.ctrl_p), idx(el.ctrl_n)
            # Output branch (same as V-source)
            if p is not None:
                A[p, k] += 1.0
            if n_node is not None:
                A[n_node, k] -= 1.0
            # Constraint row: V(p) - V(n) - mu*(V(cp) - V(cn)) = 0
            if p is not None:
                A[k, p] += 1.0
            if n_node is not None:
                A[k, n_node] -= 1.0
            if cp is not None:
                A[k, cp] -= el.gain
            if cn is not None:
                A[k, cn] += el.gain
            z[k] = 0.0

        # ------------------------------------------------------------------
        # VCCS (G element)
        # ------------------------------------------------------------------
        elif isinstance(el, VCCS):
            gm = el.transconductance
            p, n_node = idx(el.node_p), idx(el.node_n)
            cp, cn = idx(el.ctrl_p), idx(el.ctrl_n)
            # Controlled current into node_p from ctrl voltage
            if p is not None and cp is not None:
                A[p, cp] -= gm
            if p is not None and cn is not None:
                A[p, cn] += gm
            if n_node is not None and cp is not None:
                A[n_node, cp] += gm
            if n_node is not None and cn is not None:
                A[n_node, cn] -= gm

        # ------------------------------------------------------------------
        # CCCS (F element)
        # ------------------------------------------------------------------
        elif isinstance(el, CCCS):
            k_vctrl = aux(el.vctrl_name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            # Generator convention for controlling source:
            # I(vctrl) exits node_p of vctrl, so I(vctrl) = -j_vctrl.
            # I_out = gain * I(vctrl) = -gain * j_vctrl injected into node_p.
            # KCL row p (currents leaving): - I_out = + gain * j_vctrl => A[p, k_vctrl] += gain.
            if p is not None:
                A[p, k_vctrl] += el.gain
            if n_node is not None:
                A[n_node, k_vctrl] -= el.gain

        # ------------------------------------------------------------------
        # CCVS (H element)
        # ------------------------------------------------------------------
        elif isinstance(el, CCVS):
            k = aux(el.name)
            k_vctrl = aux(el.vctrl_name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            # Output branch incidence
            if p is not None:
                A[p, k] += 1.0
            if n_node is not None:
                A[n_node, k] -= 1.0
            # Constraint row: V(p) - V(n) - rm * I(vctrl) = 0
            # Since I(vctrl) = -j_vctrl: V(p) - V(n) + rm * j_vctrl = 0.
            if p is not None:
                A[k, p] += 1.0
            if n_node is not None:
                A[k, n_node] -= 1.0
            A[k, k_vctrl] += el.transresistance
            z[k] = 0.0

        # ------------------------------------------------------------------
        # Ideal Op-Amp
        # ------------------------------------------------------------------
        elif isinstance(el, OpAmp):
            k = aux(el.name)
            out = idx(el.out)
            gnd = idx(el.gnd)
            in_p = idx(el.in_p)
            in_n = idx(el.in_n)
            # Output branch: current j flows from out to gnd (or into circuit)
            if out is not None:
                A[out, k] += 1.0
            if gnd is not None:
                A[gnd, k] -= 1.0
            # Constraint: V(in_p) - V(in_n) = 0  (virtual short)
            if in_p is not None:
                A[k, in_p] += 1.0
            if in_n is not None:
                A[k, in_n] -= 1.0
            z[k] = 0.0

        # ------------------------------------------------------------------
        # Inductor in DC Steady-State (Ideal short circuit)
        # ------------------------------------------------------------------
        elif isinstance(el, Inductor):
            # Model as high conductance (1 uOhm equivalent parasitic wire)
            g = 1e6
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None:
                A[p, p] += g
            if n_node is not None:
                A[n_node, n_node] += g
            if p is not None and n_node is not None:
                A[p, n_node] -= g
                A[n_node, p] -= g

        # ------------------------------------------------------------------
        # Transformer in DC Steady-State (Windings are ideal shorts)
        # ------------------------------------------------------------------
        elif isinstance(el, Transformer):
            g = 1e6
            p1, n1 = idx(el.p1), idx(el.n1)
            p2, n2 = idx(el.p2), idx(el.n2)
            if p1 is not None: A[p1, p1] += g
            if n1 is not None: A[n1, n1] += g
            if p1 is not None and n1 is not None:
                A[p1, n1] -= g
                A[n1, p1] -= g
            if p2 is not None: A[p2, p2] += g
            if n2 is not None: A[n2, n2] += g
            if p2 is not None and n2 is not None:
                A[p2, n2] -= g
                A[n2, p2] -= g

    return A, z


def solve(
    circuit: Circuit,
    *,
    rtol: float = 1e-9,
    tol: float = 1e-6,
    check_singular: bool = True,
    max_iter: int = 100,
) -> Solution:
    """Solve the MNA system and return a :class:`~circuit_solver.core.solution.Solution`.

    If the circuit contains non-linear semiconductor elements (Diodes, Zener diodes,
    BJTs, MOSFETs), the solver automatically delegates to the Newton-Raphson iterative solver.

    Parameters
    ----------
    circuit : Circuit
        Fully described circuit (pre-validated).
    rtol : float
        Relative tolerance for singular matrix detection (condition number check).
    tol : float
        Tolerance for non-linear Newton-Raphson convergence (default 1e-6 V).
    check_singular : bool
        Raise a descriptive error if the matrix is numerically singular.
    max_iter : int
        Maximum iterations for non-linear Newton-Raphson solver (default 100).

    Returns
    -------
    Solution
        Wrapper around the raw solution vector with labelled accessors.

    Raises
    ------
    ValueError
        If the circuit has no unknown nodes (trivial / disconnected).
    numpy.linalg.LinAlgError
        If the MNA matrix is singular (open circuit, missing ground, etc.).
    """
    if circuit.size == 0:
        raise ValueError("Circuit has no unknown variables to solve.")

    if circuit.has_nonlinear:
        from circuit_solver.engine.nonlinear import solve_nonlinear
        return solve_nonlinear(circuit, max_iter=max_iter, tol=tol)

    A, z = build_mna(circuit)

    if check_singular:
        cond = np.linalg.cond(A)
        if cond > 1e12:
            raise LinAlgError(
                f"MNA matrix is numerically singular (condition number ≈ {cond:.2e}). "
                "Check for: missing ground reference, floating nodes, or "
                "short-circuited ideal voltage source loops."
            )

    try:
        x = np.linalg.solve(A, z)
    except LinAlgError as exc:
        raise LinAlgError(
            f"Could not solve MNA system: {exc}. "
            "The circuit may be ill-conditioned or improperly specified."
        ) from exc

    return Solution(circuit, x)
