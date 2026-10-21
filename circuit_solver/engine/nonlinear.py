"""
nonlinear.py — Newton-Raphson iterative solver for non-linear circuits.

Theory
------
For circuits containing non-linear elements (Diodes, Zener diodes, BJTs, MOSFETs),
the nodal equations are non-linear:

    f(x) = A_lin · x + i_nonlin(x) - z_lin = 0

At each iteration k, non-linear elements are linearized around the operating point
(Q-point) x^(k) using their first-order Taylor expansion (companion models / Norton equivalents):

    i(x) ≈ i(x^(k)) + J · (x - x^(k)) = J · x - (J · x^(k) - i(x^(k)))
         = J · x - I_eq

where J = ∂i/∂x is the dynamic Jacobian matrix (conductance and transconductance)
and I_eq = J · x^(k) - i(x^(k)) is the equivalent independent current source.

The iterative algorithm solves:

    (A_lin + J^(k)) · x^(k+1) = z_lin + I_eq^(k)

until the max-norm residual satisfies:

    ‖x^(k+1) - x^(k)‖_∞ < tol  (default 1e-6 V)

SPICE-style voltage limiting (`pnjlim`) is applied to PN junctions to prevent
numerical overflow in the exponential terms during early iterations.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.linalg import LinAlgError

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import Diode, ZenerDiode, BJT, MOSFET
from circuit_solver.engine.mna import build_mna

from circuit_solver.core.solution import Solution

__all__ = ["solve_nonlinear", "pnjlim", "fetlim"]


def pnjlim(v_new: float, v_old: float, vt: float, vcrit: float) -> float:
    """SPICE-style PN junction voltage limiting algorithm.

    Prevents exponential numeric overflow by damping positive voltage steps
    that exceed the critical threshold voltage.
    """
    if v_new > vcrit and abs(v_new - v_old) > 2.0 * vt:
        if v_old > 0.0:
            arg = (v_new - v_old) / vt
            if arg > 0.0:
                return v_old + vt * (2.0 + math.log(arg))
            return vcrit
        return vcrit
    return v_new


def fetlim(v_new: float, v_old: float) -> float:
    """SPICE-style FET channel voltage limiting function.

    Prevents large voltage overshoots and oscillations across channel when entering triode.
    """
    if v_old > 0.0 and v_new < 0.0:
        return 0.5 * v_old
    if v_old < 0.0 and v_new > 0.0:
        return 0.5 * v_old
    delta = v_new - v_old
    if abs(delta) > 2.0:
        return v_old + 2.0 * math.copysign(1.0, delta)
    return v_new


def _build_linear_mna(circuit: Circuit) -> tuple[np.ndarray, np.ndarray]:
    """Build the static linear portion of the MNA matrix and RHS vector."""
    # Share DC stamps with the linear solvers, including ideal winding currents.
    return build_mna(circuit)


def solve_nonlinear(
    circuit: Circuit,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    damping: bool = True,
) -> Solution:
    """Solve circuit containing non-linear elements via Newton-Raphson iteration.

    Parameters
    ----------
    circuit : Circuit
        Pre-validated circuit.
    max_iter : int, optional
        Maximum number of Newton-Raphson iterations (default 100).
    tol : float, optional
        Residue convergence tolerance in Volts (default 1e-6 V).
    damping : bool, optional
        Whether to apply SPICE-style exponential junction voltage limiting (default True).

    Returns
    -------
    Solution
        Solution object wrapping the converged node voltages and branch currents.

    Raises
    ------
    RuntimeError
        If Newton-Raphson iteration fails to converge within ``max_iter``.
    """
    if circuit.dc_size == 0:
        raise ValueError("Circuit has no unknown variables to solve.")

    def idx(node: str) -> int | None:
        return circuit.node_index(node)

    A_lin, z_lin = _build_linear_mna(circuit)
    x = np.zeros(circuit.dc_size, dtype=float)

    # History of junction voltages for damping
    v_prev: dict[str, tuple[float, ...]] = {}

    for it in range(max_iter):
        A = A_lin.copy()
        z = z_lin.copy()

        def v_at(node: str) -> float:
            i = idx(node)
            return float(x[i]) if i is not None else 0.0

        for el in circuit.elements:
            # --------------------------------------------------------------
            # Diode or Zener Diode
            # --------------------------------------------------------------
            if isinstance(el, (Diode, ZenerDiode)):
                p, n_node = idx(el.node_p), idx(el.node_n)
                vp = v_at(el.node_p)
                vn = v_at(el.node_n)
                vd = vp - vn

                if damping:
                    vd_old = v_prev.get(el.name, (0.0,))[0]
                    vcrit = el.n * el.Vt * math.log((el.n * el.Vt) / (math.sqrt(2.0) * el.Is))
                    vd = pnjlim(vd, vd_old, el.n * el.Vt, vcrit)
                    v_prev[el.name] = (vd,)

                id_val, gd = el.evaluate(vd)
                Ieq = gd * vd - id_val

                if p is not None:
                    A[p, p] += gd
                    z[p] += Ieq
                if n_node is not None:
                    A[n_node, n_node] += gd
                    z[n_node] -= Ieq
                if p is not None and n_node is not None:
                    A[p, n_node] -= gd
                    A[n_node, p] -= gd

            # --------------------------------------------------------------
            # Bipolar Junction Transistor (BJT)
            # --------------------------------------------------------------
            elif isinstance(el, BJT):
                node_map_bjt = {
                    "C": (el.node_c, idx(el.node_c)),
                    "B": (el.node_b, idx(el.node_b)),
                    "E": (el.node_e, idx(el.node_e)),
                }
                vc = v_at(el.node_c)
                vb = v_at(el.node_b)
                ve = v_at(el.node_e)

                if damping:
                    sign = 1.0 if el.polarity == "NPN" else -1.0
                    vbe = sign * (vb - ve)
                    vbc = sign * (vb - vc)
                    vbe_old, vbc_old = v_prev.get(el.name, (0.0, 0.0))
                    vcrit = el.Vt * math.log(el.Vt / (math.sqrt(2.0) * el.Is))
                    vbe = pnjlim(vbe, vbe_old, el.Vt, vcrit)
                    vbc = pnjlim(vbc, vbc_old, el.Vt, vcrit)
                    v_prev[el.name] = (vbe, vbc)

                    if el.polarity == "NPN":
                        vb = ve + vbe
                        vc = vb - vbc
                    else:
                        vb = ve - vbe
                        vc = vb + vbc

                currents, jacobian = el.evaluate(vc, vb, ve)
                v_terms = {"C": vc, "B": vb, "E": ve}

                for t_out in ("C", "B", "E"):
                    _, row_idx = node_map_bjt[t_out]
                    if row_idx is None:
                        continue
                    i_out = currents[t_out]
                    ieq = -i_out
                    for t_in in ("C", "B", "E"):
                        dI_dv = jacobian[(t_out, t_in)]
                        ieq += dI_dv * v_terms[t_in]
                        _, col_idx = node_map_bjt[t_in]
                        if col_idx is not None:
                            A[row_idx, col_idx] += dI_dv
                    z[row_idx] += ieq

            # --------------------------------------------------------------
            # MOSFET
            # --------------------------------------------------------------
            elif isinstance(el, MOSFET):
                node_map_mos = {
                    "D": (el.node_d, idx(el.node_d)),
                    "G": (el.node_g, idx(el.node_g)),
                    "S": (el.node_s, idx(el.node_s)),
                }
                vd = v_at(el.node_d)
                vg = v_at(el.node_g)
                vs = v_at(el.node_s)

                currents, jacobian = el.evaluate(vd, vg, vs)
                v_terms = {"D": vd, "G": vg, "S": vs}

                for t_out in ("D", "G", "S"):
                    _, row_idx = node_map_mos[t_out]
                    if row_idx is None:
                        continue
                    i_out = currents[t_out]
                    ieq = -i_out
                    for t_in in ("D", "G", "S"):
                        dI_dv = jacobian[(t_out, t_in)]
                        ieq += dI_dv * v_terms[t_in]
                        _, col_idx = node_map_mos[t_in]
                        if col_idx is not None:
                            A[row_idx, col_idx] += dI_dv
                    z[row_idx] += ieq

        try:
            x_new = np.linalg.solve(A, z)
        except LinAlgError as exc:
            raise LinAlgError(
                f"Newton-Raphson iteration {it+1} failed to solve linear step: {exc}."
            ) from exc

        if damping:
            for el in circuit.elements:
                if isinstance(el, MOSFET):
                    d_idx = idx(el.node_d)
                    if d_idx is not None:
                        x_new[d_idx] = fetlim(float(x_new[d_idx]), float(x[d_idx]))

        diff = float(np.max(np.abs(x_new - x)))
        x = x_new

        if diff < tol:
            return Solution(circuit, x, iterations=it + 1)

    raise RuntimeError(
        f"Newton-Raphson solver failed to converge after {max_iter} iterations "
        f"(final residual {diff:.2e} > tolerance {tol:.2e})."
    )
