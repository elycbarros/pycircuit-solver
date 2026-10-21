"""
thevenin.py — Thévenin / Norton equivalent circuit extraction.

Algorithm
---------
Given a circuit and two terminal nodes A and B, this module computes:

1. **V_th** (Thévenin voltage / open-circuit voltage):
   Solve the original circuit with the A–B port *open* (no element between A
   and B).  ``V_th = V(A) − V(B)`` from this solution.

2. **R_th** (Thévenin resistance):
   * Kill all **independent** sources (voltage sources → short-circuit,
     current sources → open-circuit).
   * Apply a test voltage source ``V_test = 1 V`` between A and B.
   * ``R_th = V_test / I_test`` where ``I_test`` is the current through the
     test source (the negative of the auxiliary variable).

3. **I_N** (Norton current): ``I_N = V_th / R_th``

4. **Maximum power transfer**: Sweep ``R_L`` from near-zero to many times
   ``R_th`` and find ``P_max = V_th² / (4 R_th)`` at ``R_L = R_th``.

Dependent sources are preserved during the R_th test (only independent
sources are killed).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    VoltageSource,
    CurrentSource,
    _Element,
)
from circuit_solver.engine.mna import solve

__all__ = ["TheveninResult", "thevenin_norton"]


@dataclass
class TheveninResult:
    """Container for Thévenin / Norton equivalent parameters.

    Attributes
    ----------
    V_th : float
        Thévenin (open-circuit) voltage in Volts.
    R_th : float
        Thévenin (equivalent) resistance in Ohms.
    I_N : float
        Norton short-circuit current in Amperes.
    P_max : float
        Maximum power deliverable to a matched load in Watts.
    R_L_sweep : np.ndarray
        Array of load resistance values used for the power sweep.
    P_L_sweep : np.ndarray
        Corresponding load power values.
    """

    V_th: float
    R_th: float
    I_N: float
    P_max: float
    R_L_sweep: np.ndarray
    P_L_sweep: np.ndarray

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  Thévenin / Norton Equivalent",
            "=" * 50,
            f"  V_th  = {self.V_th:>12.6f} V",
            f"  R_th  = {self.R_th:>12.6f} Ω",
            f"  I_N   = {self.I_N:>12.6f} A",
            f"  P_max = {self.P_max:>12.6f} W  (at R_L = R_th)",
            "=" * 50,
        ]
        return "\n".join(lines)


def _kill_independent_sources(elements: list[_Element]) -> list[_Element]:
    """Return a new element list with independent sources killed.

    * :class:`VoltageSource` → short-circuit (0 V source kept so the
      auxiliary variable still exists for any dependent source referencing it).
    * :class:`CurrentSource` → removed (open-circuit).
    """
    killed: list[_Element] = []
    for el in elements:
        if isinstance(el, VoltageSource):
            killed.append(replace(el, voltage=0.0))
        elif isinstance(el, CurrentSource):
            # Open-circuit: simply drop the current source
            pass
        else:
            killed.append(el)
    return killed


def thevenin_norton(
    circuit: Circuit,
    node_a: str,
    node_b: str,
    *,
    n_sweep: int = 200,
    r_min_factor: float = 1e-3,
    r_max_factor: float = 100.0,
) -> TheveninResult:
    """Compute the Thévenin / Norton equivalent seen at terminals A–B.

    Parameters
    ----------
    circuit : Circuit
        The original circuit (already validated).
    node_a : str
        Positive terminal of the output port.
    node_b : str
        Negative terminal (reference) of the output port.
    n_sweep : int
        Number of load-resistance points for the power-transfer sweep.
    r_min_factor : float
        ``R_L_min = r_min_factor * R_th`` (avoids division by zero).
    r_max_factor : float
        ``R_L_max = r_max_factor * R_th``.

    Returns
    -------
    TheveninResult
        All Thévenin / Norton parameters.
    """
    if node_a == node_b:
        raise ValueError(f"Os terminais A e B não podem ser o mesmo nó: '{node_a}'.")

    # ------------------------------------------------------------------
    # Step 1: Open-circuit voltage V_th
    # ------------------------------------------------------------------
    sol_oc = solve(circuit)
    V_th = sol_oc.V(node_a) - sol_oc.V(node_b)

    # ------------------------------------------------------------------
    # Step 2: Thévenin resistance via hybrid dual test
    # ------------------------------------------------------------------
    killed = _kill_independent_sources(circuit.elements)
    existing_names = {el.name for el in killed}

    # Method A: Test Current Source I_test = 1 A from node_b to node_a
    # This directly measures R_th = V(A) - V(B) and gracefully handles ideal
    # voltage sources / zero-resistance branches between A and B (R_th = 0).
    test_i_name = "_Itest_"
    while test_i_name in existing_names:
        test_i_name += "_"

    elements_i = list(killed)
    elements_i.append(CurrentSource(test_i_name, node_a, node_b, 1.0))

    r_th: float | None = None
    try:
        circ_i = Circuit(elements_i, validate=False)
        sol_i = solve(circ_i, check_singular=True)
        v_diff = sol_i.V(node_a) - sol_i.V(node_b)
        r_calc = float(v_diff)
        if abs(r_calc) < 1e-11:
            r_th = 0.0
        elif r_calc >= 0:
            r_th = r_calc
    except Exception:
        # If port is open-circuit, current injection will be singular -> fallback to Method B
        r_th = None

    if r_th is None:
        # Method B: Test Voltage Source V_test = 1 V between A and B
        test_v_name = "_Vtest_"
        while test_v_name in existing_names:
            test_v_name += "_"

        elements_v = list(killed)
        elements_v.append(VoltageSource(test_v_name, node_a, node_b, 1.0))
        try:
            circ_v = Circuit(elements_v, validate=False)
            sol_v = solve(circ_v, check_singular=True)
            i_test = -sol_v.I_vsource(test_v_name)
            if abs(i_test) < 1e-14:
                r_th = float("inf")
            else:
                r_th = 1.0 / i_test
        except Exception:
            r_th = float("inf")

    R_th = r_th

    # ------------------------------------------------------------------
    # Step 3: Norton current
    # ------------------------------------------------------------------
    if R_th == 0.0:
        I_N = float("inf") if V_th != 0.0 else 0.0
    elif R_th == float("inf"):
        I_N = 0.0
    else:
        I_N = V_th / R_th

    # ------------------------------------------------------------------
    # Step 4: Maximum power transfer sweep
    # ------------------------------------------------------------------
    if R_th == 0.0:
        P_max = float("inf") if V_th != 0.0 else 0.0
        R_L_sweep = np.logspace(-2, 4, n_sweep)
        P_L_sweep = (V_th ** 2) / R_L_sweep
    elif R_th == float("inf"):
        P_max = 0.0
        R_L_sweep = np.logspace(-2, 4, n_sweep)
        P_L_sweep = np.zeros_like(R_L_sweep)
    else:
        P_max = (V_th ** 2) / (4.0 * R_th)
        R_L_sweep = np.logspace(
            np.log10(max(r_min_factor * R_th, 1e-6)),
            np.log10(r_max_factor * R_th),
            n_sweep,
        )
        P_L_sweep = (V_th ** 2) * R_L_sweep / (R_th + R_L_sweep) ** 2

    return TheveninResult(
        V_th=V_th,
        R_th=R_th,
        I_N=I_N,
        P_max=P_max,
        R_L_sweep=R_L_sweep,
        P_L_sweep=P_L_sweep,
    )
