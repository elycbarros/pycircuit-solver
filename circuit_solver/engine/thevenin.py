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

import copy
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor,
    VoltageSource,
    CurrentSource,
    VCVS,
    VCCS,
    OpAmp,
    _Element,
)
from circuit_solver.engine.mna import solve, build_mna

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
    # ------------------------------------------------------------------
    # Step 1: Open-circuit voltage V_th
    # ------------------------------------------------------------------
    sol_oc = solve(circuit)
    V_th = sol_oc.V(node_a) - sol_oc.V(node_b)

    # ------------------------------------------------------------------
    # Step 2: Thévenin resistance via test source
    # ------------------------------------------------------------------
    killed = _kill_independent_sources(circuit.elements)

    # Add a 1 V test source between A and B
    test_name = "_Vtest_"
    # Ensure test name doesn't collide
    existing_names = {el.name for el in killed}
    while test_name in existing_names:
        test_name += "_"

    test_source = VoltageSource(test_name, node_a, node_b, 1.0)
    killed.append(test_source)

    # Build and solve the killed circuit (skip validation for internal use)
    circ_test = Circuit(killed, validate=False)
    sol_test = solve(circ_test)
    # The auxiliary variable j is current into '+'. Current delivered into terminal A is -j.
    I_test = -sol_test.I_vsource(test_name)

    # R_th = V_test / I_test
    if abs(I_test) < 1e-15:
        R_th = float("inf")
    else:
        R_th = 1.0 / I_test  # V_test = 1 V

    # ------------------------------------------------------------------
    # Step 3: Norton current
    # ------------------------------------------------------------------
    I_N = V_th / R_th if R_th != float("inf") else 0.0

    # ------------------------------------------------------------------
    # Step 4: Maximum power transfer sweep
    # ------------------------------------------------------------------
    P_max = (V_th ** 2) / (4.0 * R_th) if R_th > 0 else float("nan")

    if R_th > 0 and R_th != float("inf"):
        R_L_sweep = np.logspace(
            np.log10(max(r_min_factor * R_th, 1e-6)),
            np.log10(r_max_factor * R_th),
            n_sweep,
        )
    else:
        R_L_sweep = np.logspace(-2, 4, n_sweep)

    # P_L = (V_th * R_L / (R_th + R_L))^2 / R_L = V_th^2 * R_L / (R_th + R_L)^2
    if R_th != float("inf"):
        P_L_sweep = (V_th ** 2) * R_L_sweep / (R_th + R_L_sweep) ** 2
    else:
        P_L_sweep = np.zeros_like(R_L_sweep)

    return TheveninResult(
        V_th=V_th,
        R_th=R_th,
        I_N=I_N,
        P_max=P_max,
        R_L_sweep=R_L_sweep,
        P_L_sweep=P_L_sweep,
    )
