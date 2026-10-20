"""
transient.py — Time-domain transient analysis engine for linear circuits.

Implements companion models for dynamic elements (Capacitors and Inductors)
using implicit integration methods:
  * Trapezoidal Rule (2nd-order, A-stable, default)
  * Backward Euler (1st-order, L-stable, highly damped)

For a fixed time-step dt, the MNA system matrix A remains constant across
all time steps. It is LU-factorized once before the simulation loop, allowing
tens of thousands of time steps to be solved in milliseconds.
"""

from __future__ import annotations

from typing import Literal, Sequence
import numpy as np
from scipy import linalg

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
    Capacitor,
    Inductor,
    _Element,
)
from circuit_solver.core.transient_solution import TransientSolution
from circuit_solver.engine.mna import build_mna, solve as solve_dc

__all__ = ["solve_transient"]


def solve_transient(
    circuit: Circuit,
    *,
    t_stop: float,
    dt: float,
    method: Literal["trapezoidal", "euler"] = "trapezoidal",
    uic: bool = False,
) -> TransientSolution:
    """Run time-domain transient simulation on a linear circuit.

    Parameters
    ----------
    circuit : Circuit
        Pre-validated circuit topology.
    t_stop : float
        End of simulation time in seconds (must be > 0).
    dt : float
        Fixed time-step in seconds (must be > 0).
    method : {"trapezoidal", "euler"}, optional
        Numerical integration scheme (defaults to "trapezoidal").
    uic : bool, optional
        Use Initial Conditions: if True, start from the elements' ``ic``
        attributes; if False, calculate DC operating point at t=0.

    Returns
    -------
    TransientSolution
        Solution object containing time vector and waveform dictionaries.
    """
    if t_stop <= 0:
        raise ValueError(f"t_stop must be positive, got {t_stop}")
    if dt <= 0 or dt > t_stop:
        raise ValueError(f"dt must be positive and <= t_stop, got {dt}")

    time_vec = np.arange(0.0, t_stop + 0.5 * dt, dt)
    n_steps = len(time_vec)
    n = circuit.size

    def idx(node: str) -> int | None:
        return circuit.node_index(node)

    def aux(name: str) -> int:
        return circuit.aux_index(name)

    # -----------------------------------------------------------------------
    # 1. Classify dynamic elements and companion conductances
    # -----------------------------------------------------------------------
    capacitors: list[Capacitor] = [el for el in circuit.elements if isinstance(el, Capacitor)]
    inductors: list[Inductor] = [el for el in circuit.elements if isinstance(el, Inductor)]

    # Compute companion conductances
    # Capacitor:  G_eq = 2*C / dt (trap)  or  C / dt (euler)
    # Inductor:   G_eq = dt / (2*L) (trap)  or  dt / L (euler)
    g_c: dict[str, float] = {}
    for c in capacitors:
        g_c[c.name] = (2.0 * c.capacitance / dt) if method == "trapezoidal" else (c.capacitance / dt)

    g_l: dict[str, float] = {}
    for ind in inductors:
        g_l[ind.name] = (dt / (2.0 * ind.inductance)) if method == "trapezoidal" else (dt / ind.inductance)

    # -----------------------------------------------------------------------
    # 2. Build the time-invariant MNA matrix A
    # -----------------------------------------------------------------------
    A = np.zeros((n, n), dtype=float)

    for el in circuit.elements:
        # Resistor
        if isinstance(el, Resistor):
            g = 1.0 / el.resistance
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None: A[p, p] += g
            if n_node is not None: A[n_node, n_node] += g
            if p is not None and n_node is not None:
                A[p, n_node] -= g
                A[n_node, p] -= g

        # Voltage Source
        elif isinstance(el, VoltageSource):
            k = aux(el.name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None: A[p, k] += 1.0; A[k, p] += 1.0
            if n_node is not None: A[n_node, k] -= 1.0; A[k, n_node] -= 1.0

        # VCVS
        elif isinstance(el, VCVS):
            k = aux(el.name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            cp, cn = idx(el.ctrl_p), idx(el.ctrl_n)
            if p is not None: A[p, k] += 1.0; A[k, p] += 1.0
            if n_node is not None: A[n_node, k] -= 1.0; A[k, n_node] -= 1.0
            if cp is not None: A[k, cp] -= el.gain
            if cn is not None: A[k, cn] += el.gain

        # VCCS
        elif isinstance(el, VCCS):
            gm = el.transconductance
            p, n_node = idx(el.node_p), idx(el.node_n)
            cp, cn = idx(el.ctrl_p), idx(el.ctrl_n)
            if p is not None and cp is not None: A[p, cp] -= gm
            if p is not None and cn is not None: A[p, cn] += gm
            if n_node is not None and cp is not None: A[n_node, cp] += gm
            if n_node is not None and cn is not None: A[n_node, cn] -= gm

        # CCCS
        elif isinstance(el, CCCS):
            k_vctrl = aux(el.vctrl_name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None: A[p, k_vctrl] += el.gain
            if n_node is not None: A[n_node, k_vctrl] -= el.gain

        # CCVS
        elif isinstance(el, CCVS):
            k = aux(el.name)
            k_vctrl = aux(el.vctrl_name)
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None: A[p, k] += 1.0; A[k, p] += 1.0
            if n_node is not None: A[n_node, k] -= 1.0; A[k, n_node] -= 1.0
            A[k, k_vctrl] += el.transresistance

        # Op-Amp
        elif isinstance(el, OpAmp):
            k = aux(el.name)
            out = idx(el.out)
            gnd = idx(el.gnd)
            in_p = idx(el.in_p)
            in_n = idx(el.in_n)
            if out is not None: A[out, k] += 1.0
            if gnd is not None: A[gnd, k] -= 1.0
            if in_p is not None: A[k, in_p] += 1.0
            if in_n is not None: A[k, in_n] -= 1.0

        # Capacitor companion conductance
        elif isinstance(el, Capacitor):
            g = g_c[el.name]
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None: A[p, p] += g
            if n_node is not None: A[n_node, n_node] += g
            if p is not None and n_node is not None:
                A[p, n_node] -= g
                A[n_node, p] -= g

        # Inductor companion conductance
        elif isinstance(el, Inductor):
            g = g_l[el.name]
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None: A[p, p] += g
            if n_node is not None: A[n_node, n_node] += g
            if p is not None and n_node is not None:
                A[p, n_node] -= g
                A[n_node, p] -= g

    # Pre-factorize A with LU decomposition (once for the whole run)
    lu_factor = linalg.lu_factor(A)

    # -----------------------------------------------------------------------
    # 3. Initial state at t = 0
    # -----------------------------------------------------------------------
    # Track capacitor voltage and inductor current states
    v_c_state: dict[str, float] = {}
    i_c_state: dict[str, float] = {}
    v_l_state: dict[str, float] = {}
    i_l_state: dict[str, float] = {}

    # Storage arrays
    node_labels = list(circuit.nodes)
    node_voltages: dict[str, np.ndarray] = {node: np.zeros(n_steps) for node in node_labels}
    branch_currents: dict[str, np.ndarray] = {el.name: np.zeros(n_steps) for el in circuit.elements}

    if uic:
        # Use explicit initial conditions
        for c in capacitors:
            v_c_state[c.name] = c.ic
            i_c_state[c.name] = 0.0
        for ind in inductors:
            v_l_state[ind.name] = 0.0
            i_l_state[ind.name] = ind.ic

        # Solve t=0 state from companion model at t=0
        z0 = np.zeros(n)
        for el in circuit.elements:
            if isinstance(el, VoltageSource):
                val = el.value_at(0.0) if hasattr(el, "value_at") else el.voltage
                z0[aux(el.name)] = val
            elif isinstance(el, CurrentSource):
                val = el.value_at(0.0) if hasattr(el, "value_at") else el.current
                p, n_node = idx(el.node_p), idx(el.node_n)
                if p is not None: z0[p] += val
                if n_node is not None: z0[n_node] -= val

        # Invert companion currents at t=0
        for c in capacitors:
            ieq = g_c[c.name] * v_c_state[c.name]
            p, n_node = idx(c.node_p), idx(c.node_n)
            if p is not None: z0[p] += ieq
            if n_node is not None: z0[n_node] -= ieq

        for ind in inductors:
            ieq = i_l_state[ind.name]
            p, n_node = idx(ind.node_p), idx(ind.node_n)
            if p is not None: z0[p] -= ieq
            if n_node is not None: z0[n_node] += ieq

        x0 = linalg.lu_solve(lu_factor, z0)
    else:
        # Operating point DC solve at t = 0
        try:
            dc_sol = solve_dc(circuit)
            x0 = dc_sol._x
            for c in capacitors:
                vp = dc_sol.V(c.node_p) if c.node_p != "0" else 0.0
                vn = dc_sol.V(c.node_n) if c.node_n != "0" else 0.0
                v_c_state[c.name] = vp - vn
                i_c_state[c.name] = 0.0

            for ind in inductors:
                v_l_state[ind.name] = 0.0
                # Current through inductor short at DC
                vp = dc_sol.V(ind.node_p) if ind.node_p != "0" else 0.0
                vn = dc_sol.V(ind.node_n) if ind.node_n != "0" else 0.0
                i_l_state[ind.name] = (vp - vn) * 1e6
        except Exception:
            # Fallback to zero state if DC fails
            x0 = np.zeros(n)
            for c in capacitors: v_c_state[c.name] = 0.0; i_c_state[c.name] = 0.0
            for ind in inductors: v_l_state[ind.name] = 0.0; i_l_state[ind.name] = 0.0

    # Record t=0 results
    for node in node_labels:
        node_voltages[node][0] = x0[circuit.node_index(node)]

    for c in capacitors:
        branch_currents[c.name][0] = i_c_state[c.name]
    for ind in inductors:
        branch_currents[ind.name][0] = i_l_state[ind.name]
    for el in circuit.elements:
        if isinstance(el, Resistor):
            vp = x0[idx(el.node_p)] if idx(el.node_p) is not None else 0.0
            vn = x0[idx(el.node_n)] if idx(el.node_n) is not None else 0.0
            branch_currents[el.name][0] = (vp - vn) / el.resistance
        elif isinstance(el, VoltageSource):
            branch_currents[el.name][0] = x0[aux(el.name)]

    # -----------------------------------------------------------------------
    # 4. Main Transient Time Stepping Loop
    # -----------------------------------------------------------------------
    for k in range(1, n_steps):
        t_curr = time_vec[k]
        z = np.zeros(n, dtype=float)

        # Independent sources at current time t_curr
        for el in circuit.elements:
            if isinstance(el, VoltageSource):
                val = el.value_at(t_curr) if hasattr(el, "value_at") else el.voltage
                z[aux(el.name)] = val
            elif isinstance(el, CurrentSource):
                val = el.value_at(t_curr) if hasattr(el, "value_at") else el.current
                p, n_node = idx(el.node_p), idx(el.node_n)
                if p is not None: z[p] += val
                if n_node is not None: z[n_node] -= val

        # Capacitor companion current injection
        ieq_c: dict[str, float] = {}
        for c in capacitors:
            if method == "trapezoidal":
                ieq = g_c[c.name] * v_c_state[c.name] + i_c_state[c.name]
            else:
                ieq = g_c[c.name] * v_c_state[c.name]
            ieq_c[c.name] = ieq

            p, n_node = idx(c.node_p), idx(c.node_n)
            if p is not None: z[p] += ieq
            if n_node is not None: z[n_node] -= ieq

        # Inductor companion current injection
        ieq_l: dict[str, float] = {}
        for ind in inductors:
            if method == "trapezoidal":
                ieq = g_l[ind.name] * v_l_state[ind.name] + i_l_state[ind.name]
            else:
                ieq = i_l_state[ind.name]
            ieq_l[ind.name] = ieq

            p, n_node = idx(ind.node_p), idx(ind.node_n)
            if p is not None: z[p] -= ieq
            if n_node is not None: z[n_node] += ieq

        # Solve MNA system for step k
        x = linalg.lu_solve(lu_factor, z)

        # Update node voltages
        for node in node_labels:
            node_voltages[node][k] = x[circuit.node_index(node)]

        # Update capacitor states & currents
        for c in capacitors:
            vp = x[idx(c.node_p)] if idx(c.node_p) is not None else 0.0
            vn = x[idx(c.node_n)] if idx(c.node_n) is not None else 0.0
            v_curr = vp - vn
            i_curr = g_c[c.name] * v_curr - ieq_c[c.name]

            v_c_state[c.name] = v_curr
            i_c_state[c.name] = i_curr
            branch_currents[c.name][k] = i_curr

        # Update inductor states & currents
        for ind in inductors:
            vp = x[idx(ind.node_p)] if idx(ind.node_p) is not None else 0.0
            vn = x[idx(ind.node_n)] if idx(ind.node_n) is not None else 0.0
            v_curr = vp - vn
            i_curr = g_l[ind.name] * v_curr + ieq_l[ind.name]

            v_l_state[ind.name] = v_curr
            i_l_state[ind.name] = i_curr
            branch_currents[ind.name][k] = i_curr

        # Update resistors and sources currents
        for el in circuit.elements:
            if isinstance(el, Resistor):
                vp = x[idx(el.node_p)] if idx(el.node_p) is not None else 0.0
                vn = x[idx(el.node_n)] if idx(el.node_n) is not None else 0.0
                branch_currents[el.name][k] = (vp - vn) / el.resistance
            elif isinstance(el, VoltageSource):
                branch_currents[el.name][k] = x[aux(el.name)]

    return TransientSolution(
        circuit=circuit,
        time=time_vec,
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )
