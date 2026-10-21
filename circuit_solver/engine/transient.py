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

from typing import Literal
from dataclasses import replace
import numpy as np
from scipy import linalg

import time
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
    MutualCoupling,
    Transformer,
)
from circuit_solver.core.transient_solution import TransientSolution
from circuit_solver.engine.mna import solve as solve_dc
from circuit_solver.engine.transient_rust import (
    solve_transient_rust,
    can_run_in_rust,
    RUST_AVAILABLE,
)

__all__ = ["solve_transient", "RUST_AVAILABLE"]


def _capacitors_forced_by_voltage_sources(circuit: Circuit, capacitors) -> set[str]:
    """Nomes dos capacitores cujos terminais já estão ligados por fontes de tensão ideais.

    Nesse caso a tensão do capacitor é fixada pela fonte e o ic não tem efeito sobre os nós.
    Substituir o capacitor por uma fonte de tensão de valor ic criaria fontes em paralelo
    e um sistema singular, por isso esses capacitores ficam fora da solução de t = 0.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for el in circuit.elements:
        if isinstance(el, VoltageSource):
            ra, rb = find(el.node_p), find(el.node_n)
            if ra != rb:
                parent[ra] = rb

    return {c.name for c in capacitors if find(c.node_p) == find(c.node_n)}


def solve_transient(
    circuit: Circuit,
    *,
    t_stop: float,
    dt: float,
    method: Literal["trapezoidal", "euler"] = "trapezoidal",
    uic: bool = False,
    backend: Literal["auto", "rust", "python"] = "auto",
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
    backend : {"auto", "rust", "python"}, optional
        Execution backend. "auto" uses compiled Rust if available and topology
        is supported, falling back seamlessly to Python.

    Returns
    -------
    TransientSolution
        Solution object containing time vector and waveform dictionaries.
    """
    if method not in ("trapezoidal", "euler") or backend not in ("auto", "rust", "python"):
        raise ValueError("Invalid transient method or backend.")
    for el in circuit.elements:
        if not isinstance(el, (Resistor, VoltageSource, CurrentSource, VCVS, VCCS,
                               CCCS, CCVS, OpAmp, Capacitor, Inductor, MutualCoupling, Transformer)):
            label = "Circuit cannot be simulated with Rust transient engine" if backend == "rust" else "Transient analysis supports linear elements only"
            raise ValueError(f"{label}: {el.name} ({type(el).__name__})")
    if not np.isfinite(t_stop) or not np.isfinite(dt):
        raise ValueError("Transient times must be finite.")
    if t_stop <= 0:
        raise ValueError(f"t_stop must be positive, got {t_stop}")
    if dt <= 0 or dt > t_stop:
        raise ValueError(f"dt must be positive and <= t_stop, got {dt}")

    if backend == "rust":
        return solve_transient_rust(circuit, t_stop=t_stop, dt=dt, method=method, uic=uic)

    if backend == "auto" and RUST_AVAILABLE:
        ok, _ = can_run_in_rust(circuit)
        if ok:
            return solve_transient_rust(circuit, t_stop=t_stop, dt=dt, method=method, uic=uic)

    t_py_start = time.perf_counter()

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
    coupled_pairs = circuit.get_coupled_pairs()
    coupled_ind_names = circuit.get_coupled_inductor_names()

    capacitors: list[Capacitor] = [el for el in circuit.elements if isinstance(el, Capacitor)]
    inductors: list[Inductor] = [el for el in circuit.elements if isinstance(el, Inductor) and el.name not in coupled_ind_names]

    # Compute companion conductances
    # Capacitor:  G_eq = 2*C / dt (trap)  or  C / dt (euler)
    # Inductor:   G_eq = dt / (2*L) (trap)  or  dt / L (euler)
    g_c: dict[str, float] = {}
    for c in capacitors:
        g_c[c.name] = (2.0 * c.capacitance / dt) if method == "trapezoidal" else (c.capacitance / dt)

    g_l: dict[str, float] = {}
    for ind in inductors:
        g_l[ind.name] = (dt / (2.0 * ind.inductance)) if method == "trapezoidal" else (dt / ind.inductance)

    # Coupled pairs companion conductance matrix G_eq = alpha * Gamma
    alpha = (dt / 2.0) if method == "trapezoidal" else dt
    g_eq_coupled: dict[str, np.ndarray] = {
        pair["name"]: alpha * pair["Gamma"] for pair in coupled_pairs
    }

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
        elif isinstance(el, Inductor) and el.name not in coupled_ind_names:
            g = g_l[el.name]
            p, n_node = idx(el.node_p), idx(el.node_n)
            if p is not None: A[p, p] += g
            if n_node is not None: A[n_node, n_node] += g
            if p is not None and n_node is not None:
                A[p, n_node] -= g
                A[n_node, p] -= g

    # Coupled pairs companion conductance matrix stamping
    for pair in coupled_pairs:
        G = g_eq_coupled[pair["name"]]
        p1, n1 = idx(pair["p1"]), idx(pair["n1"])
        p2, n2 = idx(pair["p2"]), idx(pair["n2"])

        # Port 1
        g11 = G[0, 0]
        if p1 is not None: A[p1, p1] += g11
        if n1 is not None: A[n1, n1] += g11
        if p1 is not None and n1 is not None: A[p1, n1] -= g11; A[n1, p1] -= g11

        # Port 2
        g22 = G[1, 1]
        if p2 is not None: A[p2, p2] += g22
        if n2 is not None: A[n2, n2] += g22
        if p2 is not None and n2 is not None: A[p2, n2] -= g22; A[n2, p2] -= g22

        # Cross
        g12 = G[0, 1]
        g21 = G[1, 0]
        if p1 is not None and p2 is not None: A[p1, p2] += g12; A[p2, p1] += g21
        if p1 is not None and n2 is not None: A[p1, n2] -= g12; A[n2, p1] -= g21
        if n1 is not None and p2 is not None: A[n1, p2] -= g12; A[p2, n1] -= g21
        if n1 is not None and n2 is not None: A[n1, n2] += g12; A[n2, n1] += g21

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
    v_coupled_state: dict[str, np.ndarray] = {}
    i_coupled_state: dict[str, np.ndarray] = {}

    # Storage arrays
    node_labels = list(circuit.nodes)
    node_voltages: dict[str, np.ndarray] = {node: np.zeros(n_steps) for node in node_labels}
    branch_currents: dict[str, np.ndarray] = {el.name: np.zeros(n_steps) for el in circuit.elements}
    for pair in coupled_pairs:
        if pair["ind1_name"] not in branch_currents:
            branch_currents[pair["ind1_name"]] = np.zeros(n_steps)
        if pair["ind2_name"] not in branch_currents:
            branch_currents[pair["ind2_name"]] = np.zeros(n_steps)

    if uic:
        # Estado em t = 0+ a partir das condições iniciais. Cada capacitor vira uma fonte de
        # tensão de valor ic e cada indutor vira uma fonte de corrente de valor ic. Resolver
        # esse circuito resistivo dá v(0+) nos nós, i_C(0+) pela corrente da fonte que
        # substitui o capacitor e v_L(0+) pela tensão nos terminais do indutor.
        # O modelo companheiro não serve para t = 0: ele exigiria i_C(0) e v_L(0) já conhecidos.
        forced_caps = _capacitors_forced_by_voltage_sources(circuit, capacitors)
        for c in capacitors:
            if c.name not in forced_caps:
                v_c_state[c.name] = c.ic
        for ind in inductors:
            i_l_state[ind.name] = ind.ic

        ind_by_name = {el.name: el for el in circuit.elements if isinstance(el, Inductor)}
        for pair in coupled_pairs:
            i_coupled_state[pair["name"]] = np.array(
                [
                    getattr(ind_by_name.get(pair["ind1_name"]), "ic", 0.0),
                    getattr(ind_by_name.get(pair["ind2_name"]), "ic", 0.0),
                ],
                dtype=float,
            )

        sub_elements = []
        for el in circuit.elements:
            if isinstance(el, Capacitor):
                if el.name not in forced_caps:
                    sub_elements.append(VoltageSource(el.name, el.node_p, el.node_n, el.ic))
            elif isinstance(el, Inductor):
                # corrente ic flui de node_p para node_n: injeta em node_n, drena de node_p
                sub_elements.append(CurrentSource(el.name, el.node_n, el.node_p, el.ic))
            elif isinstance(el, VoltageSource):
                sub_elements.append(VoltageSource(el.name, el.node_p, el.node_n, el.value_at(0.0)))
            elif isinstance(el, CurrentSource):
                sub_elements.append(CurrentSource(el.name, el.node_p, el.node_n, el.value_at(0.0)))
            elif not isinstance(el, (MutualCoupling, Transformer)):
                sub_elements.append(el)
        # A Transformer carries its windings internally.
        for pair in coupled_pairs:
            if pair["is_transformer"]:
                for j in (1, 2):
                    sub_elements.append(CurrentSource(pair[f"ind{j}_name"], pair[f"n{j}"], pair[f"p{j}"], 0.0))
        sol0 = solve_dc(Circuit(sub_elements, validate=False))

        def _v0(node: str) -> float:
            return 0.0 if node == "0" else sol0.V(node)

        x0 = np.zeros(n, dtype=float)
        for node in node_labels:
            x0[circuit.node_index(node)] = sol0.V(node)
        for el in circuit.elements:
            if isinstance(el, VoltageSource):
                x0[aux(el.name)] = sol0.I_vsource(el.name)

        for c in capacitors:
            if c.name in forced_caps:
                # tensão forçada pela fonte; a corrente em t = 0 desse capacitor não é recuperada
                # e fica em zero (os nós não dependem dela)
                v_c_state[c.name] = _v0(c.node_p) - _v0(c.node_n)
                i_c_state[c.name] = 0.0
            else:
                i_c_state[c.name] = sol0.I_vsource(c.name)
        for ind in inductors:
            v_l_state[ind.name] = _v0(ind.node_p) - _v0(ind.node_n)
        for pair in coupled_pairs:
            v_coupled_state[pair["name"]] = np.array(
                [
                    _v0(pair["p1"]) - _v0(pair["n1"]),
                    _v0(pair["p2"]) - _v0(pair["n2"]),
                ],
                dtype=float,
            )
    else:
        # Operating point DC solve at t = 0
        try:
            # Use the actual t=0 source values, as the Rust backend does.
            dc_elements = [
                replace(el, voltage=el.value_at(0.0), waveform=None) if isinstance(el, VoltageSource)
                else replace(el, current=el.value_at(0.0), waveform=None) if isinstance(el, CurrentSource)
                else el for el in circuit.elements
            ]
            dc_sol = solve_dc(Circuit(dc_elements, validate=False))
            if not np.all(np.isfinite(dc_sol._x)):
                raise ValueError("Non-finite DC operating point.")
            x0 = dc_sol._x[:n].copy()
            for c in capacitors:
                vp = dc_sol.V(c.node_p) if c.node_p != "0" else 0.0
                vn = dc_sol.V(c.node_n) if c.node_n != "0" else 0.0
                v_c_state[c.name] = vp - vn
                i_c_state[c.name] = 0.0

            for ind in inductors:
                v_l_state[ind.name] = 0.0
                i_l_state[ind.name] = dc_sol.I_inductor(ind.name)

            for pair in coupled_pairs:
                v_coupled_state[pair["name"]] = np.zeros(2)
                if pair["is_transformer"]:
                    currents = [dc_sol.I_transformer_primary(pair["name"]),
                                dc_sol.I_transformer_secondary(pair["name"])]
                else:
                    currents = [dc_sol.I_inductor(pair["ind1_name"]),
                                dc_sol.I_inductor(pair["ind2_name"])]
                i_coupled_state[pair["name"]] = np.array(currents, dtype=float)
        except Exception as err:
            raise RuntimeError(
                f"DC operating point initialization failed: {err}. Specify uic=True to use explicit initial conditions."
            ) from err

    # Record t=0 results
    for node in node_labels:
        node_voltages[node][0] = x0[circuit.node_index(node)]

    for c in capacitors:
        branch_currents[c.name][0] = i_c_state[c.name]
    for ind in inductors:
        branch_currents[ind.name][0] = i_l_state[ind.name]
    for pair in coupled_pairs:
        istate = i_coupled_state[pair["name"]]
        if pair["ind1_name"] in branch_currents:
            branch_currents[pair["ind1_name"]][0] = istate[0]
        if pair["ind2_name"] in branch_currents:
            branch_currents[pair["ind2_name"]][0] = istate[1]
        if pair["name"] in branch_currents:
            branch_currents[pair["name"]][0] = istate[0]
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

        # Coupled pairs companion current injection
        ieq_coupled: dict[str, np.ndarray] = {}
        for pair in coupled_pairs:
            G = g_eq_coupled[pair["name"]]
            v_prev = v_coupled_state[pair["name"]]
            i_prev = i_coupled_state[pair["name"]]
            if method == "trapezoidal":
                ieq = i_prev + G @ v_prev
            else:
                ieq = i_prev.copy()
            ieq_coupled[pair["name"]] = ieq
            p1, n1 = idx(pair["p1"]), idx(pair["n1"])
            p2, n2 = idx(pair["p2"]), idx(pair["n2"])
            if p1 is not None: z[p1] -= ieq[0]
            if n1 is not None: z[n1] += ieq[0]
            if p2 is not None: z[p2] -= ieq[1]
            if n2 is not None: z[n2] += ieq[1]

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

        # Update coupled pairs states & currents
        for pair in coupled_pairs:
            p1, n1 = idx(pair["p1"]), idx(pair["n1"])
            p2, n2 = idx(pair["p2"]), idx(pair["n2"])
            vp1 = x[p1] if p1 is not None else 0.0
            vn1 = x[n1] if n1 is not None else 0.0
            vp2 = x[p2] if p2 is not None else 0.0
            vn2 = x[n2] if n2 is not None else 0.0
            v_curr = np.array([vp1 - vn1, vp2 - vn2], dtype=float)
            G = g_eq_coupled[pair["name"]]
            ieq = ieq_coupled[pair["name"]]
            i_curr = G @ v_curr + ieq

            v_coupled_state[pair["name"]] = v_curr
            i_coupled_state[pair["name"]] = i_curr
            if pair["ind1_name"] in branch_currents:
                branch_currents[pair["ind1_name"]][k] = i_curr[0]
            if pair["ind2_name"] in branch_currents:
                branch_currents[pair["ind2_name"]][k] = i_curr[1]
            if pair["name"] in branch_currents:
                branch_currents[pair["name"]][k] = i_curr[0]

        # Update resistors and sources currents
        for el in circuit.elements:
            if isinstance(el, Resistor):
                vp = x[idx(el.node_p)] if idx(el.node_p) is not None else 0.0
                vn = x[idx(el.node_n)] if idx(el.node_n) is not None else 0.0
                branch_currents[el.name][k] = (vp - vn) / el.resistance
            elif isinstance(el, VoltageSource):
                branch_currents[el.name][k] = x[aux(el.name)]

    t_py_end = time.perf_counter()
    total_py_time = t_py_end - t_py_start
    steps_per_sec = (n_steps / total_py_time) if total_py_time > 0 else 0.0

    # Pedagogical energy & power profile
    e_stored = np.zeros(n_steps, dtype=float)
    for c in capacitors:
        vp = node_voltages[c.node_p] if c.node_p != "0" else np.zeros(n_steps)
        vn = node_voltages[c.node_n] if c.node_n != "0" else np.zeros(n_steps)
        e_stored += 0.5 * c.capacitance * ((vp - vn) ** 2)
    for ind in inductors:
        e_stored += 0.5 * ind.inductance * (branch_currents[ind.name] ** 2)

    for pair in coupled_pairs:
        i1 = branch_currents[pair["ind1_name"]]
        i2 = branch_currents[pair["ind2_name"]]
        e_stored += 0.5 * pair["L1"] * i1**2 + pair["M"] * i1 * i2 + 0.5 * pair["L2"] * i2**2

    p_diss = np.zeros(n_steps, dtype=float)
    for el in circuit.elements:
        if isinstance(el, Resistor):
            vp = node_voltages[el.node_p] if el.node_p != "0" else np.zeros(n_steps)
            vn = node_voltages[el.node_n] if el.node_n != "0" else np.zeros(n_steps)
            p_diss += ((vp - vn) ** 2) / el.resistance

    execution_metrics = {
        "backend": "Python (SciPy LU)",
        "total_time_s": total_py_time,
        "steps_per_sec": steps_per_sec,
    }

    return TransientSolution(
        circuit=circuit,
        time=time_vec,
        node_voltages=node_voltages,
        branch_currents=branch_currents,
        energy_stored=e_stored,
        power_dissipated=p_diss,
        execution_metrics=execution_metrics,
    )

