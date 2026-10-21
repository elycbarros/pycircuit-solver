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
    Transformer,
)
from circuit_solver.core.solution import Solution

__all__ = ["build_mna", "build_mna_sparse", "solve"]


def _dc_stamp_entries(circuit: Circuit):
    """Gera as entradas (linha, coluna, valor) da matriz MNA DC e o vetor z.

    Fonte única das estampas DC: ``build_mna`` (denso) e ``build_mna_sparse``
    (esparso) derivam daqui, para que não divirjam.
    """
    n = circuit.size
    idx = circuit.node_index
    aux = circuit.aux_index

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    z = np.zeros(n, dtype=float)

    def add(r, c, v: float) -> None:
        if r is not None and c is not None:
            rows.append(r)
            cols.append(c)
            vals.append(v)

    def stamp_g(p, q, g: float) -> None:
        add(p, p, g)
        add(q, q, g)
        add(p, q, -g)
        add(q, p, -g)

    for el in circuit.elements:
        if isinstance(el, Resistor):
            stamp_g(idx(el.node_p), idx(el.node_n), 1.0 / el.resistance)

        elif isinstance(el, VoltageSource):
            k = aux(el.name)
            p, q = idx(el.node_p), idx(el.node_n)
            add(p, k, 1.0)
            add(k, p, 1.0)
            add(q, k, -1.0)
            add(k, q, -1.0)
            z[k] = el.voltage

        elif isinstance(el, CurrentSource):
            p, q = idx(el.node_p), idx(el.node_n)
            if p is not None:
                z[p] += el.current
            if q is not None:
                z[q] -= el.current

        elif isinstance(el, VCVS):
            k = aux(el.name)
            p, q = idx(el.node_p), idx(el.node_n)
            cp, cn = idx(el.ctrl_p), idx(el.ctrl_n)
            add(p, k, 1.0)
            add(k, p, 1.0)
            add(q, k, -1.0)
            add(k, q, -1.0)
            add(k, cp, -el.gain)
            add(k, cn, el.gain)
            z[k] = 0.0

        elif isinstance(el, VCCS):
            gm = el.transconductance
            p, q = idx(el.node_p), idx(el.node_n)
            cp, cn = idx(el.ctrl_p), idx(el.ctrl_n)
            add(p, cp, -gm)
            add(p, cn, gm)
            add(q, cp, gm)
            add(q, cn, -gm)

        elif isinstance(el, CCCS):
            kv = aux(el.vctrl_name)
            p, q = idx(el.node_p), idx(el.node_n)
            add(p, kv, el.gain)
            add(q, kv, -el.gain)

        elif isinstance(el, CCVS):
            k = aux(el.name)
            kv = aux(el.vctrl_name)
            p, q = idx(el.node_p), idx(el.node_n)
            add(p, k, 1.0)
            add(k, p, 1.0)
            add(q, k, -1.0)
            add(k, q, -1.0)
            add(k, kv, el.transresistance)
            z[k] = 0.0

        elif isinstance(el, OpAmp):
            k = aux(el.name)
            out, gnd = idx(el.out), idx(el.gnd)
            in_p, in_n = idx(el.in_p), idx(el.in_n)
            add(out, k, 1.0)
            add(gnd, k, -1.0)
            add(k, in_p, 1.0)
            add(k, in_n, -1.0)
            z[k] = 0.0

        elif isinstance(el, Inductor):
            stamp_g(idx(el.node_p), idx(el.node_n), 1e6)

        elif isinstance(el, Transformer):
            stamp_g(idx(el.p1), idx(el.n1), 1e6)
            stamp_g(idx(el.p2), idx(el.n2), 1e6)

    return rows, cols, vals, z


def build_mna(circuit: Circuit) -> tuple[np.ndarray, np.ndarray]:
    """Stamp all elements into the dense MNA matrix A and RHS vector z.

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
    rows, cols, vals, z = _dc_stamp_entries(circuit)
    A = np.zeros((n, n), dtype=float)
    np.add.at(
        A,
        (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)),
        np.asarray(vals, dtype=float),
    )
    return A, z


def build_mna_sparse(circuit: Circuit):
    """Stamp all elements into a sparse CSC matrix and RHS vector z.

    Parameters
    ----------
    circuit : Circuit
        Pre-validated circuit with node and auxiliary mappings.

    Returns
    -------
    A_csc : scipy.sparse.csc_matrix
        The sparse MNA coefficient matrix.
    z : np.ndarray
        The right-hand side vector.
    """
    import scipy.sparse as sp

    n = circuit.size
    rows, cols, vals, z = _dc_stamp_entries(circuit)
    A_coo = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)
    return A_coo.tocsc(), z


def solve(
    circuit: Circuit,
    *,
    rtol: float = 1e-9,
    tol: float = 1e-6,
    check_singular: bool = True,
    max_iter: int = 100,
    solver: str = "auto",
    sparse_threshold: int = 500,
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
    solver : {"auto", "dense", "sparse", "rust_sparse"}
        Linear equation solver strategy:
        - "auto": dense for N < sparse_threshold, sparse for N >= sparse_threshold.
        - "dense": numpy.linalg.solve (dense LU/LAPACK).
        - "sparse": scipy.sparse.linalg.spsolve (sparse SuperLU).
        - "rust_sparse": native Rust Faer sparse LU solver.
    sparse_threshold : int
        Node count threshold above which "auto" switches to sparse solver (default 500).

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

    use_sparse = (
        solver == "sparse"
        or solver == "rust_sparse"
        or (solver == "auto" and circuit.size >= sparse_threshold)
    )

    if use_sparse:
        # Check if native Rust sparse solver can be used
        if solver == "rust_sparse":
            from circuit_solver.engine.monte_carlo_rust import (
                _build_rust_descriptors, _has_unsupported_elements, RUST_AVAILABLE,
            )
            if RUST_AVAILABLE and not _has_unsupported_elements(circuit):
                import pycircuit_core as _rust
                (
                    n_nodes,
                    size,
                    resistors,
                    vsources,
                    isources,
                    vcvs,
                    vccs,
                    cccs,
                    ccvs,
                    opamps,
                    inductors,
                ) = _build_rust_descriptors(circuit)
                sol = _rust.solve_mna_sparse(
                    n_nodes,
                    size,
                    resistors,
                    vsources,
                    isources,
                    vcvs=vcvs,
                    vccs=vccs,
                    cccs=cccs,
                    ccvs=ccvs,
                    opamps=opamps,
                    inductors_dc=inductors,
                )
                if sol is not None:
                    return Solution(circuit, np.asarray(sol))

        # Scipy sparse solver (SuperLU / scipy.sparse.linalg.spsolve)
        import scipy.sparse.linalg as spla
        A_csc, z = build_mna_sparse(circuit)
        try:
            x = spla.spsolve(A_csc, z)
            if not np.all(np.isfinite(x)):
                raise LinAlgError("Sparse MNA matrix solution contains non-finite values (NaN/Inf).")
        except Exception as exc:
            raise LinAlgError(
                f"Could not solve sparse MNA system: {exc}. "
                "The circuit may be ill-conditioned or improperly specified."
            ) from exc
        return Solution(circuit, x)

    # Dense solver
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
