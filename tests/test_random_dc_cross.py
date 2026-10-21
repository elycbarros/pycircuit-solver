"""Teste cruzado aleatório dos solvers DC.

1. Estampas: as matrizes densa e esparsa montadas em Python são idênticas.
   Não depende do condicionamento.
2. Soluções: denso, esparso Python e esparso Rust (se compilado) dão as
   mesmas tensões, com tolerância proporcional ao número de condição.
3. Casos DC incompatíveis/indeterminados têm rejeição explícita verificada;
   uma testemunha topológica independente confirma cada diagnóstico.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from circuit_solver import Circuit, solve
from circuit_solver.core.elements import (
    CCCS,
    CCVS,
    VCCS,
    VCVS,
    CurrentSource,
    Inductor,
    Resistor,
    Transformer,
    VoltageSource,
)
from circuit_solver.engine.mna import build_mna, build_mna_sparse, DCConstraintError
from circuit_solver.engine.monte_carlo_rust import RUST_AVAILABLE

N_CASES = 60
REL_MIN = 1e-6
ABS = 1e-8
MAX_COND = 1e12

# Fixed seeds whose exact DC winding constraints have no unique solution.
# These are rejection regressions, not successful circuit simulations.
INVALID_DC = {
    False: {16: "inconsistent", 57: "inconsistent"},
    True: {8: "inconsistent", 20: "inconsistent", 24: "underdetermined",
           26: "underdetermined", 27: "inconsistent", 31: "inconsistent",
           32: "inconsistent", 43: "inconsistent", 47: "inconsistent",
           51: "underdetermined", 57: "inconsistent"},
}


def _assert_invalid_dc(circuit, seed, transformers):
    kind = INVALID_DC[transformers].get(seed)
    if kind is None:
        return False
    for solver in (["dense", "sparse", "rust_sparse"] if RUST_AVAILABLE else ["dense", "sparse"]):
        with pytest.raises(DCConstraintError) as caught:
            solve(circuit, solver=solver)
        assert caught.value.kind == kind, f"seed={seed}, transformers={transformers}, solver={solver}"
    return True


def _random_circuit(seed: int, transformers: bool = False):
    rng = random.Random(seed)
    n = rng.randint(3, 7)
    base = [str(k) for k in range(1, n + 1)]
    els = [VoltageSource("V1", "1", "0", 1.0)]

    # Árvore geradora: cada nó (exceto o 1) liga a um nó anterior
    for k in range(2, n + 1):
        j = rng.randint(1, k - 1)
        els.append(Resistor(f"R{k}", str(k), str(j), rng.uniform(100.0, 10e3)))
    # Resistor ao terra em todo nó base (evita nó com um só terminal)
    for k in base:
        els.append(Resistor(f"RG{k}", k, "0", rng.uniform(1e3, 1e4)))

    extra: list[str] = []  # nós de saída de E e H, sempre referidos ao terra

    def pick() -> str:
        return rng.choice(base + ["0"])

    def pick_pair():
        a = pick()
        b = pick()
        while b == a:
            b = pick()
        return a, b

    for c in range(rng.randint(2, 6)):
        kinds = ["I", "G", "F", "E", "H", "L", "R"] + (["T"] if transformers else [])
        kind = rng.choice(kinds)
        if kind == "I":
            a, b = pick_pair()
            els.append(CurrentSource(f"I{c}", a, b, rng.uniform(-1e-3, 1e-3)))
        elif kind == "G":
            a, b = pick_pair()
            cp, cn = pick_pair()
            els.append(VCCS(f"G{c}", a, b, cp, cn, rng.uniform(-1e-3, 1e-3)))
        elif kind == "F":
            a, b = pick_pair()
            els.append(CCCS(f"F{c}", a, b, "V1", rng.uniform(-2.0, 2.0)))
        elif kind == "E":
            out = f"e{c}"
            cp, cn = pick_pair()
            els.append(VCVS(f"E{c}", out, "0", cp, cn, rng.uniform(-5.0, 5.0)))
            els.append(Resistor(f"RL{c}", out, "0", rng.uniform(1e3, 1e4)))
            extra.append(out)
        elif kind == "H":
            out = f"h{c}"
            els.append(CCVS(f"H{c}", out, "0", "V1", rng.uniform(-1e3, 1e3)))
            els.append(Resistor(f"RL{c}", out, "0", rng.uniform(1e3, 1e4)))
            extra.append(out)
        elif kind == "T":
            a, b = pick_pair()
            c2a, c2b = pick_pair()
            els.append(
                Transformer(f"T{c}", a, b, c2a, c2b, L1=1.0, L2=1.0, coupling=0.9)
            )
        elif kind == "L":
            a, b = pick_pair()
            els.append(Inductor(f"L{c}", a, b, rng.uniform(1e-3, 1e-1)))
        else:
            a, b = pick_pair()
            els.append(Resistor(f"RX{c}", a, b, rng.uniform(100.0, 10e3)))

    return Circuit(els), base + extra


@pytest.mark.parametrize("seed", range(N_CASES))
def test_stamps_match_dense_and_sparse(seed):
    circ, _ = _random_circuit(seed)
    A, z = build_mna(circ)
    As, zs = build_mna_sparse(circ)
    scale_a = max(1.0, float(np.abs(A).max()))
    scale_z = max(1.0, float(np.abs(z).max()))
    assert np.allclose(As.toarray(), A, rtol=0.0, atol=1e-12 * scale_a)
    assert np.allclose(zs, z, rtol=0.0, atol=1e-12 * scale_z)


def _solve_reference(circ, seed, *, transformers=False):
    try:
        ref = solve(circ, solver="auto")
    except np.linalg.LinAlgError as exc:
        cond = float(np.linalg.cond(build_mna(circ)[0]))
        # A numerical exception on a well-conditioned circuit is a regression.
        # NaN also fails this comparison and must not become a skip.
        if not cond > MAX_COND:
            raise
        pytest.skip(
            f"seed={seed}, transformers={transformers}: falha numérica esperada "
            f"(cond={cond:.2e} > {MAX_COND:.0e}); {exc}"
        )
    cond = float(np.linalg.cond(build_mna(circ)[0]))
    if cond > MAX_COND:
        pytest.skip(
            f"seed={seed}, transformers={transformers}: comparação instável "
            f"(cond={cond:.2e} > {MAX_COND:.0e})"
        )
    assert np.isfinite(cond), f"seed={seed}: condicionamento não finito"
    return ref, cond


@pytest.mark.parametrize("seed", range(N_CASES))
def test_dc_solvers_agree(seed):
    circ, nodes = _random_circuit(seed)
    if _assert_invalid_dc(circ, seed, False):
        return
    ref, cond = _solve_reference(circ, seed)
    rel = max(REL_MIN, 100 * np.finfo(float).eps * cond)
    _assert_same(ref, solve(circ, solver="sparse"), nodes, "sparse", rel)
    if RUST_AVAILABLE:
        _assert_same(ref, solve(circ, solver="rust_sparse"), nodes, "rust_sparse", rel)


def _assert_same(ref, other, nodes, label: str, rel: float) -> None:
    for k in nodes:
        assert other.V(k) == pytest.approx(ref.V(k), rel=rel, abs=ABS), (
            f"{label}: V({k}) diverge do denso"
        )


def test_random_circuits_have_expected_dc_solvability():
    for transformers in (False, True):
        solved = 0
        for seed in range(N_CASES):
            circ, _ = _random_circuit(seed, transformers=transformers)
            try:
                solve(circ, solver="auto")
                solved += 1
            except np.linalg.LinAlgError:
                cond = float(np.linalg.cond(build_mna(circ)[0]))
                if not cond > MAX_COND:
                    raise
        expected = N_CASES - len(INVALID_DC[transformers])
        assert solved == expected, f"transformers={transformers}: {solved}/{N_CASES}, expected {expected}"


@pytest.mark.parametrize("seed", range(N_CASES))
def test_transformer_stamps_match(seed):
    circ, _ = _random_circuit(seed, transformers=True)
    A, z = build_mna(circ)
    As, zs = build_mna_sparse(circ)
    scale_a = max(1.0, float(np.abs(A).max()))
    scale_z = max(1.0, float(np.abs(z).max()))
    assert np.allclose(As.toarray(), A, rtol=0.0, atol=1e-12 * scale_a)
    assert np.allclose(zs, z, rtol=0.0, atol=1e-12 * scale_z)


@pytest.mark.parametrize("seed", range(N_CASES))
def test_transformer_dc_solvers_agree(seed):
    circ, nodes = _random_circuit(seed, transformers=True)
    if _assert_invalid_dc(circ, seed, True):
        return
    ref, cond = _solve_reference(circ, seed, transformers=True)
    rel = max(REL_MIN, 100 * np.finfo(float).eps * cond)
    _assert_same(ref, solve(circ, solver="sparse"), nodes, "sparse", rel)
    if RUST_AVAILABLE:
        _assert_same(ref, solve(circ, solver="rust_sparse"), nodes, "rust_sparse", rel)


@pytest.mark.parametrize("target", [test_dc_solvers_agree,
                                   test_transformer_dc_solvers_agree,
                                   test_random_circuits_have_expected_dc_solvability])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyError, np.linalg.LinAlgError])
def test_unexpected_errors_are_never_hidden(monkeypatch, target, error_type):
    def broken_solve(*args, **kwargs):
        raise error_type("injected regression")

    monkeypatch.setitem(globals(), "solve", broken_solve)
    # Even a LinAlgError must propagate if the matrix is well conditioned.
    monkeypatch.setitem(globals(), "build_mna", lambda _: (np.eye(2), np.zeros(2)))
    with pytest.raises(error_type, match="injected regression"):
        if target is test_random_circuits_have_expected_dc_solvability:
            target()
        else:
            target(0)


@pytest.mark.parametrize("transformers", [False, True])
def test_expected_numerical_skip_has_reproducible_diagnostic(monkeypatch, transformers):
    def singular_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("injected numerical failure")

    monkeypatch.setitem(globals(), "solve", singular_solve)
    monkeypatch.setitem(globals(), "build_mna", lambda _: (np.diag([1., 1e-15]), np.zeros(2)))
    with pytest.raises(pytest.skip.Exception, match=rf"seed=42, transformers={transformers}.*cond="):
        _solve_reference(None, 42, transformers=transformers)


@pytest.mark.parametrize('transformers,seed,kind', [
    (flag, seed, kind) for flag, cases in INVALID_DC.items() for seed, kind in cases.items()
])
def test_invalid_random_constraints_have_a_topological_witness(transformers, seed, kind):
    # Independent of matrix ranks: zero-voltage windings merge DC potentials.
    # A nonzero source inside a merged component is contradictory. A winding
    # cycle permits an arbitrary circulating DC current.
    circ, _ = _random_circuit(seed, transformers=transformers)
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            node = parent[node]
        return node

    cycle = False
    for el in circ.elements:
        branches = [(el.node_p, el.node_n)] if isinstance(el, Inductor) else (
            [(el.p1, el.n1), (el.p2, el.n2)] if isinstance(el, Transformer) else [])
        for p, n in branches:
            a, b = find(p), find(n)
            if a == b:
                cycle = True
            else:
                parent[a] = b
    conflict = any(isinstance(el, VoltageSource) and el.voltage != 0
                   and find(el.node_p) == find(el.node_n) for el in circ.elements)
    assert conflict == (kind == 'inconsistent')
    if kind == 'underdetermined':
        assert cycle
