"""Teste cruzado aleatório dos solvers DC.

1. Estampas: as matrizes densa e esparsa montadas em Python são idênticas.
   Não depende do condicionamento.
2. Soluções: denso, esparso Python e esparso Rust (se compilado) dão as
   mesmas tensões, com tolerância proporcional ao número de condição.
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
    VoltageSource,
)
from circuit_solver.engine.mna import build_mna, build_mna_sparse
from circuit_solver.engine.monte_carlo_rust import RUST_AVAILABLE

N_CASES = 60
REL_MIN = 1e-6
ABS = 1e-8
MAX_COND = 1e12


def _random_circuit(seed: int):
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
        kind = rng.choice(["I", "G", "F", "E", "H", "L", "R"])
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


@pytest.mark.parametrize("seed", range(N_CASES))
def test_dc_solvers_agree(seed):
    circ, nodes = _random_circuit(seed)
    try:
        ref = solve(circ, solver="auto")
    except Exception:
        pytest.skip("circuito aleatório singular")
    cond = np.linalg.cond(build_mna(circ)[0])
    if cond > MAX_COND:
        pytest.skip(f"mal condicionado (cond = {cond:.1e})")
    rel = max(REL_MIN, 100 * np.finfo(float).eps * cond)
    _assert_same(ref, solve(circ, solver="sparse"), nodes, "sparse", rel)
    if RUST_AVAILABLE:
        _assert_same(ref, solve(circ, solver="rust_sparse"), nodes, "rust_sparse", rel)


def _assert_same(ref, other, nodes, label: str, rel: float) -> None:
    for k in nodes:
        assert other.V(k) == pytest.approx(ref.V(k), rel=rel, abs=ABS), (
            f"{label}: V({k}) diverge do denso"
        )


def test_random_circuits_are_mostly_solvable():
    solved = 0
    for seed in range(N_CASES):
        circ, _ = _random_circuit(seed)
        try:
            solve(circ, solver="auto")
            solved += 1
        except Exception:
            pass
    assert solved >= int(0.8 * N_CASES), f"só {solved}/{N_CASES} resolvíveis"
