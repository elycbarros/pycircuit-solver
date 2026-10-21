"""Ordem de convergência do transiente com condições iniciais (UIC).

Referência analítica. Com dt caindo pela metade, o erro máximo do método trapezoidal
deve cair por ~4 (ordem 2), e o de Euler implícito por ~2 (ordem 1). Antes da correção
do t = 0 com UIC, o trapezoidal caía só por ~2 (ordem 1), porque a amostra inicial vinha
do modelo companheiro e i_C(0) era zero.

Também cobre i_C(0) = -v0/R na descarga RC, que é a derivada da tensão no t = 0+.
"""

import math

import numpy as np
import pytest

from circuit_solver import Capacitor, Circuit, Inductor, Resistor, solve_transient

R_RC, C_RC, V0 = 1e3, 1e-6, 1.0
TAU = R_RC * C_RC  # 1 ms


def _rc_decay_error(method: str, dt: float, t_stop: float = 3e-3) -> float:
    c = Circuit([Resistor("R1", "2", "0", R_RC), Capacitor("C1", "2", "0", C_RC, ic=V0)])
    ts = solve_transient(c, t_stop=t_stop, dt=dt, method=method, uic=True, backend="python")
    t = ts.time
    return float(np.max(np.abs(ts.V("2") - V0 * np.exp(-t / TAU))))


def _order(method: str, dts=(1e-4, 5e-5, 2.5e-5)) -> float:
    e = [_rc_decay_error(method, dt) for dt in dts]
    return math.log(e[0] / e[-1]) / math.log(dts[0] / dts[-1])


def test_trapezoidal_is_second_order_in_RC_decay():
    p = _order("trapezoidal")
    assert 1.85 <= p <= 2.15, f"ordem observada {p:.3f}, esperada ~2"


def test_euler_is_first_order_in_RC_decay():
    p = _order("euler")
    assert 0.9 <= p <= 1.1, f"ordem observada {p:.3f}, esperada ~1"


def test_trapezoidal_error_at_fine_step_is_small():
    assert _rc_decay_error("trapezoidal", 1e-5) < 1e-5


def test_capacitor_initial_current_is_minus_v0_over_R():
    # i_C(0+) = C dv/dt(0+) = -v0/R, descarga pelo resistor em paralelo
    c = Circuit([Resistor("R1", "2", "0", R_RC), Capacitor("C1", "2", "0", C_RC, ic=V0)])
    ts = solve_transient(c, t_stop=1e-4, dt=1e-7, method="trapezoidal", uic=True, backend="python")
    assert ts.I("C1")[0] == pytest.approx(-V0 / R_RC, rel=1e-3)


def test_rlc_series_underdamped_matches_analytic():
    R, L, C = 10.0, 1e-3, 1e-6
    circ = Circuit([
        Resistor("R1", "1", "2", R),
        Inductor("L1", "2", "0", L),
        Capacitor("C1", "1", "0", C, ic=1.0),
    ])
    a = R / (2 * L)
    wd = math.sqrt(1 / (L * C) - a * a)
    ts = solve_transient(circ, t_stop=1e-3, dt=1e-6, method="trapezoidal", uic=True, backend="python")
    t = ts.time
    vex = np.exp(-a * t) * (np.cos(wd * t) + (a / wd) * np.sin(wd * t))
    assert float(np.max(np.abs(ts.V("1") - vex))) < 5e-3


def test_capacitor_across_voltage_source_with_inconsistent_ic():
    # Capacitor em paralelo com fonte de tensão: a tensão é fixada pela fonte, e o ic
    # (0.5 V) não pode criar fontes em paralelo nem tornar o sistema singular.
    circ = Circuit.from_netlist(
        "t\nV1 1 0 DC 1\nC1 1 0 1u IC=0.5\nR1 1 2 1k\nR2 2 0 1k\n.end\n"
    )
    ts = solve_transient(circ, t_stop=1e-3, dt=1e-5, method="trapezoidal", uic=True, backend="python")
    assert np.allclose(ts.V("1"), 1.0, atol=1e-9)
    assert ts.V("2")[-1] == pytest.approx(0.5, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="Fase 2.3 pendente: o backend Rust ainda calcula o t=0 com UIC pelo modelo companheiro",
)
def test_rust_backend_trapezoidal_is_second_order():
    from circuit_solver.engine.transient_rust import RUST_AVAILABLE

    if not RUST_AVAILABLE:
        pytest.skip("Rust não compilado nesta máquina")

    def err(dt: float) -> float:
        c = Circuit([Resistor("R1", "2", "0", R_RC), Capacitor("C1", "2", "0", C_RC, ic=V0)])
        ts = solve_transient(c, t_stop=3e-3, dt=dt, method="trapezoidal", uic=True, backend="rust")
        return float(np.max(np.abs(ts.V("2") - V0 * np.exp(-ts.time / TAU))))

    dts = (1e-4, 5e-5, 2.5e-5)
    p = math.log(err(dts[0]) / err(dts[-1])) / math.log(dts[0] / dts[-1])
    assert 1.85 <= p <= 2.15, f"ordem observada {p:.3f}, esperada ~2"
