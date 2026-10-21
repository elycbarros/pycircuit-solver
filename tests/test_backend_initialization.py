"""Regression checks for coupled states, failed initialization and controlled sources."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import trapezoid

from circuit_solver import (
    Circuit, VoltageSource, CurrentSource, Resistor, Capacitor, Inductor,
    MutualCoupling, Transformer, CCCS, CCVS, StepWaveform,
    solve, solve_transient, analyze_sensitivity,
)
from circuit_solver.engine.transient_rust import RUST_AVAILABLE

BACKENDS = ['python', pytest.param('rust', marks=pytest.mark.skipif(
    not RUST_AVAILABLE, reason='Rust extension is optional'))]


def coupled_circuit(k=.5, ic1=.2, ic2=-.1, transformer=False):
    coils = ([Transformer('T', 'p', '0', 's', '0', L1=.02, L2=.005, coupling=k)]
             if transformer else [Inductor('L1', 'p', '0', .02, ic=ic1),
                                  Inductor('L2', 's', '0', .005, ic=ic2),
                                  MutualCoupling('K', 'L1', 'L2', k)])
    return Circuit([VoltageSource('V', 'in', '0', 10),
                    Resistor('R1', 'in', 'p', 10), Resistor('R2', 's', '0', 20)] + coils)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('method', ['euler', 'trapezoidal'])
@pytest.mark.parametrize('k', [.5, -.5])
def test_coupled_initial_currents_and_energy_against_exponential(backend, method, k):
    c = coupled_circuit(k)
    r = solve_transient(c, t_stop=.001, dt=1e-6, uic=True, method=method, backend=backend)
    initial = np.array([.2, -.1])
    L = np.array([[.02, k * .01], [k * .01, .005]])
    dynamics = -np.linalg.solve(L, np.diag([10, 20]))
    steady = np.array([1., 0.])
    indices = [0, 100, 500, 1000]
    expected = np.array([steady + expm(dynamics * r.time[j]) @ (initial-steady) for j in indices])
    actual = np.column_stack([r.I('L1'), r.I('L2')])
    np.testing.assert_allclose(actual[indices], expected, atol=2e-4)
    np.testing.assert_allclose(actual[0], initial, atol=1e-15)
    assert r.V('p')[0] == pytest.approx(8)
    assert r.V('s')[0] == pytest.approx(2)
    expected_energy = .5 * np.einsum('ti,ij,tj->t', actual, L, actual)
    np.testing.assert_allclose(r.E_stored(), expected_energy, atol=1e-14)
    assert r.E_stored()[0] == pytest.approx(.000325 if k > 0 else .000525)
    if method == 'trapezoidal':
        supplied = trapezoid(-10 * r.I('V'), r.time)
        dissipated = trapezoid(r.P_dissipated(), r.time)
        assert supplied == pytest.approx(dissipated + r.E_stored()[-1] - r.E_stored()[0], abs=2e-8)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('transformer', [False, True])
def test_dc_initialization_includes_both_windings(backend, transformer):
    c = coupled_circuit(transformer=transformer)
    pair = c.get_coupled_pairs()[0]
    r = solve_transient(c, t_stop=.001, dt=1e-6, backend=backend)
    expected = 10 / (10 + 1e-6)  # The engine's declared 1 micro-ohm DC short.
    assert r.I(pair['ind1_name'])[0] == pytest.approx(expected, rel=1e-9)
    assert r.I(pair['ind2_name'])[0] == pytest.approx(0, abs=1e-12)
    assert r.V('p')[0] == pytest.approx(expected * 1e-6, abs=1e-12)
    assert r.E_stored()[0] == pytest.approx(.5 * .02 * expected**2, rel=1e-9)
    np.testing.assert_allclose(r.I(pair['ind1_name']), 1., atol=3e-7)
    np.testing.assert_allclose(r.I(pair['ind2_name']), 0., atol=3e-7)


@pytest.mark.parametrize('backend', BACKENDS)
def test_failed_dc_initialization_is_explicit_and_uic_is_opt_in(backend):
    # Capacitor makes the transient system solvable, but the DC system is open.
    c = Circuit([CurrentSource('I', 'out', '0', .001),
                 Capacitor('C', 'out', '0', 1e-6, ic=1)])
    with pytest.raises((ValueError, RuntimeError), match='DC operating point initialization failed'):
        solve_transient(c, t_stop=.001, dt=1e-5, backend=backend)
    r = solve_transient(c, t_stop=.001, dt=1e-5, uic=True, backend=backend)
    np.testing.assert_allclose(r.V('out'), 1 + 1000*r.time, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS)
def test_dc_start_uses_waveform_at_zero(backend):
    c = Circuit([VoltageSource('V', 'in', '0', 99,
                               waveform=StepWaveform(v1=3, v2=7, t_step=.0005)),
                 Resistor('R', 'in', 'out', 1000), Capacitor('C', 'out', '0', 1e-6)])
    r = solve_transient(c, t_stop=.001, dt=1e-5, backend=backend)
    assert r.V('in')[0] == pytest.approx(3)
    assert r.V('out')[0] == pytest.approx(3)


@pytest.mark.parametrize('kind', [CCCS, CCVS])
@pytest.mark.parametrize('gain', [2., -2.])
def test_current_controlled_sensitivity_and_corners(kind, gain):
    device = kind('F' if kind is CCCS else 'H', 'out', '0', 'Vctrl', gain)
    c = Circuit([CurrentSource('I', 'sense', '0', .001),
                 VoltageSource('Vctrl', 'sense', '0', 0),
                 Resistor('R', 'out', '0', 1000), device])
    before = deepcopy(c.elements)
    r = analyze_sensitivity(c, 'out', custom_tolerances={'I': 0, 'R': 0, device.name: .1})
    component = r.components[device.name]
    expected_derivative = -1. if kind is CCCS else -.001
    assert component.abs_sensitivity == pytest.approx(expected_derivative)
    h = 1e-5
    low, high = deepcopy(c), deepcopy(c)
    attribute = 'gain' if kind is CCCS else 'transresistance'
    setattr(low.elements[-1], attribute, gain-h)
    setattr(high.elements[-1], attribute, gain+h)
    finite_difference = (solve(high).V('out')-solve(low).V('out')) / (2*h)
    assert component.abs_sensitivity == pytest.approx(finite_difference, rel=1e-8)
    nominal = expected_derivative*gain
    delta = abs(nominal)*.1
    assert r.worst_case_min == pytest.approx(nominal-delta)
    assert r.worst_case_max == pytest.approx(nominal+delta)
    assert r.rss_max == pytest.approx(nominal+delta)
    assert c.elements == before


@pytest.mark.parametrize('kind', [CCCS, CCVS])
@pytest.mark.parametrize('tol', [-.1, float('nan')])
def test_invalid_current_controlled_tolerance_rejected(kind, tol):
    c = Circuit([CurrentSource('I', 'sense', '0', .001),
                 VoltageSource('Vctrl', 'sense', '0', 0),
                 Resistor('R', 'out', '0', 1000), kind('X', 'out', '0', 'Vctrl', 2)])
    with pytest.raises(ValueError, match='Invalid tolerance'):
        analyze_sensitivity(c, 'out', custom_tolerances={'X': tol})


@pytest.mark.skipif(not RUST_AVAILABLE, reason='Rust extension is optional')
@pytest.mark.parametrize('method', ['euler', 'trapezoidal'])
@pytest.mark.parametrize('uic', [False, True])
@pytest.mark.parametrize('transformer', [False, True])
def test_coupled_backends_match_all_samples(method, uic, transformer):
    c = coupled_circuit(k=-.5, transformer=transformer)
    py = solve_transient(c, t_stop=.001, dt=1e-6, uic=uic, method=method, backend='python')
    rs = solve_transient(c, t_stop=.001, dt=1e-6, uic=uic, method=method, backend='rust')
    for node in c.nodes:
        np.testing.assert_allclose(rs.V(node), py.V(node), rtol=1e-9, atol=1e-12)
    pair = c.get_coupled_pairs()[0]
    for name in (pair['ind1_name'], pair['ind2_name']):
        np.testing.assert_allclose(rs.I(name), py.I(name), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rs.E_stored(), py.E_stored(), rtol=1e-9, atol=1e-14)
