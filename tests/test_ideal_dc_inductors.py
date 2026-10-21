"""Exact DC winding constraints, solved currents and analysis-specific dimensions."""
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.sparse.linalg import MatrixRankWarning

from circuit_solver import (
    Circuit, Resistor, VoltageSource, Inductor, Transformer, Diode,
    solve, solve_ac, solve_transient,
)
from circuit_solver.engine.mna import build_mna, build_mna_sparse, DCConstraintError
from circuit_solver.engine.monte_carlo_rust import monte_carlo, RUST_AVAILABLE

RUST = pytest.mark.skipif(not RUST_AVAILABLE, reason='Rust extension is optional')
SOLVERS = ['dense', 'sparse', pytest.param('rust_sparse', marks=RUST)]
BACKENDS = ['python', pytest.param('rust', marks=RUST)]


def series_rl(resistance=10, reverse=False):
    return Circuit([VoltageSource('V', 'in', '0', 12),
                    Resistor('R', 'in', 'out', resistance),
                    Inductor('L', '0' if reverse else 'out', 'out' if reverse else '0', .01)])


@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('resistance', [.001, 10., 1e9])
@pytest.mark.parametrize('reverse', [False, True])
def test_exact_short_voltage_and_current_no_artificial_resistance(solver, resistance, reverse):
    c = series_rl(resistance, reverse)
    r = solve(c, solver=solver)
    assert abs(r.V('out')) < 1e-14
    assert r.I_inductor('L') == pytest.approx((-1 if reverse else 1)*12/resistance, rel=1e-12)
    assert r.I_vsource('V') == pytest.approx(-12/resistance, rel=1e-12)
    audit = r.power_audit()
    assert audit['L'] == 0
    assert abs(audit['_total']) <= 1e-10*max(1, abs(audit['V']))
    assert 'DC ideal short' in r.summary()


def test_exact_incidence_stamp_and_analysis_specific_dimensions():
    c = series_rl()
    A, z = build_mna(c)
    np.testing.assert_array_equal(A, [[.1, -.1, 1, 0], [-.1, .1, 0, 1],
                                     [1, 0, 0, 0], [0, 1, 0, 0]])
    np.testing.assert_array_equal(z, [0, 0, 12, 0])
    sparse, sparse_z = build_mna_sparse(c)
    np.testing.assert_array_equal(sparse.toarray(), A)
    np.testing.assert_array_equal(sparse_z, z)
    assert c.size == 3 and c.dc_size == 4
    assert c.aux_index('V') == 2 and c.dc_inductor_index('L') == 3
    assert len(solve(c)._x) == c.dc_size


@pytest.mark.parametrize('solver', SOLVERS)
def test_floating_winding_endpoints_and_absorbing_source(solver):
    c = Circuit([VoltageSource('V1', 'in', '0', 12), VoltageSource('V2', 'ref', '0', 5),
                 Resistor('R', 'in', 'out', 7), Inductor('L', 'out', 'ref', .1)])
    r = solve(c, solver=solver)
    assert r.V('out') == pytest.approx(5, abs=1e-14)
    assert r.I_inductor('L') == pytest.approx(1, abs=1e-14)
    assert r.P_vsource('V2') == pytest.approx(-5)
    assert abs(r.power_audit()['_total']) < 1e-12


@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('k', [-1., 0., 1.])
def test_each_transformer_winding_has_independent_dc_current(solver, k):
    c = Circuit([VoltageSource('V1', 'a', '0', 12), Resistor('R1', 'a', 'p', 4),
                 VoltageSource('V2', 'b', '0', 5), Resistor('R2', 'b', 's', 10),
                 Transformer('T', 'p', '0', 's', '0', L1=.1, L2=.4, coupling=k)])
    r = solve(c, solver=solver)
    np.testing.assert_allclose([r.V('p'), r.V('s')], 0, atol=1e-14)
    assert r.I_transformer_primary('T') == pytest.approx(3)
    assert r.I_transformer_secondary('T') == pytest.approx(.5)
    assert c.dc_size == c.size+2
    assert abs(r.power_audit()['_total']) < 1e-12


@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('kind', ['inconsistent', 'underdetermined'])
def test_ideal_constraints_without_unique_dc_solution_are_rejected(solver, kind):
    c = (Circuit([VoltageSource('V', 'p', '0', 1), Inductor('L', 'p', '0', .01)])
         if kind == 'inconsistent' else
         Circuit([Resistor('R', 'p', '0', 10), Inductor('L1', 'p', '0', .01),
                  Inductor('L2', 'p', '0', .02)]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MatrixRankWarning)
        with pytest.raises(DCConstraintError) as caught:
            solve(c, solver=solver)
        assert caught.value.kind == kind
        assert caught.value.rank < c.dc_size
        assert (caught.value.augmented_rank > caught.value.rank) == (kind == 'inconsistent')


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('method', ['euler', 'trapezoidal'])
def test_transient_dc_current_is_exact_and_stationary(backend, method):
    r = solve_transient(series_rl(), t_stop=.001, dt=1e-6, method=method, backend=backend)
    np.testing.assert_allclose(r.I('L'), 1.2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(r.V('out'), 0, atol=1e-12)
    np.testing.assert_allclose(r.E_stored(), .5*.01*1.2**2, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('backend', BACKENDS)
def test_parallel_inductor_initial_currents_require_explicit_uic(backend):
    c = Circuit([Resistor('R', 'p', '0', 10), Inductor('L1', 'p', '0', .01, ic=.2),
                 Inductor('L2', 'p', '0', .02, ic=-.2)])
    with pytest.raises((RuntimeError, ValueError), match='DC operating point initialization failed'):
        solve_transient(c, t_stop=.001, dt=1e-6, backend=backend)
    r = solve_transient(c, t_stop=.001, dt=1e-6, backend=backend, uic=True)
    np.testing.assert_allclose(r.V('p'), 0, atol=1e-14)
    np.testing.assert_allclose(r.I('L1'), .2, atol=1e-14)
    np.testing.assert_allclose(r.I('L2'), -.2, atol=1e-14)


def test_nonlinear_dc_shares_ideal_winding_stamp():
    c = Circuit([VoltageSource('V', 'in', '0', 5), Resistor('R', 'in', 'd', 1000),
                 Diode('D', 'd', 'p'), Inductor('L', 'p', '0', .01)])
    ref_voltage = brentq(lambda v: (5-v)/1000 - 1e-14*np.expm1(v/.02585), .1, .9)
    r = solve(c)
    assert abs(r.V('p')) < 1e-14
    assert r.V('d') == pytest.approx(ref_voltage, abs=1e-7)
    assert r.I_inductor('L') == pytest.approx((5-ref_voltage)/1000, rel=1e-7)
    assert r.I_inductor('L') == pytest.approx(r.I_diode('D'), rel=1e-7)


def test_dc_extension_does_not_change_ac_dimensions_or_impedance():
    c = series_rl()
    c.elements[0].ac_mag = 1
    r = solve_ac(c, frequency=1000)
    assert len(r.x) == c.size < c.dc_size
    assert r.I_inductor('L') == pytest.approx(1/(10+2j*np.pi*1000*.01))


@pytest.mark.parametrize('backend', BACKENDS)
def test_monte_carlo_winding_voltage_is_exactly_zero(backend):
    c = series_rl()
    c.elements[1].tolerance = .05
    r = monte_carlo(c, n_samples=100, seed=42, force_python=(backend == 'python'))
    assert r.n_failed == 0
    np.testing.assert_array_equal(r.samples[:, r.node_labels.index('out')], np.zeros(100))


@pytest.mark.parametrize('function_name', ['solve_mna_single', 'solve_mna_sparse'])
@RUST
def test_direct_rust_caller_can_append_ideal_current_automatically(function_name):
    import pycircuit_core as rust
    out = getattr(rust, function_name)(
        2, 3, [rust.PyResistor(10, 0, 0, 1)], [rust.PyVSource(12, 0, -1, 2)], [],
        inductors_dc=[rust.PyInductorDC(1, -1)],
    )
    assert out[1] == pytest.approx(0, abs=1e-14)
    if function_name == 'solve_mna_sparse':
        assert len(out) == 4
        assert out[3] == pytest.approx(1.2)


def test_dc_sweep_exposes_the_solved_winding_current():
    from circuit_solver import dc_sweep
    c = series_rl()
    r = dc_sweep(c, 'V', 0, 10, 2)
    np.testing.assert_allclose(r.I('L'), np.arange(0, 11, 2)/10)
    np.testing.assert_allclose(r.V('out'), 0, atol=1e-14)
    assert c.elements[0].voltage == 12


def test_nonlinear_extension_keeps_the_linear_dc_source_conventions():
    from circuit_solver import VCCS
    elements = [VoltageSource('V', 'in', '0', 1), Inductor('L', 'in', 'ctrl', .01),
                Resistor('Rc', 'ctrl', '0', 1000), VCCS('G', 'out', '0', 'ctrl', '0', .001),
                Resistor('Rout', 'out', '0', 1000)]
    linear = solve(Circuit(elements))
    nonlinear = solve(Circuit(elements + [Diode('Doff', '0', 'out')]))
    assert nonlinear.V('out') == pytest.approx(linear.V('out'), abs=1e-7)
    assert nonlinear.V('out') == pytest.approx(1, abs=1e-7)
    assert nonlinear.I_inductor('L') == pytest.approx(.001, abs=1e-12)


@pytest.mark.parametrize('backend', BACKENDS)
def test_monte_carlo_never_returns_zero_statistics_for_indeterminate_dc(backend):
    c = Circuit([Resistor('R', 'p', '0', 10), Inductor('L1', 'p', '0', .01),
                 Inductor('L2', 'p', '0', .02)])
    with pytest.raises(RuntimeError, match='All Monte Carlo samples failed'):
        monte_carlo(c, n_samples=10, seed=42, force_python=(backend == 'python'))
