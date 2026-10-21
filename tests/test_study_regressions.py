"""Independent mathematical checks for the personal-study API and reports."""
import importlib
import math
from copy import deepcopy

import numpy as np
import pytest

from circuit_solver import (
    Circuit, VoltageSource, Resistor, Inductor, MutualCoupling,
    Transformer, Diode, ZenerDiode, BJT, MOSFET, dc_sweep, solve_transient,
    analyze_two_wattmeters, analyze_sensitivity,
)


def divider():
    return Circuit([VoltageSource('V1', 'in', '0', 10),
                    Resistor('R1', 'in', 'out', 1000), Resistor('R2', 'out', '0', 1000)])


@pytest.mark.parametrize('start,stop,step,expected', [
    (0, 1, .3, [0, .3, .6, .9]), (1, 0, -.3, [1, .7, .4, .1]),
    (0, .3, .1, [0, .1, .2, .3]), (2, 2, 1, [2]),
])
def test_sweep_step_and_input_preserved(start, stop, step, expected):
    c = divider()
    before = deepcopy(c.elements)
    r = dc_sweep(c, 'V1', start, stop, step)
    np.testing.assert_allclose(r.sweep_values, expected, atol=1e-15)
    np.testing.assert_allclose(r.V('out'), np.array(expected) / 2)
    assert c.elements == before


@pytest.mark.parametrize('kwargs', [dict(start=0, stop=1, step=-.1),
    dict(start=0, stop=1, step=0), dict(start=0, stop=math.inf, step=1),
    dict(start=0, stop=1, step=1, param2_name='R1'),
    dict(start=0, stop=1, step=1, param2_name='R1', start2=1, stop2=2, step2=0)])
def test_invalid_sweep_rejected(kwargs):
    with pytest.raises(ValueError):
        dc_sweep(divider(), 'V1', **kwargs)


def test_nested_sweep_and_failure_preserve_input(monkeypatch):
    c = divider()
    r = dc_sweep(c, 'V1', 0, 1, .3, param2_name='R1', start2=1000, stop2=2000, step2=600)
    np.testing.assert_allclose(r.V('out'), np.array([r.sweep_values / 2, r.sweep_values / 2.6]))
    assert c.elements[0].voltage == 10 and c.elements[1].resistance == 1000
    module = importlib.import_module('circuit_solver.engine.dc_sweep')
    def fail(_):
        raise RuntimeError('injected solve failure')
    monkeypatch.setattr(module, 'solve', fail)
    with pytest.raises(RuntimeError):
        dc_sweep(c, 'V1', 0, 1, .3)
    assert c.elements[0].voltage == 10


@pytest.mark.parametrize('device', [Diode('D', 'out', '0'), ZenerDiode('D', 'out', '0', 5.1),
    BJT('Q', 'out', 'in', '0'), MOSFET('M', 'out', 'in', '0')])
@pytest.mark.parametrize('backend', ['python', 'auto'])
def test_transient_never_ignores_semiconductors(device, backend):
    c = divider()
    with pytest.raises(ValueError, match='linear'):
        solve_transient(Circuit(c.elements + [device]), t_stop=.01, dt=.001, backend=backend)


@pytest.mark.parametrize('method', ['euler', 'trapezoidal'])
@pytest.mark.parametrize('transformer', [False, True])
def test_python_coupled_inductors_against_state_space(method, transformer):
    from scipy.linalg import expm
    # L di/dt + R i = [1, 0]; negative mutual inductance tests dot polarity.
    L = np.array([[.02, -.005], [-.005, .005]])
    coils = ([Transformer('T', 'p', '0', 's', '0', L1=.02, L2=.005, coupling=-.5)]
             if transformer else [Inductor('L1', 'p', '0', .02), Inductor('L2', 's', '0', .005),
                                  MutualCoupling('K', 'L1', 'L2', -.5)])
    c = Circuit([VoltageSource('V', 'in', '0', 1), Resistor('R1', 'in', 'p', 10),
                 Resistor('R2', 's', '0', 20)] + coils)
    r = solve_transient(c, t_stop=.002, dt=1e-6, method=method, uic=True, backend='python')
    pairs = c.get_coupled_pairs()[0]
    i1, i2 = r.I(pairs['ind1_name']), r.I(pairs['ind2_name'])
    steady = np.array([.1, 0])
    exact = steady - expm(-np.linalg.solve(L, np.diag([10, 20])) * r.time[-1]) @ steady
    np.testing.assert_allclose([i1[-1], i2[-1]], exact, atol=2e-5)
    energy = .5 * L[0, 0] * i1**2 + L[0, 1] * i1 * i2 + .5 * L[1, 1] * i2**2
    np.testing.assert_allclose(r.E_stored(), energy, atol=1e-14)


@pytest.mark.parametrize('rotation', [-1, 1])
@pytest.mark.parametrize('impedances', [[20+15j]*3, [10+2j, 30-5j, 20+15j]])
def test_three_wire_power_unbalanced_both_sequences(rotation, impedances):
    v = 120 * np.exp(rotation * 2j * np.pi * np.arange(3) / 3)
    z = np.array(impedances)
    neutral = np.sum(v/z) / np.sum(1/z)
    i = (v-neutral)/z
    assert abs(sum(i)) < 1e-12
    expected = sum((v-neutral)*np.conj(i))
    r = analyze_two_wattmeters(v[0]-v[2], v[1]-v[2], i[0], i[1])
    assert r.P_total == pytest.approx(expected.real)
    assert r.Q_total == pytest.approx(expected.imag)
    assert r.S_total == pytest.approx(abs(expected))


def test_negative_source_corner_and_rss_meaning():
    c = divider()
    c.elements[0].voltage = -10
    r = analyze_sensitivity(c, 'out', custom_tolerances={'V1': .1, 'R1': 0, 'R2': 0})
    assert r.worst_case_max == pytest.approx(-4.5)
    assert r.worst_case_min == pytest.approx(-5.5)
    assert 'not automatically a 3-sigma' in r.summary()


def test_failed_corner_is_not_reported_as_exact(monkeypatch):
    module = importlib.import_module('circuit_solver.engine.sensitivity')
    original = module.solve
    calls = 0
    def fail_corner(c):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError('corner failed')
        return original(c)
    monkeypatch.setattr(module, 'solve', fail_corner)
    with pytest.raises(RuntimeError, match='corner failed'):
        analyze_sensitivity(divider(), 'out')


def test_calculation_memory_references():
    from scripts.generate_calculation_memory import (
        EXERCISES_PART1, EXERCISES_PART2, EXERCISES_PART3, EXERCISES_PART4,
    )
    for ex in EXERCISES_PART1 + EXERCISES_PART2 + EXERCISES_PART3 + EXERCISES_PART4:
        for name, expected, actual, _ in ex['run']():
            assert math.isfinite(actual)
            assert abs(actual-expected) <= ex.get('atol', 1e-6) + ex.get('rtol', .005)*abs(expected), (ex['id'], name)


def test_report_does_not_approve_divergence_or_nan(monkeypatch, tmp_path, capsys):
    from scripts import generate_calculation_memory as report
    ex = dict(report.EXERCISES_PART1[0], python_code='print("<hello>")',
              run=lambda: [('wrong', 1., 2., 'V'), ('nan', 1., float('nan'), 'V'), ('zero', 0., .01, 'V')])
    monkeypatch.setattr(report, 'EXERCISES_PART1', [ex])
    for part in (2, 3, 4):
        monkeypatch.setattr(report, f'EXERCISES_PART{part}', [])
    # The generator resolves its destination relative to __file__.
    script = tmp_path / 'scripts' / 'generate.py'
    script.parent.mkdir()
    monkeypatch.setattr(report, '__file__', str(script))
    report.build_calculation_memory_html()
    html = (tmp_path / 'MEMORIA_DE_CALCULO.html').read_text()
    assert '✓ Validado' not in html
    assert html.count('⚠ Divergente') == 3
    assert '&lt;hello&gt;' in html
    assert '0/3' in capsys.readouterr().out


def test_tutorial_python_examples(monkeypatch, tmp_path):
    pytest.importorskip("pycircuit_core")
    import re
    from pathlib import Path
    import matplotlib.pyplot as plt
    tutorial = Path(__file__).resolve().parents[1] / 'ME_ENSINE_A_USAR.md'
    snippets = re.findall(r'```python\n(.*?)```', tutorial.read_text(), re.S)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda: None)
    scope = {}
    for index, code in enumerate(snippets, 1):
        exec(compile(code, f'tutorial snippet {index}', 'exec'), scope)
    assert scope['sol_tran'].V('2')[100] == pytest.approx(10 * (1 - math.exp(-1)), abs=.001)
    plt.close('all')
