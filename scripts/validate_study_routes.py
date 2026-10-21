"""Five study routes checked against independent equations, with a saved report.

Run from the project root: .venv/bin/python scripts/validate_study_routes.py
"""
from itertools import product
from pathlib import Path
import math
import platform

import scipy

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import trapezoid
from scipy.linalg import expm

from circuit_solver import (
    Circuit, VoltageSource, Resistor, Capacitor, Inductor, Transformer,
    solve, solve_ac, solve_transient, analyze_sensitivity,
    build_three_phase_source, analyze_two_wattmeters,
)
from circuit_solver.engine.monte_carlo_rust import monte_carlo, RUST_AVAILABLE


def check_divider(solver):
    c = Circuit([VoltageSource('V', 'in', '0', 12), Resistor('R1', 'in', 'out', 4),
                 Resistor('R2', 'out', '0', 6)])
    r = solve(c, solver=solver)
    np.testing.assert_allclose([r.V('out'), r.I_resistor('R1'), r.I_vsource('V'),
                                r.P_resistor('R1'), r.P_resistor('R2'), r.P_vsource('V')],
                               [7.2, 1.2, -1.2, 5.76, 8.64, 14.4], atol=1e-12, rtol=1e-12)
    audit = r.power_audit()
    assert abs(audit['_total']) < 1e-12
    assert audit['R1'] == -r.P_resistor('R1')
    return {'Vout (V)': r.V('out'), 'I (A)': r.I_resistor('R1'),
            'P fornecida (W)': r.P_vsource('V'), 'resíduo potência (W)': audit['_total']}


def check_first_order(kind, method, backend):
    # RC: vC=10(1-exp(-t/tau)); RL: iL=(10/R)(1-exp(-t/tau)).
    resistance = 1000. if kind == 'RC' else 10.
    reactive = Capacitor('C', 'out', '0', 1e-6) if kind == 'RC' else Inductor('L', 'out', '0', .01)
    c = Circuit([VoltageSource('V', 'in', '0', 10), Resistor('R', 'in', 'out', resistance), reactive])
    tau = .001
    errors = []
    for dt in (tau/20, tau/40, tau/80):
        r = solve_transient(c, t_stop=3*tau, dt=dt, method=method, uic=True, backend=backend)
        observed = r.V('out') if kind == 'RC' else r.I('L')
        exact = (10 if kind == 'RC' else 10/resistance) * (1-np.exp(-r.time/tau))
        errors.append(float(np.max(np.abs(observed-exact))))
    order = math.log(errors[0]/errors[-1])/math.log(4)
    assert (1.9 < order < 2.1) if method == 'trapezoidal' else (.9 < order < 1.1)
    assert errors[-1] < errors[1] < errors[0]
    supplied = trapezoid(-10*r.I('V'), r.time)
    dissipated = trapezoid(r.P_dissipated(), r.time)
    energy_residual = supplied - dissipated - (r.E_stored()[-1]-r.E_stored()[0])
    # Backward Euler dissipates numerical energy; it is not energy preserving.
    assert abs(energy_residual) / supplied < (.0001 if method == 'trapezoidal' else .01)
    return {'ordem observada': order, 'erro dt=tau/80 (V ou A)': errors[-1],
            'resíduo energia (J)': energy_residual}


def check_transformer(k, backend):
    # Independent two-winding frequency-domain reference: eliminate secondary current.
    omega = 2*math.pi*60
    l1, l2, load = 10., 2.5, 100.
    mutual = k*math.sqrt(l1*l2)
    current1 = 120/(1j*omega*l1 + omega**2*mutual**2/(load+1j*omega*l2))
    current2 = -1j*omega*mutual*current1/(load+1j*omega*l2)
    expected_v2 = -load*current2
    c_ac = Circuit([VoltageSource('V', 'p', '0', 0, ac_mag=120*math.sqrt(2)),
                    Transformer('T', 'p', '0', 's', '0', L1=l1, L2=l2, coupling=k),
                    Resistor('R', 's', '0', load)])
    ac = solve_ac(c_ac, frequency=60)
    np.testing.assert_allclose(ac.V('s')/math.sqrt(2), expected_v2, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(ac.complex_power('R'), abs(expected_v2)**2/load, rtol=1e-11, atol=1e-11)

    # Independent transient reference: i(t)=iss-exp(-L^-1 R t) iss.
    L = np.array([[.02, k*.01], [k*.01, .005]])
    c = Circuit([VoltageSource('V', 'in', '0', 10), Resistor('R1', 'in', 'p', 10),
                 Transformer('T', 'p', '0', 's', '0', L1=.02, L2=.005, coupling=k),
                 Resistor('R2', 's', '0', 20)])
    r = solve_transient(c, t_stop=.003, dt=1e-6, uic=True, backend=backend)
    currents = np.column_stack([r.I('T_p'), r.I('T_s')])
    steady = np.array([1., 0.])
    exact_final = steady-expm(-np.linalg.solve(L, np.diag([10., 20.]))*r.time[-1])@steady
    np.testing.assert_allclose(currents[-1], exact_final, atol=2e-8)
    energy = .5*np.einsum('ti,ij,tj->t', currents, L, currents)
    np.testing.assert_allclose(r.E_stored(), energy, rtol=1e-11, atol=1e-14)
    assert np.sign(r.V('s')[1]) == np.sign(k)
    balance = trapezoid(-10*r.I('V')-r.P_dissipated(), r.time) - (energy[-1]-energy[0])
    assert abs(balance) < 5e-8
    return {'Vsec RMS (V)': abs(ac.V('s'))/math.sqrt(2),
            'fase Vsec (graus)': float(np.angle(ac.V('s'), deg=True)),
            'E final (J)': energy[-1], 'resíduo energia (J)': balance}


def check_three_phase(sequence):
    # Unequal series R-L impedances, floating neutral: vn=sum(V/Z)/sum(1/Z).
    z = np.array([10+2j, 30+5j, 20+15j])
    sign = -1 if sequence == 'ABC' else 1
    voltage = 120*np.exp(sign*2j*np.pi*np.arange(3)/3)
    neutral = np.sum(voltage/z)/np.sum(1/z)
    current = (voltage-neutral)/z
    expected_power = np.sum((voltage-neutral)*np.conj(current))
    omega = 2*math.pi*60
    elements = build_three_phase_source(V_rms=120, sequence=sequence, node_neutral='0')
    for phase, impedance in zip('abc', z):
        elements.extend([Resistor(f'R{phase}', phase, f'x{phase}', impedance.real),
                         Inductor(f'L{phase}', f'x{phase}', 'n', impedance.imag/omega)])
    r = solve_ac(Circuit(elements), frequency=60)
    assert abs(r.V_rms('a')-120) < 1e-10
    measured_current = np.array([-r.I_vsource(f'Vs_{p}')/math.sqrt(2) for p in 'abc'])
    np.testing.assert_allclose(measured_current, current, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(r.V('n')/math.sqrt(2), neutral, atol=1e-10)
    rms_voltage = np.array([r.V(p)/math.sqrt(2) for p in 'abc'])
    wm = analyze_two_wattmeters(rms_voltage[0]-rms_voltage[2], rms_voltage[1]-rms_voltage[2],
                                measured_current[0], measured_current[1])
    np.testing.assert_allclose(wm.P_total+1j*wm.Q_total, expected_power, rtol=1e-11, atol=1e-9)
    component_power = sum(r.complex_power(f'{kind}{phase}') for kind in 'RL' for phase in 'abc')
    np.testing.assert_allclose(component_power, expected_power, rtol=1e-11, atol=1e-9)
    return {'P (W)': wm.P_total, 'Q (var)': wm.Q_total, 'FP': wm.power_factor,
            'corrente neutro (A)': abs(np.sum(measured_current))}


def check_tolerances(backend):
    # Independent uniform resistor tolerances, fixed source. Gauss-Legendre
    # quadrature provides moments of the exact divider, without a circuit solve.
    c = Circuit([VoltageSource('V', 'in', '0', 10),
                 Resistor('R1', 'in', 'out', 1000, tolerance=.05),
                 Resistor('R2', 'out', '0', 1000, tolerance=.05)])
    sensitivity = analyze_sensitivity(c, 'out', custom_tolerances={'V': 0})
    np.testing.assert_allclose([sensitivity.worst_case_min, sensitivity.worst_case_max], [4.75, 5.25])
    linear_rss = math.sqrt(2)*.125
    assert abs(sensitivity.rss_max-5-linear_rss) < 1e-12
    sigma_linear = linear_rss/math.sqrt(3)
    x, w = leggauss(32)
    resistors = 1000*(1+.05*x)
    values = 10*resistors[None, :]/(resistors[:, None]+resistors[None, :])
    weights = w[:, None]*w[None, :]/4
    mean = float(np.sum(weights*values))
    variance = float(np.sum(weights*(values-mean)**2))
    fourth_moment = float(np.sum(weights*(values-mean)**4))
    n = 8000
    mc = monte_carlo(c, n_samples=n, seed=42, force_python=(backend == 'python'))
    idx = mc.node_labels.index('out')
    assert mc.n_failed == 0 and mc.samples.shape[0] == n
    assert mc.min[idx] >= 4.75 and mc.max[idx] <= 5.25
    # Statistical checks use five standard errors, not a guarantee on manufacturing.
    assert abs(mc.mean[idx]-mean) < 5*math.sqrt(variance/n)
    variance_se = math.sqrt((fourth_moment-variance**2)/n)
    assert abs(mc.std[idx]**2-variance) < 5*variance_se
    assert abs(sigma_linear-math.sqrt(variance))/math.sqrt(variance) < .002
    if backend == 'python':
        draws = np.random.default_rng(42).uniform(-.05, .05, size=(n, 2))
        exact_samples = 10*(1+draws[:, 1])/(2+draws[:, 0]+draws[:, 1])
        np.testing.assert_allclose(mc.samples[:, idx], exact_samples, rtol=1e-12, atol=1e-12)
    return {'RSS linear (V)': linear_rss, 'sigma linear (V)': sigma_linear,
            'sigma quadratura (V)': math.sqrt(variance), 'sigma Monte Carlo (V)': mc.std[idx],
            'média Monte Carlo (V)': mc.mean[idx]}


def run_all():
    if not RUST_AVAILABLE:
        raise RuntimeError('Compile Rust before running all study routes; no silent fallback.')
    results = []
    for solver in ('dense', 'sparse', 'rust_sparse'):
        results.append(('Divisor', solver, check_divider(solver)))
    for kind, method, backend in product(('RC', 'RL'), ('euler', 'trapezoidal'), ('python', 'rust')):
        results.append((kind, f'{method}/{backend}', check_first_order(kind, method, backend)))
    for k, backend in product((.5, -.5), ('python', 'rust')):
        results.append(('Transformador', f'k={k}/{backend}', check_transformer(k, backend)))
    for sequence in ('ABC', 'ACB'):
        results.append(('Trifásico', sequence, check_three_phase(sequence)))
    for backend in ('python', 'rust'):
        results.append(('Tolerâncias', backend, check_tolerances(backend)))
    return results


def write_report(results):
    rows = ['# Validação dos cinco roteiros de estudo', '',
            f'Executado com CPython {platform.python_version()}, NumPy {np.__version__} e SciPy {scipy.__version__}, com Rust disponível.', '',
            'Todas as verificações abaixo passaram. O relatório só é escrito após todas as verificações.', '',
            '| Roteiro | Configuração | Valores observados |', '|---|---|---|']
    for name, configuration, metrics in results:
        values = '; '.join(f'{label}: {value:.9g}' for label, value in metrics.items())
        rows.append(f'| {name} | {configuration} | {values} |')
    rows.extend(['', '## Referências e hipóteses', '',
        '- Divisor: Vout=12·6/(4+6)=7,2 V; I=1,2 A; P fornecida=14,4 W. '
        '`P_resistor` retorna dissipação positiva; `power_audit` apresenta absorção negativa.',
        '- RC/RL: resposta exponencial com tau=1 ms; passos tau/20, tau/40, tau/80; '
        'ordem esperada 1 em Euler e 2 no trapézio. O resíduo de energia inclui a integração numérica; Euler tem dissipação numérica.',
        '- Transformador: referência AC por eliminação da corrente secundária; transitório '
        'por exponencial matricial da matriz de indutâncias. Energia inclui M·i1·i2.',
        '- Trifásico: cargas desiguais e neutro flutuante; vn=sum(V/Z)/sum(1/Z). '
        'A fonte recebe RMS e gera amplitude de pico. Os wattímetros recebem fasores RMS; '
        'a soma das potências dos componentes usa os fasores de pico do solver e o fator 1/2.',
        '- Tolerâncias: dois resistores uniformes independentes ±5%, fonte fixa. '
        'Cantos exatos deste divisor: [4,75; 5,25] V. RSS não é automaticamente 3σ. '
        'A quadratura calcula os momentos do divisor exato; Monte Carlo usa 8.000 amostras e semente 42 '
        'em cada backend. Os fluxos aleatórios Python/Rust diferem; a comparação é estatística.',
        '', 'Reproduzir: `.venv/bin/python scripts/validate_study_routes.py`.', ''])
    output = Path(__file__).resolve().parents[1]/'RELATORIO_ROTEIROS_ESTUDO.md'
    output.write_text('\n'.join(rows), encoding='utf-8')
    return output


if __name__ == '__main__':
    checked = run_all()
    print(f'{len(checked)} configurações aprovadas nos cinco roteiros.')
    print(write_report(checked))
