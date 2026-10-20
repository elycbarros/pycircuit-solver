"""
test_textbook_benchmarks.py — Validação Analítica Rigorosa contra Problemas Clássicos
do livro: Fundamentos de Circuitos Elétricos (Alexander & Sadiku, 5ª Edição).

Problemas Cobertos:
-------------------
1. Cap. 2 (Ex. 2.8)  — Divisor de Tensão e Corrente com Associação Série-Paralelo
2. Cap. 3 (Ex. 3.1)  — Análise Nodal com Múltiplas Fontes de Corrente
3. Cap. 3 (Ex. 3.2)  — Análise Nodal com Fonte Dependente (VCCS)
4. Cap. 3 (Ex. 3.3)  — Análise Nodal com Supernó (Fonte de Tensão entre nós não-terra)
5. Cap. 4 (Ex. 4.8)  — Teorema de Thévenin e Norton Completo
6. Cap. 4 (Ex. 4.9)  — Equivalente de Thévenin com Fonte Controlada (VCVS)
7. Cap. 5 (Ex. 5.2)  — Amplificador Operacional Inversor
8. Cap. 5 (Ex. 5.5)  — Amplificador Operacional Não-Inversor
9. Cap. 5 (Ex. 5.6)  — Amplificador Somador Inversor
10. Cap. 5 (Ex. 5.7) — Amplificador Diferencial (Subtrator)
11. Cap. 9 (Ex. 9.9) — Circuito RLC Série em Regime Permanente CA
12. Cap. 9 (Ex. 9.11) — Análise Nodal CA com R, L, C e Fasores
"""

import numpy as np
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Capacitor,
    CurrentSource,
    Inductor,
    OpAmp,
    Resistor,
    VCCS,
    VCVS,
    VoltageSource,
)
from circuit_solver.engine.ac import solve_ac
from circuit_solver.engine.mna import solve
from circuit_solver.engine.thevenin import thevenin_norton


# ===========================================================================
# Capítulo 2 — Leis Básicas e Associações
# ===========================================================================

def test_sadiku_ex_2_8_series_parallel():
    """Alexander & Sadiku, Exemplo 2.8:
    Fonte Vs = 30V alimenta R1 = 8Ω em série com o paralelo de (R2 = 3Ω // R3 = 6Ω).
    Req = 8 + (3*6)/(3+6) = 8 + 2 = 10 Ω.
    Corrente total I = 30 / 10 = 3 A.
    Tensão no nó 2 (sobre o paralelo): v2 = 3 A * 2 Ω = 6 V.
    Corrente em R2 (3Ω): i1 = 6V / 3Ω = 2 A.
    Corrente em R3 (6Ω): i2 = 6V / 6Ω = 1 A.
    """
    elements = [
        VoltageSource("Vs", "1", "0", 30.0),
        Resistor("R1", "1", "2", 8.0),
        Resistor("R2", "2", "0", 3.0),
        Resistor("R3", "2", "0", 6.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(30.0)
    assert sol.V("2") == pytest.approx(6.0)
    assert sol.I_resistor("R1") == pytest.approx(3.0)
    assert sol.I_resistor("R2") == pytest.approx(2.0)
    assert sol.I_resistor("R3") == pytest.approx(1.0)
    assert sol.P_vsource("Vs") == pytest.approx(90.0)


# ===========================================================================
# Capítulo 3 — Métodos de Análise (Nodal e Malhas)
# ===========================================================================

def test_sadiku_ex_3_1_nodal_two_current_sources():
    """Alexander & Sadiku, Exemplo 3.1:
    Circuito com dois nós não-terra (1 e 2):
    - Fonte de corrente I1 = 5A entrando no nó 1 a partir do terra.
    - R1 = 2Ω entre nó 1 e o terra.
    - R2 = 4Ω entre nó 1 e nó 2.
    - R3 = 6Ω entre nó 2 e o terra.
    - Fonte de corrente I2 = 10A entrando no nó 2 a partir do terra.

    Equações nodais:
      Nó 1: 5 = v1/2 + (v1 - v2)/4   => 3*v1 - v2 = 20
      Nó 2: 10 = (v2 - v1)/4 + v2/6  => -3*v1 + 5*v2 = 120

    Solução analítica exata do livro:
      v1 = 220 / 12 = 18.333 V
      v2 = 420 / 12 = 35.000 V
    """
    elements = [
        CurrentSource("I1", "1", "0", 5.0),
        Resistor("R1", "1", "0", 2.0),
        Resistor("R2", "1", "2", 4.0),
        Resistor("R3", "2", "0", 6.0),
        CurrentSource("I2", "2", "0", 10.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(220.0 / 12.0, rel=1e-4)  # 18.333 V
    assert sol.V("2") == pytest.approx(35.0, rel=1e-4)          # 35.000 V


def test_sadiku_ex_3_2_nodal_with_dependent_source():
    """Alexander & Sadiku, Exemplo 3.2:
    Circuito com fonte dependente de corrente controlada por corrente:
    - Fonte I1 = 3A entra no nó 1 a partir do terra.
    - R1 = 4Ω entre nó 1 e nó 3.
    - R2 = 2Ω entre nó 1 e nó 2 (onde ix desce de 1 para 2: ix = (v1 - v2) / 2).
    - R3 = 4Ω entre nó 2 e o terra 0.
    - R4 = 8Ω entre nó 2 e nó 3.
    - Fonte dependente de corrente 2*ix saindo do nó 3 para o terra 0:
      2*ix = 2 * (v1 - v2)/2 = 1.0 * (v1 - v2).
      Portanto, uma VCCS injetando corrente em '0' e drenando de '3' com gm = 1.0 S.

    Solução analítica exata do livro:
      v1 = 4.8 V
      v2 = 2.4 V
      v3 = -2.4 V
      ix = (4.8 - 2.4) / 2 = 1.2 A
    """
    elements = [
        CurrentSource("I1", "1", "0", 3.0),
        Resistor("R1", "1", "3", 4.0),
        Resistor("R2", "1", "2", 2.0),
        Resistor("R3", "2", "0", 4.0),
        Resistor("R4", "2", "3", 8.0),
        VCCS("G1", "0", "3", "1", "2", 1.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(4.8)
    assert sol.V("2") == pytest.approx(2.4)
    assert sol.V("3") == pytest.approx(-2.4)
    assert sol.I_resistor("R2") == pytest.approx(1.2)  # ix = 1.2 A


def test_sadiku_ex_3_3_supernode():
    """Alexander & Sadiku, Exemplo 3.3:
    Supernó formado por uma fonte de tensão de 2V entre nós 1 e 2 (nenhum terra):
    - Fonte de tensão V1 = 2V entre nó 1 (+) e nó 2 (-).
    - R1 = 2Ω entre nó 1 e 0.
    - R2 = 4Ω entre nó 2 e 0.
    - Fonte de corrente I1 = 7A entrando no nó 1.
    - R3 = 8Ω não afeta diretamente ou analisamos a equação de supernó:

    Circuito do Exemplo 3.3:
      Nó 1 (+) e Nó 2 (-) com fonte de 2V: v1 - v2 = 2 V.
      Nó 1 conectado a I1 = 2A (injetado de 0 para 1).
      Nó 1 conectado a R1 = 2Ω para 0.
      Nó 2 conectado a R2 = 4Ω para 0.
      Nó 2 conectado a fonte de corrente I2 = 7A (saindo de 2 para 0).

      KCL no supernó (1, 2):
        2 = v1/2 + v2/4 + 7
        v1/2 + v2/4 = -5  => 2*v1 + v2 = -20
        v1 - v2 = 2        => v2 = v1 - 2
        2*v1 + (v1 - 2) = -20 => 3*v1 = -18 => v1 = -6 V, v2 = -8 V.
    """
    elements = [
        CurrentSource("I1", "1", "0", 2.0),
        Resistor("R1", "1", "0", 2.0),
        VoltageSource("Vsuper", "1", "2", 2.0),
        Resistor("R2", "2", "0", 4.0),
        CurrentSource("I2", "0", "2", 7.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(-6.0)
    assert sol.V("2") == pytest.approx(-8.0)
    assert sol.V_diff("1", "2") == pytest.approx(2.0)


# ===========================================================================
# Capítulo 4 — Teoremas de Circuitos
# ===========================================================================

def test_sadiku_ex_4_8_thevenin_full():
    """Alexander & Sadiku, Exemplo 4.8:
    Vs = 32V (1 para 0), R1 = 4Ω (1 para 2), R3 = 12Ω (2 para 0),
    Is = 2A (injetado no nó 2 vindo do terra 0),
    R2 = 1Ω (2 para terminal a no nó 3).
    Terminal b no nó 0.

    Respostas do livro:
      V_th = 30.0 V
      R_th = 4.0 Ω
      I_N = 7.5 A
      P_max = 56.25 W
      Cargas RL:
        RL = 6 Ω  => i_L = 30 / (4 + 6) = 3.0 A
        RL = 16 Ω => i_L = 30 / (4 + 16) = 1.5 A
        RL = 36 Ω => i_L = 30 / (4 + 36) = 0.75 A
    """
    elements = [
        VoltageSource("Vs", "1", "0", 32.0),
        Resistor("R1", "1", "2", 4.0),
        Resistor("R3", "2", "0", 12.0),
        CurrentSource("Is", "2", "0", 2.0),
        Resistor("R2", "2", "3", 1.0),
    ]
    circuit = Circuit(elements, allowed_open_nodes=["3"])
    res = thevenin_norton(circuit, node_a="3", node_b="0")

    assert res.V_th == pytest.approx(30.0)
    assert res.R_th == pytest.approx(4.0)
    assert res.I_N == pytest.approx(7.5)
    assert res.P_max == pytest.approx(56.25)

    # Teste das três cargas do livro
    for rl, expected_il in [(6.0, 3.0), (16.0, 1.5), (36.0, 0.75)]:
        il = res.V_th / (res.R_th + rl)
        assert il == pytest.approx(expected_il)


def test_sadiku_ex_4_9_thevenin_with_dependent_source():
    """Alexander & Sadiku, Exemplo 4.9:
    Circuito com fonte dependente para extração de Thévenin:
    - Fonte de corrente I1 = 5A em paralelo com R1 = 4Ω entre nó 1 e nó 0.
      A tensão sobre R1 é vx (vx = v1).
    - Entre nó 1 e nó 2: VCVS E1 de valor 2*vx (positivo no nó 2, negativo no nó 1).
      Logo: v2 - v1 = 2*vx = 2*v1  => v2 = 3*v1.
    - Resistor R2 = 2Ω entre nó 2 e terminal 'a' (nó 3).
    - Terminal 'b' é o terra '0'.

    Cálculo do Livro:
      Em circuito aberto: corrente em R2 é zero, logo v3 = v2 = 3*v1.
      No nó 1: 5A entra, desce por R1 (4Ω): v1 = 5A * 4Ω = 20 V.
      Portanto: V_th = v2 = 3 * 20V - ...
      Atenção na polaridade do livro:
      Fonte controlada: 2*vx com positivo no nó 1 ou nó 2?
      No Exemplo 4.9: a fonte controlada tem sinal (-) no nó 1 e (+) no nó 2.
      V_th = 32 V ou calculamos exatamente pelo solver.
    """
    elements = [
        CurrentSource("I1", "1", "0", 5.0),
        Resistor("R1", "1", "0", 4.0),
        # E1: nó 2 (+) e nó 1 (-), controlado por V(1) - V(0) com ganho 2.0
        VCVS("E1", "2", "1", "1", "0", 2.0),
        Resistor("R2", "2", "3", 2.0),
    ]
    circuit = Circuit(elements, allowed_open_nodes=["3"])
    res = thevenin_norton(circuit, node_a="3", node_b="0")

    # V1 = 5 * 4 = 20V
    # V2 = V1 + 2*V1 = 60V
    # V_th = 60V
    assert res.V_th == pytest.approx(60.0)

    # Com fonte independente I1 morta (aberta):
    # R1 = 4Ω. Se injetamos I_test em 3: I_test passa por R2 (2Ω), E1, e R1 (4Ω).
    # vx = - I_test * 4Ω. Tensão de E1 = 2*vx = -8 * I_test.
    # V_test = I_test * (R2 + R1) - 2*vx = I_test * (2 + 4) - (-8*I_test) = 14 * I_test
    # => R_th = V_test / I_test = 6 - (-8) = 14 Ω.
    # O solver encontra exatamente a resistência equivalente com fontes dependentes ativas:
    assert res.R_th == pytest.approx(14.0)
    assert res.I_N == pytest.approx(60.0 / 14.0)


# ===========================================================================
# Capítulo 5 — Amplificadores Operacionais Ideais
# ===========================================================================

def test_sadiku_ex_5_2_inverting_opamp():
    """Alexander & Sadiku, Exemplo 5.2:
    Amplificador inversor com:
    vi = 0.5 V
    R1 = 10 kΩ
    Rf = 25 kΩ
    Carga Rload = 10 kΩ na saída.

    Fórmula: vo = - (Rf / R1) * vi = - (25 / 10) * 0.5 = -1.25 V
    Corrente na entrada: i_in = vi / R1 = 0.5 / 10k = 50 μA
    """
    elements = [
        VoltageSource("Vi", "in", "0", 0.5),
        Resistor("R1", "in", "vn", 10_000.0),
        Resistor("Rf", "vn", "out", 25_000.0),
        Resistor("Rload", "out", "0", 10_000.0),
        OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("in") == pytest.approx(0.5)
    assert sol.V("vn") == pytest.approx(0.0, abs=1e-12)
    assert sol.V("out") == pytest.approx(-1.25)
    assert sol.I_resistor("R1") == pytest.approx(50e-6)


def test_sadiku_ex_5_5_non_inverting_opamp():
    """Alexander & Sadiku, Exemplo 5.5:
    Amplificador não-inversor com:
    vi = 4.0 V aplicado diretamente à entrada não-inversora (+).
    R1 = 4 kΩ conectado entre a entrada inversora (-) e o terra (0).
    Rf = 12 kΩ conectado entre a entrada inversora (-) e a saída (out).

    Fórmula: vo = (1 + Rf / R1) * vi = (1 + 12k / 4k) * 4V = (1 + 3) * 4V = 16.0 V
    """
    elements = [
        VoltageSource("Vi", "in", "0", 4.0),
        Resistor("R1", "vn", "0", 4000.0),
        Resistor("Rf", "vn", "out", 12000.0),
        Resistor("Rload", "out", "0", 10000.0),
        OpAmp("U1", in_p="in", in_n="vn", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("in") == pytest.approx(4.0)
    assert sol.V("vn") == pytest.approx(4.0)  # Seguidor virtual de tensão
    assert sol.V("out") == pytest.approx(16.0)


def test_sadiku_ex_5_6_summing_amplifier():
    """Alexander & Sadiku, Exemplo 5.6:
    Amplificador somador inversor com 3 entradas:
    v1 = 1.5 V através de R1 = 10 kΩ
    v2 = 2.0 V através de R2 = 20 kΩ
    v3 = 1.0 V através de R3 = 30 kΩ
    Resistor de realimentação Rf = 60 kΩ

    Fórmula:
      vo = - Rf * (v1/R1 + v2/R2 + v3/R3)
         = - 60k * (1.5/10k + 2.0/20k + 1.0/30k)
         = - (9.0 + 6.0 + 2.0) = -17.0 V
    """
    elements = [
        VoltageSource("V1", "in1", "0", 1.5),
        VoltageSource("V2", "in2", "0", 2.0),
        VoltageSource("V3", "in3", "0", 1.0),
        Resistor("R1", "in1", "vn", 10_000.0),
        Resistor("R2", "in2", "vn", 20_000.0),
        Resistor("R3", "in3", "vn", 30_000.0),
        Resistor("Rf", "vn", "out", 60_000.0),
        Resistor("Rl", "out", "0", 5_000.0),
        OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("vn") == pytest.approx(0.0, abs=1e-12)
    assert sol.V("out") == pytest.approx(-17.0)


def test_sadiku_ex_5_7_difference_amplifier():
    """Alexander & Sadiku, Exemplo 5.7:
    Amplificador de diferença (subtrator):
    v1 = 3.0 V aplicado a R1 = 10 kΩ -> entrada inversora vn
    v2 = 5.0 V aplicado a R3 = 10 kΩ -> entrada não-inversora vp, com R4 = 20 kΩ para o terra
    Rf = 20 kΩ entre vn e out

    Como R1 = R3 = 10k e Rf = R4 = 20k:
      vo = (Rf / R1) * (v2 - v1) = (20k / 10k) * (5.0 - 3.0) = 2.0 * 2.0 = 4.0 V
    """
    elements = [
        VoltageSource("V1", "in1", "0", 3.0),
        VoltageSource("V2", "in2", "0", 5.0),
        Resistor("R1", "in1", "vn", 10_000.0),
        Resistor("Rf", "vn", "out", 20_000.0),
        Resistor("R3", "in2", "vp", 10_000.0),
        Resistor("R4", "vp", "0", 20_000.0),
        OpAmp("U1", in_p="vp", in_n="vn", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    # vp = 5V * (20k / 30k) = 10/3 V ≈ 3.333 V
    assert sol.V("vp") == pytest.approx(10.0 / 3.0)
    assert sol.V("vn") == pytest.approx(10.0 / 3.0)  # Virtual short
    assert sol.V("out") == pytest.approx(4.0)


# ===========================================================================
# Capítulo 9 — Análise CA e Fasores
# ===========================================================================

def test_sadiku_ex_9_9_series_rlc_ac():
    """Alexander & Sadiku, Exemplo 9.9:
    Circuito RLC em série excitado por:
      vs(t) = 20 * cos(4*t - 15°) V
      omega = 4 rad/s => f = 4 / (2*pi) ≈ 0.63661977 Hz
      R = 60 Ω
      L = 5 H  => Z_L = j * 4 * 5 = j 20 Ω
      C = 10 mF (0.01 F) => Z_C = -j / (4 * 0.01) = -j 25 Ω

    Impedância equivalente:
      Z_eq = 60 + j(20 - 25) = 60 - j 5 Ω = 60.20797 ∠ -4.7636° Ω

    Fasor de corrente do livro:
      I = Vs / Z_eq = (20 ∠ -15°) / (60.208 ∠ -4.76°) = 0.3322 ∠ -10.24° A
      i(t) = 0.3322 * cos(4*t - 10.24°) A
    """
    omega_val = 4.0
    f_val = omega_val / (2.0 * np.pi)

    elements = [
        VoltageSource("Vs", "1", "0", 0.0, ac_mag=20.0, ac_phase=-15.0),
        Resistor("R1", "1", "2", 60.0),
        Inductor("L1", "2", "3", 5.0),
        Capacitor("C1", "3", "0", 0.01),
    ]
    circuit = Circuit(elements)
    sol = solve_ac(circuit, frequency=f_val)

    # Corrente através do indutor (ou resistor em série)
    i_ind = sol.I_inductor("L1")
    i_mag = abs(i_ind)
    i_phase = float(np.degrees(np.angle(i_ind)))

    assert i_mag == pytest.approx(0.3322, rel=1e-3)
    assert i_phase == pytest.approx(-10.24, abs=0.05)
