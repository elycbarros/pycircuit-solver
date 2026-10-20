"""
test_textbook_benchmarks.py — Validação Analítica Rigorosa Multi-Livros

Validação comparativa contra soluções analíticas e problemas canônicos dos principais
manuais mundiais de Engenharia Elétrica:
1. Alexander & Sadiku — Fundamentos de Circuitos Elétricos (5ª Edição)
2. Nilsson & Riedel — Circuitos Elétricos (8ª/10ª Edição)
3. Boylestad — Introdução à Análise de Circuitos (12ª/13ª Edição)
4. Dorf & Svoboda — Introdução aos Circuitos Elétricos (8ª/9ª Edição)

Problemas Cobertos:
-------------------
[Alexander & Sadiku]
1. Cap. 2 (Ex. 2.8)  — Divisor de Tensão e Corrente com Associação Série-Paralelo
2. Cap. 2 (Ex. 2.15) — Transformação Delta-Estrela (Δ-Y) em Circuito em Ponte
3. Cap. 3 (Ex. 3.1)  — Análise Nodal com Múltiplas Fontes de Corrente
4. Cap. 3 (Ex. 3.2)  — Análise Nodal com Fonte Dependente (VCCS)
5. Cap. 3 (Ex. 3.3)  — Análise Nodal com Supernó (Fonte de Tensão entre nós não-terra)
6. Cap. 4 (Ex. 4.8)  — Teorema de Thévenin, Norton e Máxima Transferência de Potência
7. Cap. 4 (Ex. 4.9)  — Equivalente de Thévenin com Fonte Controlada (VCVS)
8. Cap. 5 (Ex. 5.2)  — Amplificador Operacional Inversor
9. Cap. 5 (Ex. 5.5)  — Amplificador Operacional Não-Inversor
10. Cap. 5 (Ex. 5.6) — Amplificador Somador Inversor
11. Cap. 5 (Ex. 5.7) — Amplificador Diferencial (Subtrator)
12. Cap. 9 (Ex. 9.9) — Circuito RLC Série em Regime Permanente CA

[Nilsson & Riedel]
13. Cap. 4 (Ex. 4.10) — Equivalente de Thévenin com Fonte Controlada
14. Cap. 5 (Ex. 5.3)  — Amplificador Somador Inversor com Pesos Arbitrários
15. Cap. 9 (Ex. 9.6)  — Análise Fasorial de Circuito RLC Paralelo em Alta Frequência

[Boylestad]
16. Cap. 9 — Teorema de Millman para Múltiplas Baterias com Resistência Interna
17. Cap. 8 — Rede Escada R-2R de 4 Bits para Conversor Digital-Analógico (D/A)
18. Cap. 6 — Ponte de Wheatstone em Equilíbrio (Condição de Nulidade do Detector)

[Dorf & Svoboda]
19. Cap. 6 — Amplificador Inversor com Rede em T na Realimentação (Alto Ganho)
20. Cap. 6 — Buffer Seguidor de Tensão e Isolamento de Carga com Alta Impedância
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
# 1. ALEXANDER & SADIKU (5ª Edição)
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


def test_sadiku_ex_2_15_delta_wye_bridge():
    """Alexander & Sadiku, Exemplo 2.15:
    Circuito em ponte com transformação Delta-Estrela (Δ-Y):
    Terminais da ponte: nó 'a' (positivo) e terra '0' (nó 'b').
    Vs = 120V aplicado aos terminais a e 0.
    R1 = 10Ω (a para c), R2 = 20Ω (a para d)
    R3 = 12.5Ω (ponte entre c e d)
    R4 = 15Ω (c para 0), R5 = 30Ω (d para 0)

    Solução analítica do livro:
      Transformando o delta superior (R1, R2, R3) em estrela:
        R_total = 10 + 20 + 12.5 = 42.5 Ω
        R_an = 10*20 / 42.5 = 4.70588 Ω
        R_cn = 10*12.5 / 42.5 = 2.94118 Ω
        R_dn = 20*12.5 / 42.5 = 5.88235 Ω
      Ramo esquerdo: R_cn + 15 = 17.94118 Ω
      Ramo direito:  R_dn + 30 = 35.88235 Ω (exatamente 2x o ramo esquerdo)
      Paralelo: (17.94118 * 35.88235) / (17.94118 + 35.88235) = 11.96078 Ω
      Req_total = 4.70588 + 11.96078 = 16.66667 Ω = 50 / 3 Ω
      Corrente total fornecida: I_total = 120V / (50/3) = 7.200 A.
    """
    elements = [
        VoltageSource("Vs", "a", "0", 120.0),
        Resistor("R1", "a", "c", 10.0),
        Resistor("R2", "a", "d", 20.0),
        Resistor("R3", "c", "d", 12.5),
        Resistor("R4", "c", "0", 15.0),
        Resistor("R5", "d", "0", 30.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    i_total = -sol.I_vsource("Vs")
    req = 120.0 / i_total

    assert i_total == pytest.approx(7.200, rel=1e-5)
    assert req == pytest.approx(50.0 / 3.0, rel=1e-5)
    audit = sol.power_audit()
    assert audit["_total"] == pytest.approx(0.0, abs=1e-6)


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
    - Fonte de corrente I1 = 2A entrando no nó 1.
    - Fonte de corrente I2 = 7A saindo de 2 para 0.

    KCL no supernó (1, 2):
      2 = v1/2 + v2/4 + 7  => 2*v1 + v2 = -20
      v1 - v2 = 2          => v2 = v1 - 2
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


def test_sadiku_ex_4_8_thevenin_full():
    """Alexander & Sadiku, Exemplo 4.8:
    Vs = 32V (1 para 0), R1 = 4Ω (1 para 2), R3 = 12Ω (2 para 0),
    Is = 2A (injetado no nó 2 vindo do terra 0),
    R2 = 1Ω (2 para terminal a no nó 3). Terminal b no nó 0.

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

    for rl, expected_il in [(6.0, 3.0), (16.0, 1.5), (36.0, 0.75)]:
        il = res.V_th / (res.R_th + rl)
        assert il == pytest.approx(expected_il)


def test_sadiku_ex_4_9_thevenin_with_dependent_source():
    """Alexander & Sadiku, Exemplo 4.9:
    Circuito com fonte dependente para extração de Thévenin:
    - Fonte de corrente I1 = 5A em paralelo com R1 = 4Ω entre nó 1 e nó 0.
      A tensão sobre R1 é vx (vx = v1).
    - Entre nó 1 e nó 2: VCVS E1 de valor 2*vx (positivo no nó 2, negativo no nó 1).
    - Resistor R2 = 2Ω entre nó 2 e terminal 'a' (nó 3).
    - Terminal 'b' é o terra '0'.

    Solução analítica:
      V_th = 60.0 V
      R_th = 14.0 Ω
      I_N = 60 / 14 A ≈ 4.2857 A
    """
    elements = [
        CurrentSource("I1", "1", "0", 5.0),
        Resistor("R1", "1", "0", 4.0),
        VCVS("E1", "2", "1", "1", "0", 2.0),
        Resistor("R2", "2", "3", 2.0),
    ]
    circuit = Circuit(elements, allowed_open_nodes=["3"])
    res = thevenin_norton(circuit, node_a="3", node_b="0")

    assert res.V_th == pytest.approx(60.0)
    assert res.R_th == pytest.approx(14.0)
    assert res.I_N == pytest.approx(60.0 / 14.0)


def test_sadiku_ex_5_2_inverting_opamp():
    """Alexander & Sadiku, Exemplo 5.2:
    Amplificador inversor com: vi = 0.5 V, R1 = 10 kΩ, Rf = 25 kΩ, Rload = 10 kΩ.
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
    Amplificador não-inversor com: vi = 4.0 V, R1 = 4 kΩ, Rf = 12 kΩ.
    Fórmula: vo = (1 + Rf / R1) * vi = (1 + 12k / 4k) * 4V = 16.0 V
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
    assert sol.V("vn") == pytest.approx(4.0)
    assert sol.V("out") == pytest.approx(16.0)


def test_sadiku_ex_5_6_summing_amplifier():
    """Alexander & Sadiku, Exemplo 5.6:
    Amplificador somador inversor com 3 entradas:
    v1 = 1.5 V (R1 = 10k), v2 = 2.0 V (R2 = 20k), v3 = 1.0 V (R3 = 30k), Rf = 60 kΩ.
    Fórmula: vo = - 60k * (1.5/10k + 2.0/20k + 1.0/30k) = -17.0 V
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
    v1 = 3.0 V (R1 = 10k -> vn, Rf = 20k -> out)
    v2 = 5.0 V (R3 = 10k -> vp, R4 = 20k -> 0)
    Como R1 = R3 = 10k e Rf = R4 = 20k:
      vo = (Rf / R1) * (v2 - v1) = 2.0 * (5.0 - 3.0) = 4.0 V
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

    assert sol.V("vp") == pytest.approx(10.0 / 3.0)
    assert sol.V("vn") == pytest.approx(10.0 / 3.0)
    assert sol.V("out") == pytest.approx(4.0)


def test_sadiku_ex_9_9_series_rlc_ac():
    """Alexander & Sadiku, Exemplo 9.9:
    Circuito RLC em série excitado por: vs(t) = 20 * cos(4*t - 15°) V
      omega = 4 rad/s, R = 60 Ω, L = 5 H, C = 10 mF
    Fasor de corrente analítico do livro:
      I = 0.3322 ∠ -10.24° A
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

    i_ind = sol.I_inductor("L1")
    i_mag = abs(i_ind)
    i_phase = float(np.degrees(np.angle(i_ind)))

    assert i_mag == pytest.approx(0.3322, rel=1e-3)
    assert i_phase == pytest.approx(-10.24, abs=0.05)


# ===========================================================================
# 2. NILSSON & RIEDEL (8ª/10ª Edição)
# ===========================================================================

def test_nilsson_ex_4_10_thevenin_dependent_source():
    """Nilsson & Riedel, Exemplo 4.10:
    Circuito com fonte dependente para determinação do Equivalente de Thévenin:
    - Vs = 24 V (1 para 0)
    - R1 = 6 Ω (1 para 2)
    - R2 = 12 Ω (2 para 0)
    - VCVS E1: ligado entre nó 3 (+) e nó 2 (-), com ganho 1.5 * V(2, 0).
      Logo: V(3) = V(2) + 1.5 * V(2) = 2.5 * V(2).
    - Terminais de Thévenin: terminal 'a' = nó 3, terminal 'b' = nó 0.

    Solução analítica fechada:
      Divisor de tensão no nó 2 em circuito aberto:
        V(2) = 24 * (12 / (6 + 12)) = 16.0 V
        V_th = V(3) = 2.5 * 16.0 V = 40.0 V
      Resistência de Thévenin:
        Com Vs desativado (curto-circuito):
        R_eq no nó 2 = 6 // 12 = 4.0 Ω
        Aplicando corrente de teste I_test em '3':
        V(2) = I_test * 4.0 Ω
        V_test = V(3) = V(2) + 1.5 * V(2) = 2.5 * (4.0 * I_test) = 10.0 * I_test
        R_th = V_test / I_test = 10.0 Ω
      Corrente de Norton: I_N = 40V / 10Ω = 4.0 A
      Potência máxima transferível: P_max = 40^2 / (4 * 10) = 40.0 W
    """
    elements = [
        VoltageSource("Vs", "1", "0", 24.0),
        Resistor("R1", "1", "2", 6.0),
        Resistor("R2", "2", "0", 12.0),
        VCVS("E1", "3", "2", "2", "0", 1.5),
    ]
    circuit = Circuit(elements, allowed_open_nodes=["3"])
    res = thevenin_norton(circuit, node_a="3", node_b="0")

    assert res.V_th == pytest.approx(40.0)
    assert res.R_th == pytest.approx(10.0)
    assert res.I_N == pytest.approx(4.0)
    assert res.P_max == pytest.approx(40.0)


def test_nilsson_ex_5_3_weighted_inverting_summing_opamp():
    """Nilsson & Riedel, Exemplo 5.3:
    Amplificador somador inversor com 3 tensões ponderadas desiguais:
      va = 1.0 V através de Ra = 10 kΩ (ganho -10)
      vb = 2.0 V através de Rb = 20 kΩ (ganho -5)
      vc = 1.5 V através de Rc = 50 kΩ (ganho -2)
      Rf = 100 kΩ
      Carga RL = 25 kΩ na saída

    Solução analítica do livro:
      vo = - [ (100/10)*va + (100/20)*vb + (100/50)*vc ]
         = - [ 10*(1.0) + 5*(2.0) + 2*(1.5) ]
         = - [ 10 + 10 + 3 ] = -23.0 V
      Corrente através da carga RL:
        i_L = vo / RL = -23V / 25kΩ = -0.92 mA
      Corrente através do resistor de realimentação Rf:
        i_f = (0 - vo) / Rf = +23V / 100kΩ = +0.23 mA
    """
    elements = [
        VoltageSource("Va", "in_a", "0", 1.0),
        VoltageSource("Vb", "in_b", "0", 2.0),
        VoltageSource("Vc", "in_c", "0", 1.5),
        Resistor("Ra", "in_a", "vn", 10_000.0),
        Resistor("Rb", "in_b", "vn", 20_000.0),
        Resistor("Rc", "in_c", "vn", 50_000.0),
        Resistor("Rf", "vn", "out", 100_000.0),
        Resistor("RL", "out", "0", 25_000.0),
        OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("vn") == pytest.approx(0.0, abs=1e-12)
    assert sol.V("out") == pytest.approx(-23.0)
    assert sol.I_resistor("RL") == pytest.approx(-0.92e-3)
    assert sol.I_resistor("Rf") == pytest.approx(0.23e-3)


def test_nilsson_ex_9_6_ac_parallel_rlc():
    """Nilsson & Riedel, Exemplo 9.6:
    Circuito RLC em paralelo excitado por fonte de corrente senoidal de alta frequência:
      is(t) = 8 * cos(200 000 * t) A  =>  Is = 8 ∠ 0° A
      omega = 200 000 rad/s
      R = 50 Ω
      L = 125 μH (1.25e-4 H)  =>  ZL = j * 200 000 * 1.25e-4 = j 25 Ω
      C = 0.1 μF (1.0e-7 F)   =>  ZC = -j / (200 000 * 1.0e-7) = -j 50 Ω

    Admitância total:
      Y = 1/50 + 1/(j25) + 1/(-j50) = 0.02 - j 0.04 + j 0.02 = 0.02 - j 0.02 S
      Y = 0.02 * sqrt(2) ∠ -45° S

    Tensão no nó:
      V = Is / Y = 8 / (0.02 * (1 - j)) = 200 * (1 + j) = 200 + j 200 V
      |V| = 200 * sqrt(2) ≈ 282.8427 V,  fase = +45.0°

    Corrente no indutor:
      IL = V / (j25) = (200 + j200) / j25 = 8 - j 8 A
      |IL| = 8 * sqrt(2) ≈ 11.3137 A,  fase = -45.0°
    """
    omega_val = 200_000.0
    f_val = omega_val / (2.0 * np.pi)

    elements = [
        CurrentSource("Is", "1", "0", 0.0, ac_mag=8.0, ac_phase=0.0),
        Resistor("R1", "1", "0", 50.0),
        Inductor("L1", "1", "0", 1.25e-4),
        Capacitor("C1", "1", "0", 1.0e-7),
    ]
    circuit = Circuit(elements)
    sol = solve_ac(circuit, frequency=f_val)

    v1 = sol.V("1")
    v_mag = abs(v1)
    v_phase = float(np.degrees(np.angle(v1)))

    assert v_mag == pytest.approx(200.0 * np.sqrt(2.0), rel=1e-4)
    assert v_phase == pytest.approx(45.0, abs=0.05)

    i_ind = sol.I_inductor("L1")
    assert abs(i_ind) == pytest.approx(8.0 * np.sqrt(2.0), rel=1e-4)
    assert float(np.degrees(np.angle(i_ind))) == pytest.approx(-45.0, abs=0.05)


# ===========================================================================
# 3. BOYLESTAD (12ª/13ª Edição)
# ===========================================================================

def test_boylestad_cap09_millman_theorem():
    """Boylestad, Introdução à Análise de Circuitos, Capítulo 9 (Teoremas de Redes):
    Teorema de Millman aplicado a 3 geradores em paralelo alimentando carga comum:
      Ramo 1: E1 = 24 V com resistência interna R1 = 3 Ω  (G1 = 1/3 S)
      Ramo 2: E2 = -12 V com resistência interna R2 = 6 Ω (G2 = 1/6 S)
      Ramo 3: Carga pura R3 = 2 Ω (E3 = 0 V, G3 = 1/2 S)

    Fórmula clássica de Millman:
      VL = (E1*G1 + E2*G2 + E3*G3) / (G1 + G2 + G3)
         = (24/3 + (-12)/6 + 0) / (1/3 + 1/6 + 1/2)
         = (8 - 2) / 1.0 = 6.000 V

    Correntes em cada ramo:
      I1 = (24 - 6) / 3 = 6.0 A
      I2 = (-12 - 6) / 6 = -3.0 A  (bateria sendo recarregada)
      I3 = (6 - 0) / 2 = 3.0 A      (corrente drenada pela carga)
      KCL: I1 + I2 = 6 - 3 = 3.0 A = I3.
      Potência absorvida pela carga: PL = 6^2 / 2 = 18.0 W.
    """
    elements = [
        VoltageSource("E1", "n1", "0", 24.0),
        Resistor("R1", "n1", "L", 3.0),
        VoltageSource("E2", "0", "n2", 12.0),  # V(n2) = -12 V
        Resistor("R2", "n2", "L", 6.0),
        Resistor("R3", "L", "0", 2.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("L") == pytest.approx(6.0)
    assert sol.I_resistor("R1") == pytest.approx(6.0)
    assert sol.I_resistor("R2") == pytest.approx(-3.0)
    assert sol.I_resistor("R3") == pytest.approx(3.0)
    assert sol.P_resistor("R3") == pytest.approx(18.0)


def test_boylestad_cap08_r2r_ladder_network_4bit():
    """Boylestad, Capítulo 8: Rede Escada R-2R de 4 Bits para D/A Converter.
    Topologia da rede escada clássica:
      R = 10 kΩ, 2R = 20 kΩ.
      4 bits de entrada (d0=LSB, d1, d2, d3=MSB).
      Terminação 2R aterrada no nó n0.
      Tensão analítica na saída (nó n3 - MSB) para nível lógico alto (Vref = 16.0 V):
        - Apenas MSB (1000_2): Vout = Vref / 2  = 8.0 V
        - Apenas Bit2 (0100_2): Vout = Vref / 4  = 4.0 V
        - Apenas Bit1 (0010_2): Vout = Vref / 8  = 2.0 V
        - Apenas LSB  (0001_2): Vout = Vref / 16 = 1.0 V
        - Todos bits  (1111_2): Vout = Vref * (15/16) = 15.0 V
    """
    def run_ladder(v_bits: list[float]) -> float:
        elements = [
            Resistor("R_term", "n0", "0", 20_000.0),
            VoltageSource("Vd0", "d0", "0", v_bits[0]),
            Resistor("R_b0", "d0", "n0", 20_000.0),
            VoltageSource("Vd1", "d1", "0", v_bits[1]),
            Resistor("R_b1", "d1", "n1", 20_000.0),
            VoltageSource("Vd2", "d2", "0", v_bits[2]),
            Resistor("R_b2", "d2", "n2", 20_000.0),
            VoltageSource("Vd3", "d3", "0", v_bits[3]),
            Resistor("R_b3", "d3", "n3", 20_000.0),
            Resistor("Rs01", "n0", "n1", 10_000.0),
            Resistor("Rs12", "n1", "n2", 10_000.0),
            Resistor("Rs23", "n2", "n3", 10_000.0),
        ]
        return solve(Circuit(elements)).V("n3")

    v_ref = 16.0
    assert run_ladder([0.0, 0.0, 0.0, v_ref]) == pytest.approx(8.0)
    assert run_ladder([0.0, 0.0, v_ref, 0.0]) == pytest.approx(4.0)
    assert run_ladder([0.0, v_ref, 0.0, 0.0]) == pytest.approx(2.0)
    assert run_ladder([v_ref, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert run_ladder([v_ref, v_ref, v_ref, v_ref]) == pytest.approx(15.0)


def test_boylestad_cap06_balanced_wheatstone_null_condition():
    """Boylestad, Capítulo 6: Ponte de Wheatstone Balanceada (Condição Nula).
    Alimentação Vs = 20 V:
      Braço esquerdo: R1 = 100 Ω, R2 = 200 Ω (R1/R2 = 0.5)
      Braço direito:  R3 = 300 Ω, R4 = 600 Ω (R3/R4 = 0.5)
      Detector galvanômetro Rg = 50 Ω entre nós intermediários a e b.

    Como os braços são rigorosamente proporcionais:
      Va = 20 * (200 / 300) = 40/3 V ≈ 13.333333 V
      Vb = 20 * (600 / 900) = 40/3 V ≈ 13.333333 V
      Vab = 0.000000 V
      Ig = 0.000000 A
      Pg = 0.000000 W
    """
    elements = [
        VoltageSource("Vs", "1", "0", 20.0),
        Resistor("R1", "1", "a", 100.0),
        Resistor("R2", "a", "0", 200.0),
        Resistor("R3", "1", "b", 300.0),
        Resistor("R4", "b", "0", 600.0),
        Resistor("Rg", "a", "b", 50.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("a") == pytest.approx(40.0 / 3.0)
    assert sol.V("b") == pytest.approx(40.0 / 3.0)
    assert sol.V_diff("a", "b") == pytest.approx(0.0, abs=1e-12)
    assert sol.I_resistor("Rg") == pytest.approx(0.0, abs=1e-12)
    assert sol.P_resistor("Rg") == pytest.approx(0.0, abs=1e-15)


# ===========================================================================
# 4. DORF & SVOBODA (8ª/9ª Edição)
# ===========================================================================

def test_dorf_svoboda_cap06_inverting_opamp_t_network():
    """Dorf & Svoboda, Introdução aos Circuitos Elétricos, Capítulo 6:
    Amplificador Inversor com Rede em T de Realimentação (Feedback T-Network).
    Esta topologia permite ganhos muito elevados utilizando apenas resistores de
    valores práticos e moderados, evitando ruído e instabilidade térmica.

    Circuito:
      vin = 0.1 V
      R1 = 10 kΩ (entrada inversora vn)
      Rede T de Realimentação:
        R2 = 10 kΩ (de vn até o nó intermediário t)
        R4 = 1 kΩ  (do nó t até o terra 0)
        R3 = 10 kΩ (do nó t até a saída out)
      Carga de saída: RL = 20 kΩ

    Fórmula do Ganho de Tensão (Dorf & Svoboda):
      Av = vo / vin = - (R2 / R1) * [ 1 + R3/R2 + R3/R4 ]
         = - (10k / 10k) * [ 1 + 10k/10k + 10k/1k ]
         = - 1.0 * [ 1 + 1 + 10 ] = -12.0
      Tensão esperada na saída:
        vo = -12.0 * 0.1 V = -1.200 V
      Tensão no nó intermediário t:
        vt = -0.1 V
    """
    elements = [
        VoltageSource("Vin", "in", "0", 0.1),
        Resistor("R1", "in", "vn", 10_000.0),
        Resistor("R2", "vn", "t", 10_000.0),
        Resistor("R4", "t", "0", 1_000.0),
        Resistor("R3", "t", "out", 10_000.0),
        Resistor("RL", "out", "0", 20_000.0),
        OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("in") == pytest.approx(0.1)
    assert sol.V("vn") == pytest.approx(0.0, abs=1e-12)
    assert sol.V("t") == pytest.approx(-0.1)
    assert sol.V("out") == pytest.approx(-1.2)


def test_dorf_svoboda_cap06_voltage_follower_impedance_isolation():
    """Dorf & Svoboda, Capítulo 6:
    Buffer Seguidor de Tensão (Voltage Follower) para Casamento de Impedância.
    Demonstra a eliminação do efeito de carga (loading effect):
      - Fonte de sinal Vin = 5.0 V com resistência interna alta: Rs = 100 kΩ.
      - Carga de baixa resistência: RL = 50 Ω.
      - Sem buffer: Vload = 5.0 * 50 / (100 000 + 50) ≈ 2.49 mV (perda de 99.95%!).
      - Com Buffer Operacional: Rin = infinito -> corrente em Rs é zero -> Vp = 5.0 V.
        Saída do OpAmp reproduz integralmente 5.0 V sobre a carga de 50 Ω fornecendo 100 mA.
    """
    elements = [
        VoltageSource("Vin", "src", "0", 5.0),
        Resistor("Rs", "src", "vp", 100_000.0),
        OpAmp("U1", in_p="vp", in_n="out", out="out", gnd="0"),
        Resistor("RL", "out", "0", 50.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("src") == pytest.approx(5.0)
    assert sol.V("vp") == pytest.approx(5.0)   # Nenhuma queda em Rs
    assert sol.V("out") == pytest.approx(5.0)  # Reprodução integral 1:1
    assert sol.I_resistor("RL") == pytest.approx(100e-3)  # 100 mA
    assert sol.I_resistor("Rs") == pytest.approx(0.0, abs=1e-12)
