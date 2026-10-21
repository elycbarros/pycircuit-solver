"""
test_textbook_benchmarks.py — Validação Analítica Rigorosa Multi-Livros

Validação comparativa contra soluções analíticas e problemas canônicos dos 7 principais
manuais mundiais de Engenharia Elétrica e Eletrônica:
1. Alexander & Sadiku — Fundamentos de Circuitos Elétricos (5ª/6ª Edição)
2. Nilsson & Riedel — Circuitos Elétricos (8ª/10ª Edição)
3. Boylestad — Introdução à Análise de Circuitos (12ª/13ª Edição)
4. Dorf & Svoboda — Introdução aos Circuitos Elétricos (8ª/9ª Edição)
5. Sedra & Smith — Microelectronic Circuits / Microeletrônica (7ª/8ª Edição)
6. Hayt, Kemmerly & Durbin — Engineering Circuit Analysis / Análise de Circuitos em Engenharia (8ª/9ª Edição)
7. Irwin & Nelms — Basic Engineering Circuit Analysis / Análise Básica de Circuitos (11ª/12ª Edição)

Problemas Cobertos:
-------------------
[Alexander & Sadiku]
1. Cap. 2 (Ex. 2.8)   — Divisor de Tensão e Corrente com Associação Série-Paralelo
2. Cap. 2 (Ex. 2.15)  — Transformação Delta-Estrela (Δ-Y) em Circuito em Ponte
3. Cap. 3 (Ex. 3.1)   — Análise Nodal com Múltiplas Fontes de Corrente
4. Cap. 3 (Ex. 3.2)   — Análise Nodal com Fonte Dependente (VCCS)
5. Cap. 3 (Ex. 3.3)   — Análise Nodal com Supernó (Fonte de Tensão entre nós não-terra)
6. Cap. 4 (Ex. 4.8)   — Teorema de Thévenin, Norton e Máxima Transferência de Potência
7. Cap. 4 (Ex. 4.9)   — Equivalente de Thévenin com Fonte Controlada (VCVS)
8. Cap. 5 (Ex. 5.2)   — Amplificador Operacional Inversor
9. Cap. 5 (Ex. 5.5)   — Amplificador Operacional Não-Inversor
10. Cap. 5 (Ex. 5.6)  — Amplificador Somador Inversor
11. Cap. 5 (Ex. 5.7)  — Amplificador Diferencial (Subtrator)
12. Cap. 9 (Ex. 9.9)  — Circuito RLC Série em Regime Permanente CA
13. Cap. 6 (Ex. 6.13) — Amplificador Integrador Ideal com Op-Amp e Resposta Transitória
14. Cap. 13 (Ex. 13.1)— Acoplamento Magnético Mútuo e Transformador Linear

[Nilsson & Riedel]
15. Cap. 4 (Ex. 4.10) — Equivalente de Thévenin com Fonte Controlada
16. Cap. 5 (Ex. 5.3)  — Amplificador Somador Inversor com Pesos Arbitrários
17. Cap. 9 (Ex. 9.6)  — Análise Fasorial de Circuito RLC Paralelo em Alta Frequência

[Boylestad]
18. Cap. 9 — Teorema de Millman para Múltiplas Baterias com Resistência Interna
19. Cap. 8 — Rede Escada R-2R de 4 Bits para Conversor Digital-Analógico (D/A)
20. Cap. 6 — Ponte de Wheatstone em Equilíbrio (Condição de Nulidade do Detector)

[Dorf & Svoboda]
21. Cap. 6 — Amplificador Inversor com Rede em T na Realimentação (Alto Ganho)
22. Cap. 6 — Buffer Seguidor de Tensão e Isolamento de Carga com Alta Impedância

[Sedra & Smith]
23. Cap. 4 — Diodo Retificador PN e Ponto de Operação Quiescente Q com Reta de Carga
24. Cap. 6 — Polarização de BJT NPN por Divisor de Tensão com 4 Resistores na Região Ativa
25. Cap. 5 — Transistor MOSFET NMOS Operando em Saturação com Carga de Dreno

[Hayt, Kemmerly & Durbin]
26. Cap. 5 — Princípio e Teorema da Superposição com Três Fontes Independentes
27. Cap. 7 — Resposta Transitória ao Degrau de Circuito RC de Primeira Ordem
28. Cap. 9 — Circuito RLC Série em Ressonância e Fator de Qualidade Q

[Irwin & Nelms]
29. Cap. 4 — Amplificador de Instrumentação (INA) Clássico de 3 Op-Amps
30. Cap. 5 — Ponte de Wheatstone Desbalanceada e Equivalente de Thévenin Exato
31. Cap. 3 — Fonte Controlada por Corrente CCCS (F) em Análise Nodal Modificada
"""

import math
import numpy as np
import pytest

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    BJT,
    CCCS,
    CCVS,
    MOSFET,
    Capacitor,
    CurrentSource,
    Diode,
    Inductor,
    MutualCoupling,
    OpAmp,
    Resistor,
    Transformer,
    VCCS,
    VCVS,
    VoltageSource,
)
from circuit_solver.engine.ac import solve_ac
from circuit_solver.engine.mna import solve
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.engine.transient import solve_transient


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


# ===========================================================================
# 5. SEDRA & SMITH — Microelectronic Circuits (7ª/8ª Edição)
# ===========================================================================

def test_sedra_smith_cap04_diode_operating_point():
    """Sedra & Smith, Microelectronic Circuits, Capítulo 4 (Diodes):
    Ponto de Operação Quiescente (Q-point) de Diodo com Reta de Carga.
    Circuito com fonte DC Vs = 5.0 V, resistor limitador R = 1.0 kΩ
    e diodo de junção PN com parâmetros de modelo de Shockley:
      Is = 1e-14 A, n = 1.0, Vt = 25.85 mV (T = 300 K).

    Equação transcendental não-linear resolvida por Newton-Raphson:
      Vs - R * ID - VD = 0, com ID = Is * (exp(VD / (n*Vt)) - 1).

    Solução analítica e de simulação:
      VD ≈ 0.6925 V (tensão direta típica de junção de silício)
      ID = (5.0 - 0.6925) / 1000 ≈ 4.3075 mA
      PD = VD * ID ≈ 2.983 mW
    """
    elements = [
        VoltageSource("Vs", "1", "0", 5.0),
        Resistor("R1", "1", "2", 1000.0),
        Diode("D1", "2", "0", Is=1e-14, n=1.0, Vt=0.02585),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(5.0)
    assert 0.68 < sol.V("2") < 0.71
    vd = sol.V("2")
    id_expected = (5.0 - vd) / 1000.0
    assert sol.I_diode("D1") == pytest.approx(id_expected, rel=1e-5)
    assert sol.iterations <= 15


def test_sedra_smith_cap06_bjt_four_resistor_bias_network():
    """Sedra & Smith, Capítulo 6 (BJT):
    Rede clássica de polarização de transistor BJT NPN por divisor de tensão (4 resistores).
    Esta é a topologia canônica universal para estabilização térmica do ponto Q:
      Vcc = 12.0 V
      R1 = 47 kΩ (vcc para base 'b')
      R2 = 10 kΩ (base 'b' para terra '0')
      Rc = 3.3 kΩ (vcc para coletor 'c')
      Re = 1.0 kΩ (emissor 'e' para terra '0')
      Transistor NPN com ganho de corrente beta = 100.

    Análise analítica de Thévenin na base:
      Vth_base = 12 * (10 / (47 + 10)) = 120 / 57 ≈ 2.105 V
      Rth_base = 47k // 10k ≈ 8.246 kΩ
      Considerando V_BE ≈ 0.70 V:
      IB = (Vth - V_BE) / [Rth + (beta + 1) * Re]
         = (2.105 - 0.70) / [8246 + 101 * 1000] ≈ 1.405 / 109246 ≈ 12.86 μA
      IC = beta * IB ≈ 1.286 mA
      IE = (beta + 1) * IB ≈ 1.299 mA
      VE = IE * Re ≈ 1.30 V
      VC = Vcc - IC * Rc = 12 - 1.286m * 3.3k ≈ 7.75 V
      VCE = VC - VE ≈ 6.45 V > 0.2 V  -->  Região Ativa Direta (Forward Active)
    """
    elements = [
        VoltageSource("VCC", "vcc", "0", 12.0),
        Resistor("R1", "vcc", "b", 47_000.0),
        Resistor("R2", "b", "0", 10_000.0),
        Resistor("RC", "vcc", "c", 3_300.0),
        Resistor("RE", "e", "0", 1_000.0),
        BJT("Q1", "c", "b", "e", polarity="NPN", beta=100.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert 7.4 < sol.V("c") < 7.8
    assert 1.9 < sol.V("b") < 2.1
    assert 1.2 < sol.V("e") < 1.4
    # Garantia de operação na região ativa direta:
    v_ce = sol.V("c") - sol.V("e")
    assert v_ce > 5.0


def test_sedra_smith_cap05_nmos_common_source_saturation():
    """Sedra & Smith, Capítulo 5 (MOSFET):
    Amplificador Fonte Comum NMOS polarizado rigorosamente na Região de Saturação.
    Equação quadrática canônica:
      ID = 0.5 * Kp * (VGS - Vth)^2 * (1 + lambda * VDS)

    Circuito:
      VDD = 10.0 V, VG = 3.5 V, VS = 0 V (terra) -> VGS = 3.5 V
      Parâmetros do NMOS: Vth = 1.5 V, Kp = 5e-4 A/V^2, lambda = 0
      Tensão de overdrive: Vov = VGS - Vth = 3.5 - 1.5 = 2.0 V
      Corrente teórica de dreno:
        ID = 0.5 * 5e-4 * (2.0)^2 = 0.5 * 5e-4 * 4.0 = 1.0 mA
      Resistor de carga de dreno: RD = 2.5 kΩ
      Tensão esperada no dreno:
        VD = VDD - ID * RD = 10.0 - 1.0m * 2.5k = 7.500 V
      Condição de saturação:
        VDS = 7.5 V >= Vov = 2.0 V  (Saturação confirmada com ampla margem!)
    """
    elements = [
        VoltageSource("VDD", "vdd", "0", 10.0),
        VoltageSource("VG", "g", "0", 3.5),
        Resistor("RD", "vdd", "d", 2_500.0),
        MOSFET("M1", "d", "g", "0", polarity="NMOS", Vth=1.5, Kp=5e-4, lambda_param=0.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("vdd") == pytest.approx(10.0)
    assert sol.V("g") == pytest.approx(3.5)
    assert sol.V("d") == pytest.approx(7.5, abs=1e-4)


# ===========================================================================
# 6. HAYT, KEMMERLY & DURBIN — Engineering Circuit Analysis (8ª/9ª Edição)
# ===========================================================================

def test_hayt_cap05_superposition_theorem():
    """Hayt, Kemmerly & Durbin, Engineering Circuit Analysis, Capítulo 5:
    Teorema e Princípio da Superposição Linear em rede com 3 fontes independentes.
    Topologia da rede conectada ao nó central 2:
      - Ramo 1: Fonte V1 = 24 V em série com R1 = 8 Ω
      - Ramo 2: Fonte V2 = 12 V em série com R2 = 4 Ω
      - Ramo 3: Fonte de corrente I1 = 3 A injetada diretamente no nó 2
      - Ramo 4: Resistor de terminação R3 = 8 Ω ligado ao terra 0

    Equação Nodal exata no nó 2:
      (V2 - 24)/8 + (V2 - 12)/4 + V2/8 - 3 = 0
      Multiplicando por 8:
      (V2 - 24) + 2*(V2 - 12) + V2 - 24 = 0
      4*V2 - 72 = 0  ==>  V2 = 18.000 V

    Decomposição analítica termo a termo por Superposição:
      1. Apenas V1 ativa (V2=0, I1=0): V2_V1 = 24 * (8//4 / (8 + 8//4)) = 24 * (8/32) = 6.0 V
      2. Apenas V2 ativa (V1=0, I1=0): V2_V2 = 12 * (8//8 / (4 + 8//8)) = 12 * (4/8)  = 6.0 V
      3. Apenas I1 ativa (V1=0, V2=0): V2_I1 = 3 A * (8 // 4 // 8) = 3 A * 2.0 Ω      = 6.0 V
      Soma total: V2 = 6.0 + 6.0 + 6.0 = 18.000 V.
    """
    elements = [
        VoltageSource("V1", "1", "0", 24.0),
        Resistor("R1", "1", "2", 8.0),
        VoltageSource("V2", "3", "0", 12.0),
        Resistor("R2", "3", "2", 4.0),
        CurrentSource("I1", "2", "0", 3.0),
        Resistor("R3", "2", "0", 8.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("2") == pytest.approx(18.0)
    assert sol.I_resistor("R1") == pytest.approx((24.0 - 18.0) / 8.0)  # 0.75 A
    assert sol.I_resistor("R2") == pytest.approx((12.0 - 18.0) / 4.0)  # -1.5 A
    assert sol.I_resistor("R3") == pytest.approx(18.0 / 8.0)           # 2.25 A


def test_hayt_cap07_first_order_rc_transient_step_response():
    """Hayt, Kemmerly & Durbin, Capítulo 7:
    Resposta Transitória ao Degrau de Circuito RC de Primeira Ordem.
    Fonte de degrau Vs = 10.0 V aplicada em t = 0 a R = 1.0 kΩ e C = 1.0 μF.
    Constante de tempo: tau = R * C = 1000 * 1e-6 = 1.0 ms.

    Solução analítica fechada da equação diferencial:
      v_C(t) = Vs * (1 - exp(-t / tau))

    Marcos canônicos temporais da curva de carga:
      - Em t = 1*tau (1.0 ms): v_C = 10 * (1 - e^-1) ≈ 6.3212 V (63.2% da carga)
      - Em t = 2*tau (2.0 ms): v_C = 10 * (1 - e^-2) ≈ 8.6466 V (86.5% da carga)
      - Em t = 5*tau (5.0 ms): v_C = 10 * (1 - e^-5) ≈ 9.9326 V (>99.3%, regime permanente)
    """
    elements = [
        VoltageSource("Vs", "1", "0", 10.0),
        Resistor("R1", "1", "2", 1000.0),
        Capacitor("C1", "2", "0", 1.0e-6, ic=0.0),
    ]
    circuit = Circuit(elements)
    # Simulação transitória com uic=True (condição inicial nula)
    dt = 1e-5
    res = solve_transient(circuit, t_stop=5.0e-3, dt=dt, uic=True)

    idx_1ms = int(round(1.0e-3 / dt))
    idx_2ms = int(round(2.0e-3 / dt))
    idx_5ms = int(round(5.0e-3 / dt))

    v_1ms_analytical = 10.0 * (1.0 - math.exp(-1.0))
    v_2ms_analytical = 10.0 * (1.0 - math.exp(-2.0))
    v_5ms_analytical = 10.0 * (1.0 - math.exp(-5.0))

    assert res.V("2")[idx_1ms] == pytest.approx(v_1ms_analytical, rel=0.005)
    assert res.V("2")[idx_2ms] == pytest.approx(v_2ms_analytical, rel=0.005)
    assert res.V("2")[idx_5ms] == pytest.approx(v_5ms_analytical, rel=0.005)


def test_hayt_cap09_series_rlc_resonance_and_quality_factor():
    """Hayt, Kemmerly & Durbin, Capítulo 9:
    Circuito RLC Série em Frequência de Ressonância e Fator de Qualidade Q.
    Valores:
      R = 10.0 Ω, L = 0.1 H, C = 10.0 μF
      Fonte CA: Vs = 10.0 ∠ 0° V

    Parâmetros analíticos de ressonância:
      omega_0 = 1 / sqrt(L * C) = 1 / sqrt(0.1 * 10e-6) = 1000 rad/s
      f_0 = omega_0 / (2 * pi) ≈ 159.155 Hz
      Impedância indutiva: XL = omega_0 * L = 1000 * 0.1 = 100.0 Ω
      Impedância capacitiva: XC = 1 / (omega_0 * C) = 1 / (1000 * 10e-6) = 100.0 Ω
      Impedância total série na ressonância:
        Z = R + j(XL - XC) = 10.0 + j(100 - 100) = 10.0 Ω (puramente resistiva!)
      Fator de qualidade: Q = omega_0 * L / R = 100 / 10 = 10.0

    Consequências analíticas:
      - Corrente máxima de pico: I = Vs / R = 10 V / 10 Ω = 1.000 ∠ 0° A
      - Tensão no capacitor: V_C = I * (-j XC) = 1.0 * (-j100) = 100.0 ∠ -90° V
        (Amplificação de tensão reativa de Q vezes: V_C = Q * Vs = 10 * 10 = 100 V!)
    """
    f0 = 1000.0 / (2.0 * math.pi)
    elements = [
        VoltageSource("Vs", "1", "0", voltage=10.0, ac_mag=10.0, ac_phase=0.0),
        Resistor("R", "1", "2", 10.0),
        Inductor("L", "2", "3", 0.1),
        Capacitor("C", "3", "0", 10.0e-6),
    ]
    circuit = Circuit(elements)
    sol = solve_ac(circuit, frequency=f0)

    # Corrente de pico em ressonância = 1.0 A
    assert abs(sol.I_resistor("R")) == pytest.approx(1.0, rel=1e-4)
    # Tensão amplificada sobre o capacitor = 100.0 V
    assert sol.V_mag("3") == pytest.approx(100.0, rel=1e-3)


# ===========================================================================
# 7. IRWIN & NELMS — Basic Engineering Circuit Analysis (11ª/12ª Edição)
# ===========================================================================

def test_irwin_cap04_three_opamp_instrumentation_amplifier():
    """Irwin & Nelms, Basic Engineering Circuit Analysis, Capítulo 4:
    Amplificador de Instrumentação (INA) Clássico de Três Op-Amps.
    Topologia de padrão industrial para aquisição de sinais médicos e industriais:
      - Dois amplificadores não-inversores de entrada desacoplados com resistor de ganho RG.
      - Um estágio subtrator diferencial na saída com resistores casados.

    Parâmetros:
      R1a = R1b = 20.0 kΩ, RG = 1.0 kΩ
      R2a = R2b = 10.0 kΩ, R3a = R3b = 10.0 kΩ
      Entradas diferenciais com modo comum:
        V1 = 1.05 V (entrada não-inversora)
        V2 = 1.00 V (entrada inversora)
        Tensão diferencial: Vd = V1 - V2 = +50.0 mV
        Tensão de modo comum: Vcm = 1.025 V

    Fórmula Analítica do Ganho Diferencial (Irwin & Nelms):
      Ad = (1 + 2*R1 / RG) * (R3 / R2)
         = (1 + 2*20k / 1k) * (10k / 10k) = (1 + 40) * 1.0 = 41.0
      Tensão esperada na saída:
        Vout = Ad * (V1 - V2) = 41.0 * 0.050 V = 2.050 V
      A componente de modo comum Vcm é idealmente rejeitada (CMRR = infinito).
    """
    elements = [
        VoltageSource("V1", "in1", "0", 1.05),
        VoltageSource("V2", "in2", "0", 1.00),
        # Estágio de entrada com buffers cruzados
        OpAmp("U1", in_p="in1", in_n="vn1", out="vo1", gnd="0"),
        Resistor("R1a", "vo1", "vn1", 20_000.0),
        Resistor("RG", "vn1", "vn2", 1_000.0),
        Resistor("R1b", "vn2", "vo2", 20_000.0),
        OpAmp("U2", in_p="in2", in_n="vn2", out="vo2", gnd="0"),
        # Estágio de saída subtrator diferencial
        Resistor("R2a", "vo2", "vn3", 10_000.0),
        Resistor("R3a", "vn3", "out", 10_000.0),
        Resistor("R2b", "vo1", "vp3", 10_000.0),
        Resistor("R3b", "vp3", "0", 10_000.0),
        OpAmp("U3", in_p="vp3", in_n="vn3", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("in1") == pytest.approx(1.05)
    assert sol.V("in2") == pytest.approx(1.00)
    assert sol.V("out") == pytest.approx(2.050)


def test_irwin_cap05_unbalanced_bridge_thevenin_equivalent():
    """Irwin & Nelms, Capítulo 5:
    Ponte de Wheatstone Desbalanceada e Extração Exata do Equivalente de Thévenin.
    Circuito alimentado por Vs = 12.0 V:
      Ramo esquerdo: R1 = 100 Ω (Vs para 'a'), R2 = 200 Ω ('a' para terra 0)
      Ramo direito:  R3 = 300 Ω (Vs para 'b'), R4 = 400 Ω ('b' para terra 0)

    Cálculo analítico entre os nós a e b:
      Va = 12 * (200 / (100 + 200)) = 12 * (2/3) = 8.000 V
      Vb = 12 * (400 / (300 + 400)) = 12 * (4/7) = 48/7 ≈ 6.857143 V
      Vth = Va - Vb = 8 - 48/7 = 8/7 ≈ 1.142857 V

    Resistência de Thévenin vista de 'a' e 'b' (fonte desativada em curto):
      Rth = (R1 // R2) + (R3 // R4)
          = (100*200 / 300) + (300*400 / 700)
          = 200/3 + 1200/7 = (1400 + 3600) / 21 = 5000 / 21 ≈ 238.0952 Ω
    """
    elements = [
        VoltageSource("Vs", "1", "0", 12.0),
        Resistor("R1", "1", "a", 100.0),
        Resistor("R2", "a", "0", 200.0),
        Resistor("R3", "1", "b", 300.0),
        Resistor("R4", "b", "0", 400.0),
    ]
    circuit = Circuit(elements)
    th = thevenin_norton(circuit, node_a="a", node_b="b")

    assert th.V_th == pytest.approx(8.0 / 7.0, rel=1e-6)
    assert th.R_th == pytest.approx(5000.0 / 21.0, rel=1e-6)


def test_irwin_cap03_cccs_dependent_source_nodal_analysis():
    """Irwin & Nelms, Capítulo 3:
    Análise Nodal Modificada com Fonte Controlada por Corrente (CCCS - Elemento F SPICE).
    Circuito:
      - Fonte independente Vs = 10.0 V conectada ao resistor R1 = 5.0 Ω.
      - Corrente gerada pela fonte de controle: I(Vs) = Vs / R1 = 10 / 5 = 2.0 A saindo do terminal positivo.
      - Fonte controlada por corrente CCCS F1: ganho beta = 4.0, bombeia corrente no nó 2.
      - Resistor de carga R2 = 10.0 Ω ligado entre o nó 2 e o terra.

    Solução analítica:
      Corrente injetada por F1: I_F1 = 4.0 * I(Vs) = 4.0 * 2.0 A = 8.0 A
      Tensão gerada sobre a carga: V(2) = I_F1 * R2 = 8.0 A * 10.0 Ω = 80.000 V
    """
    elements = [
        VoltageSource("Vs", "1", "0", 10.0),
        Resistor("R1", "1", "0", 5.0),
        CCCS("F1", "2", "0", vctrl_name="Vs", gain=4.0),
        Resistor("R2", "2", "0", 10.0),
    ]
    circuit = Circuit(elements)
    sol = solve(circuit)

    assert sol.V("1") == pytest.approx(10.0)
    assert sol.V("2") == pytest.approx(80.0)
    assert sol.I_resistor("R2") == pytest.approx(8.0)


# ===========================================================================
# 8. ALEXANDER & SADIKU — Casos Avançados (Capítulos 6 e 13)
# ===========================================================================

def test_sadiku_cap06_ideal_opamp_integrator_transient():
    """Alexander & Sadiku, Capítulo 6:
    Amplificador Integrador Ideal com Amplificador Operacional e Capacitor no Feedback.
    Equação constitutiva ideal de transferência no tempo:
      v_out(t) = - 1 / (R * C) * integral_0^t v_in(tau) dtau + v_out(0)

    Valores:
      v_in(t) = constante de +2.0 V (degrau DC)
      R = 10.0 kΩ, C = 1.0 μF  -->  tau = R * C = 10.0 ms
      Taxa de integração linear analítica (rampa descendente):
        dv_out / dt = - Vin / (R * C) = - 2.0 / 0.01 = -200.0 V/s

    Em t = 5.0 ms:
      v_out(5 ms) = - 200.0 V/s * 0.005 s = - 1.000 V
    """
    elements = [
        VoltageSource("Vin", "in", "0", 2.0),
        Resistor("R", "in", "vn", 10_000.0),
        Capacitor("C", "vn", "out", 1.0e-6, ic=0.0),
        OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
    ]
    circuit = Circuit(elements)
    dt = 1e-5
    res = solve_transient(circuit, t_stop=5.0e-3, dt=dt, uic=True)

    assert res.V("out")[-1] == pytest.approx(-1.000, abs=0.005)


def test_sadiku_cap13_linear_transformer_mutual_coupling():
    """Alexander & Sadiku, Capítulo 13 (Exemplo 13.1):
    Circuito com Bobinas Acopladas Magneticamente em Regime Permanente Senoidal CA.
    Parâmetros:
      Vs = 12 ∠ 0° V a omega = 1.0 rad/s (f = 1 / (2*pi) Hz)
      Primário: R1 = 4.0 Ω, L1 = 8.0 H
      Secundário: L2 = 10.0 H, C1 = 0.5 F (XC = -j2 Ω), R2 = 5.0 Ω
      Acoplamento mútuo: M = 1.0 H ==> k = 1 / sqrt(8 * 10) ≈ 0.1118034

    Solução analítica exata de Sadiku:
      Corrente fasorial de malha primária:   I1 = 1.350 ∠ -62.85° A
      Corrente fasorial de malha secundária: I2 = 0.1431 ∠ -30.85° A
    """
    omega = 1.0
    f = omega / (2.0 * math.pi)
    k = 1.0 / math.sqrt(8.0 * 10.0)

    elements = [
        VoltageSource("V1", "1", "0", voltage=12.0, ac_mag=12.0, ac_phase=0.0),
        Resistor("R1", "1", "2", 4.0),
        Inductor("L1", "2", "0", 8.0),
        Inductor("L2", "3", "0", 10.0),
        Capacitor("C1", "3", "4", 0.5),
        Resistor("R2", "4", "0", 5.0),
        MutualCoupling("K1", "L1", "L2", k),
    ]
    circuit = Circuit(elements)
    sol = solve_ac(circuit, frequency=f)

    # Corrente do primário (em R1)
    i1 = sol.I_resistor("R1")
    i1_mag = abs(i1)
    i1_phase = float(np.degrees(np.angle(i1)))
    assert i1_mag == pytest.approx(1.350, abs=0.01)
    assert i1_phase == pytest.approx(-62.85, abs=0.2)

    # Corrente do secundário (em R2)
    i2 = sol.I_resistor("R2")
    i2_mag = abs(i2)
    i2_phase = float(np.degrees(np.angle(i2)))
    assert i2_mag == pytest.approx(0.1431, abs=0.005)
    assert i2_phase == pytest.approx(-30.85, abs=0.3)

