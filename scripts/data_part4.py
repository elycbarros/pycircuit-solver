r"""
data_part4.py — Memória de Cálculo: Exercícios 44 a 48
Foco Especial em Análise Avançada de Circuitos (Circuit Theory & Systems):
- Análise de Supermalhas com Fontes Compartilhadas
- Transitórios RLC de 2ª Ordem Superamortecidos (\alpha > \omega_0)
- Transitórios RLC de 2ª Ordem Criticamente Amortecidos (\alpha = \omega_0)
- Sistemas Trifásicos Equilibrados e Método dos Dois Wattímetros (Teorema de Blondel)
- Redes de Duas Portas / Quadripolos (Extração das Matrizes Z, Y, H, ABCD)
"""
import math
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor, VoltageSource, CurrentSource, Capacitor, Inductor
)
from circuit_solver.engine.mna import solve
from circuit_solver.engine.ac import solve_ac
from circuit_solver.engine.transient import solve_transient
from circuit_solver.engine.two_port import analyze_two_port
from circuit_solver.engine.three_phase import (
    build_three_phase_source,
    build_delta_load,
    analyze_two_wattmeters,
)

def run_ex44():
    c = Circuit([
        VoltageSource("V1", "1", "0", 20.0),
        Resistor("R1", "1", "2", 6.0),
        Resistor("R2", "2", "0", 10.0),
        CurrentSource("Is", "3", "2", 4.0),
        Resistor("R3", "3", "0", 4.0),
        Resistor("R4", "3", "0", 2.0),
    ])
    sol = solve(c)
    return [
        ("Tensão Nodal V(1)", 20.0, float(sol.V("1")), "V"),
        ("Tensão Nodal V(2)", -2.5, float(sol.V("2")), "V"),
        ("Tensão Nodal V(3)", 16.0 / 3.0, float(sol.V("3")), "V"),
        ("Corrente de Ramo I(R1)", 3.75, float(sol.I_resistor("R1")), "A"),
        ("Corrente de Ramo I(R2)", -0.25, float(sol.I_resistor("R2")), "A"),
        ("Corrente de Ramo I(R3)", 4.0 / 3.0, float(sol.I_resistor("R3")), "A"),
        ("Corrente de Ramo I(R4)", 8.0 / 3.0, float(sol.I_resistor("R4")), "A"),
    ]

def run_ex45():
    c = Circuit([
        VoltageSource("Vs", "1", "0", 24.0),
        Resistor("R", "1", "2", 9.0),
        Inductor("L", "2", "3", 1.0, ic=0.0),
        Capacitor("C", "3", "0", 0.05, ic=0.0),
    ])
    res = solve_transient(c, t_stop=2.0, dt=1e-3, uic=True)
    vc = res.V("3")
    # Ponto t = 0.5 s:
    idx_05 = int(0.5 / 1e-3)
    vc_ana_05 = 24.0 - 120.0 * np.exp(-4.0 * 0.5) + 96.0 * np.exp(-5.0 * 0.5)
    # Ponto t = 1.0 s:
    idx_10 = int(1.0 / 1e-3)
    vc_ana_10 = 24.0 - 120.0 * np.exp(-4.0 * 1.0) + 96.0 * np.exp(-5.0 * 1.0)
    return [
        ("Tensão v_C(0.5 s)", vc_ana_05, float(vc[idx_05]), "V"),
        ("Tensão v_C(1.0 s)", vc_ana_10, float(vc[idx_10]), "V"),
        ("Tensão v_C(2.0 s), antes do regime", 24.0 - 120.0 * np.exp(-8.0) + 96.0 * np.exp(-10.0), float(vc[-1]), "V"),
    ]

def run_ex46():
    c = Circuit([
        Resistor("R", "1", "0", 50.0),
        Capacitor("C", "1", "0", 2.0e-6, ic=12.0),
        Inductor("L", "1", "0", 0.020, ic=0.0),
    ])
    dt = 2e-6
    res = solve_transient(c, t_stop=0.0015, dt=dt, uic=True)
    v = res.V("1")
    # Ponto t = 0.1 ms (0.0001 s)
    idx_01ms = round(0.0001 / dt)
    v_ana_01ms = (12.0 - 60000.0 * 0.0001) * np.exp(-5000.0 * 0.0001)
    # Ponto de cruzamento por zero t = 0.2 ms (0.0002 s)
    idx_zero = round(0.0002 / dt)
    return [
        ("Tensão v(0.1 ms)", v_ana_01ms, float(v[idx_01ms]), "V"),
        ("Tensão v no Zero (0.2 ms)", 0.0, float(v[idx_zero]), "V"),
    ]

def run_ex47():
    freq = 60.0
    w = 2.0 * math.pi * freq
    r_branch = 24.0
    l_branch = 18.0 / w
    v_ln = 120.0

    sources = build_three_phase_source(
        connection="Y", V_rms=v_ln, sequence="ABC",
        node_a="s_a", node_b="s_b", node_c="s_c", node_neutral="0", prefix="Vs"
    )
    load = build_delta_load(
        R=r_branch, L=l_branch, node_a="s_a", node_b="s_b", node_c="s_c", prefix="Zd"
    )
    c = Circuit(sources + load)
    sol = solve_ac(c, frequency=freq)

    v_a = sol.V("s_a")
    v_b = sol.V("s_b")
    v_c = sol.V("s_c")
    z_branch = r_branch + 1j * w * l_branch
    i_ab = (v_a - v_b) / z_branch
    i_bc = (v_b - v_c) / z_branch
    i_ca = (v_c - v_a) / z_branch
    i_a = i_ab - i_ca
    i_b = i_bc - i_ab

    wm = analyze_two_wattmeters(
        V_line_ac=(v_a - v_c),
        V_line_bc=(v_b - v_c),
        I_line_a=i_a,
        I_line_b=i_b,
    )
    return [
        ("Corrente de Linha |I_L|", 12.0, float(abs(i_a)), "A"),
        ("Leitura Wattímetro 1 (W1)", 2476.24594887, float(wm.W1), "W"),
        ("Leitura Wattímetro 2 (W2)", 979.75405113, float(wm.W2), "W"),
        ("Potência Ativa Total P_3φ", 3456.0, float(wm.P_total), "W"),
        ("Potência Reativa Total Q_3φ", 2592.0, float(wm.Q_total), "VAR"),
        ("Potência Aparente Total S_3φ", 4320.0, float(wm.S_total), "VA"),
        ("Fator de Potência Global (FP)", 0.800, float(wm.power_factor), "-"),
    ]

def run_ex48():
    c = Circuit([
        Resistor("R1", "n1", "mid", 20.0),
        Resistor("R2", "n2", "mid", 40.0),
        Resistor("R3", "mid", "0", 30.0),
    ], allowed_open_nodes=["n1", "n2"])

    res = analyze_two_port(c, port1=("n1", "0"), port2=("n2", "0"))
    zin = res.input_impedance(Z_load=50.0)
    return [
        ("Parâmetro Z11", 50.0, float(res.z11.real), "Ω"),
        ("Parâmetro Z12", 30.0, float(res.z12.real), "Ω"),
        ("Parâmetro Z21", 30.0, float(res.z21.real), "Ω"),
        ("Parâmetro Z22", 70.0, float(res.z22.real), "Ω"),
        ("Parâmetro de Transmissão A", 5.0 / 3.0, float(res.A.real), "-"),
        ("Parâmetro de Transmissão B", 260.0 / 3.0, float(res.B.real), "Ω"),
        ("Parâmetro de Transmissão C", 1.0 / 30.0, float(res.C.real), "S"),
        ("Parâmetro de Transmissão D", 7.0 / 3.0, float(res.D.real), "-"),
        ("Impedância de Entrada Zin (RL = 50 Ω)", 42.5, float(zin.real), "Ω"),
    ]


EXERCISES_PART4 = [
    {
        "id": 44,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª/6ª Edição)",
        "chapter": "Capítulo 3 — Métodos de Análise",
        "title": "Análise de Malhas com Supermalha (Supermesh) e Fonte de Corrente Compartilhada",
        "category": "Teoria de Circuitos DC",
        "enunciado": (
            "Determine analiticamente todas as tensões nodais e correntes de ramo no circuito com supermalha formado por "
            "uma fonte independente de tensão contínua $V_1 = 20.0\\text{ V}$, quatro resistores lineares ($R_1 = 6.0\\ \\Omega$, "
            "$R_2 = 10.0\\ \\Omega$, $R_3 = 4.0\\ \\Omega$, $R_4 = 2.0\\ \\Omega$) e uma fonte independente de corrente $I_s = 4.0\\text{ A}$ "
            "compartilhada entre duas malhas essenciais adjacentes."
        ),
        "topologia": [
            "Fonte independente de tensão contínua $V_1 = 20.0\\text{ V}$ conectada entre nó 1 e terra (0).",
            "Resistor $R_1 = 6.0\\ \\Omega$ conectado entre nó 1 e nó 2.",
            "Resistor $R_2 = 10.0\\ \\Omega$ conectado entre nó 2 e terra (0).",
            "Fonte independente de corrente $I_s = 4.0\\text{ A}$ conectada do nó 2 ao nó 3 (ramo da supermalha).",
            "Resistor $R_3 = 4.0\\ \\Omega$ conectado entre nó 3 e terra (0).",
            "Resistor $R_4 = 2.0\\ \\Omega$ conectado entre nó 3 e terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Conceito e Identificação da Supermalha</h5>
Quando uma fonte de corrente (independente ou controlada) situa-se na fronteira compartilhada entre duas malhas essenciais, a tensão sobre seus terminais é inicialmente desconhecida. Cria-se então uma <em>Supermalha</em> removendo-se temporariamente a fonte de corrente e aplicando a Lei das Tensões de Kirchhoff (LKT) ao contorno combinado, complementada pela equação de restrição de ramo (LKC).

Pela formulação da Análise Nodal Modificada (MNA), a presença da fonte de corrente entre nós não-terra '2' e '3' é injetada naturalmente no vetor de excitação nodal $z$.

<h5>Passo 2: Formulação das Equações Nodais (LKC)</h5>
Para o nó 1 (conectado diretamente à fonte ideal de tensão):
\[
V_1 = 20.000\text{ V}
\]

Para o nó 2: A corrente $I_s = 4.0\text{ A}$ sai do nó 2 em direção ao nó 3:
\[
\sum I_{sai} = 0 \implies \frac{V_2 - V_1}{R_1} + \frac{V_2}{R_2} + I_s = 0
\]
Substituindo os valores numéricos $V_1 = 20\text{ V}$, $R_1 = 6\ \Omega$, $R_2 = 10\ \Omega$ e $I_s = 4\text{ A}$:
\[
\frac{V_2 - 20}{6} + \frac{V_2}{10} + 4.0 = 0
\]
\[
V_2 \left( \frac{1}{6} + \frac{1}{10} \right) = \frac{20}{6} - 4.0 = \frac{10}{3} - \frac{12}{3} = -\frac{2}{3}\text{ A}
\]
Determinando o mínimo múltiplo comum: $\frac{1}{6} + \frac{1}{10} = \frac{5 + 3}{30} = \frac{8}{30} = \frac{4}{15}\text{ S}$.
\[
V_2 \cdot \frac{4}{15} = -\frac{2}{3} \implies V_2 = -\frac{2}{3} \cdot \frac{15}{4} = -\frac{30}{12} = -2.500\text{ V}
\]

Para o nó 3: A corrente $I_s = 4.0\text{ A}$ entra no nó 3 e se ramifica através de $R_3$ e $R_4$:
\[
I_s = \frac{V_3}{R_3} + \frac{V_3}{R_4} \implies 4.0 = V_3 \left( \frac{1}{4} + \frac{1}{2} \right) = V_3 \left( \frac{1 + 2}{4} \right) = V_3 \cdot \frac{3}{4}
\]
\[
V_3 = 4.0 \cdot \frac{4}{3} = \frac{16}{3} \approx 5.333333\text{ V}
\]

<h5>Passo 3: Determinação Rigorosa das Correntes de Ramo</h5>
Com o campo de potenciais eletrostáticos nodais unicamente determinado:
\[
I(R_1) = \frac{V_1 - V_2}{R_1} = \frac{20.0 - (-2.5)}{6.0} = \frac{22.5}{6.0} = 3.750\text{ A}
\]
\[
I(R_2) = \frac{V_2 - 0}{R_2} = \frac{-2.5}{10.0} = -0.250\text{ A}
\]
\[
I(R_3) = \frac{V_3 - 0}{R_3} = \frac{16/3}{4.0} = \frac{4}{3} \approx 1.333333\text{ A}
\]
\[
I(R_4) = \frac{V_3 - 0}{R_4} = \frac{16/3}{2.0} = \frac{8}{3} \approx 2.666667\text{ A}
\]

<h5>Passo 4: Verificação de Balanço Energético e Leis de Kirchhoff</h5>
- LKC no nó 2: $I(R_1) - I(R_2) - I_s = 3.750 - (-0.250) - 4.000 = 4.000 - 4.000 = 0.000\text{ A}$ (Verificado!).
- LKC no nó 3: $I_s - I(R_3) - I(R_4) = 4.000 - \frac{4}{3} - \frac{8}{3} = 4.000 - 4.000 = 0.000\text{ A}$ (Verificado!).
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, CurrentSource, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('V1', '1', '0', 20.0),\n"
            "    Resistor('R1', '1', '2', 6.0),\n"
            "    Resistor('R2', '2', '0', 10.0),\n"
            "    CurrentSource('Is', '3', '2', 4.0),\n"
            "    Resistor('R3', '3', '0', 4.0),\n"
            "    Resistor('R4', '3', '0', 2.0),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "print(f'V2 = {sol.V(\"2\"):.3f} V, V3 = {sol.V(\"3\"):.3f} V')\n"
            "print(f'I(R1) = {sol.I_resistor(\"R1\"):.3f} A, I(R2) = {sol.I_resistor(\"R2\"):.3f} A')"
        ),
        "run": run_ex44,
    },
    {
        "id": 45,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª/6ª Edição)",
        "chapter": "Capítulo 8 — Circuitos de Segunda Ordem",
        "title": "Resposta ao Degrau de Circuito RLC Série Superamortecido (Overdamped: α > ω₀)",
        "category": "Regime Transitório de 2ª Ordem",
        "enunciado": (
            "Determine analiticamente a forma de onda transitória completa da tensão no capacitor $v_C(t)$ e da corrente $i_L(t)$ "
            "em um circuito RLC série excitado por uma fonte degrau de tensão contínua $V_s = 24.0\\text{ V}$ em $t=0$, "
            "com parâmetros concentrados $R = 9.0\\ \\Omega$, $L = 1.0\\text{ H}$, $C = 0.05\\text{ F}$ e condições iniciais nulas "
            "($i_L(0) = 0\\text{ A}, v_C(0) = 0\\text{ V}$)."
        ),
        "topologia": [
            "Fonte independente degrau de tensão contínua $V_s = 24.0\\text{ V}$ aplicada em $t = 0$.",
            "Resistor linear em série $R = 9.0\\ \\Omega$ entre nós 1 e 2.",
            "Indutor ideal em série $L = 1.0\\text{ H}$ entre nós 2 e 3 com corrente inicial $i_L(0) = 0\\text{ A}$.",
            "Capacitor ideal em série $C = 0.05\\text{ F}$ entre nós 3 e 0 com tensão inicial $v_C(0) = 0\\text{ V}$."
        ],
        "deducao": r"""
<h5>Passo 1: Equação Diferencial Ordinária (EDO) de 2ª Ordem</h5>
Aplicando a Lei das Tensões de Kirchhoff à malha única para $t > 0$:
\[
R \cdot i(t) + L \frac{di(t)}{dt} + v_C(t) = V_s
\]
Como a relação constitutiva do capacitor estabelece $i(t) = C \frac{dv_C(t)}{dt}$, derivando em relação ao tempo:
\[
\frac{d^2 v_C(t)}{dt^2} + \frac{R}{L} \frac{dv_C(t)}{dt} + \frac{1}{L C} v_C(t) = \frac{V_s}{L C}
\]
Forma canônica:
\[
\frac{d^2 v_C(t)}{dt^2} + 2\alpha \frac{dv_C(t)}{dt} + \omega_0^2 v_C(t) = \omega_0^2 V_s
\]

<h5>Passo 2: Fatores Característicos e Regime de Amortecimento</h5>
O coeficiente neperiano de amortecimento $\alpha$ e a frequência natural não-amortecida $\omega_0$ são dados por:
\[
\alpha = \frac{R}{2 L} = \frac{9.0}{2 \cdot 1.0} = 4.500\text{ s}^{-1}
\]
\[
\omega_0 = \frac{1}{\sqrt{L C}} = \frac{1}{\sqrt{1.0 \cdot 0.05}} = \frac{1}{\sqrt{0.05}} = \sqrt{20} \approx 4.472136\text{ rad/s}
\]
Como $\alpha^2 = (4.5)^2 = 20.25 > \omega_0^2 = 20.00$, temos categoricamente $\alpha > \omega_0$, caracterizando um sistema <strong>Superamortecido (Overdamped)</strong> com duas raízes reais e distintas:
\[
s_{1,2} = -\alpha \pm \sqrt{\alpha^2 - \omega_0^2} = -4.5 \pm \sqrt{20.25 - 20.00} = -4.5 \pm \sqrt{0.25} = -4.5 \pm 0.5
\]
\[
s_1 = -4.000\text{ s}^{-1}, \quad s_2 = -5.000\text{ s}^{-1}
\]

<h5>Passo 3: Solução Geral e Aplicação das Condições de Contorno</h5>
A resposta completa compõe-se da resposta homogênea transitória somada à resposta particular forçada em regime permanente ($v_C(\infty) = V_s = 24.0\text{ V}$):
\[
v_C(t) = V_s + A_1 e^{s_1 t} + A_2 e^{s_2 t} = 24.0 + A_1 e^{-4.0 t} + A_2 e^{-5.0 t}
\]
Condição inicial de tensão ($t = 0^+$):
\[
v_C(0) = 24.0 + A_1 + A_2 = 0 \implies A_1 + A_2 = -24.0
\]
Condição inicial de corrente e derivada temporal ($t = 0^+$):
\[
i(0) = C \left. \frac{dv_C}{dt} \right|_{t=0} = 0 \implies \left. \frac{dv_C}{dt} \right|_{t=0} = s_1 A_1 + s_2 A_2 = -4.0 A_1 - 5.0 A_2 = 0
\]
Da equação de derivadas: $A_2 = -\frac{4.0}{5.0} A_1 = -0.8 A_1$.
Substituindo na primeira equação:
\[
A_1 - 0.8 A_1 = 0.2 A_1 = -24.0 \implies A_1 = -\frac{24.0}{0.2} = -120.000\text{ V}
\]
\[
A_2 = -0.8 \cdot (-120.0) = +96.000\text{ V}
\]

<h5>Passo 4: Expressões Analíticas Fechadas</h5>
\[
v_C(t) = 24.0 - 120.0 \, e^{-4.0 t} + 96.0 \, e^{-5.0 t}\quad [\text{V}]
\]
\[
i(t) = C \frac{dv_C}{dt} = 0.05 \cdot \left[ 480.0 \, e^{-4.0 t} - 480.0 \, e^{-5.0 t} \right] = 24.0 \cdot \left( e^{-4.0 t} - e^{-5.0 t} \right)\quad [\text{A}]
\]
Valores analíticos instantâneos de referência:
- Em $t = 0.5\text{ s}$: $v_C(0.5) = 24.0 - 120 e^{-2.0} + 96 e^{-2.5} \approx 24.0 - 16.2402 + 7.8802 = 15.6400\text{ V}$.
- Em $t = 1.0\text{ s}$: $v_C(1.0) = 24.0 - 120 e^{-4.0} + 96 e^{-5.0} \approx 24.0 - 2.1979 + 0.6468 = 22.4489\text{ V}$.
- Em $t \to \infty$: $v_C(\infty) = 24.000\text{ V}$.
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, Inductor, Capacitor, solve_transient\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', 24.0),\n"
            "    Resistor('R', '1', '2', 9.0),\n"
            "    Inductor('L', '2', '3', 1.0, ic=0.0),\n"
            "    Capacitor('C', '3', '0', 0.05, ic=0.0),\n"
            "])\n"
            "res = solve_transient(circuit, t_stop=2.0, dt=1e-3, uic=True)\n"
            "print(f'v_C(0.5s) = {res.V(\"3\")[500]:.3f} V, v_C(2.0s) = {res.V(\"3\")[-1]:.3f} V')"
        ),
        "run": run_ex45,
    },
    {
        "id": 46,
        "atol": 2e-4,
        "model_note": "Comparação transitória: atol=0,0002 V e rtol=0,5%, incluindo erro de discretização trapezoidal com dt=2 µs. No cruzamento por zero usa-se erro absoluto, pois erro relativo não está definido.",
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª/6ª Edição)",
        "chapter": "Capítulo 8 — Circuitos de Segunda Ordem",
        "title": "Resposta Natural de Circuito RLC Paralelo Criticamente Amortecido (Critically Damped: α = ω₀)",
        "category": "Regime Transitório de 2ª Ordem",
        "enunciado": (
            "Projete e determine a resposta natural transitória da tensão $v(t)$ em um circuito RLC paralelo operando no regime de "
            "amortecimento crítico (critically damped: $\\alpha = \\omega_0$), dado $C = 2.0\\ \\mu\\text{F}$, $L = 20.0\\text{ mH}$, "
            "com energia inicial acumulada $v_C(0) = 12.0\\text{ V}$ e $i_L(0) = 0.0\\text{ A}$."
        ),
        "topologia": [
            "Resistor linear em paralelo $R = 50.0\\ \\Omega$ entre nós 1 e 0 (terra).",
            "Capacitor ideal em paralelo $C = 2.0\\ \\mu\\text{F}$ com tensão inicial $v_C(0) = 12.0\\text{ V}$.",
            "Indutor ideal em paralelo $L = 20.0\\text{ mH}$ com corrente inicial $i_L(0) = 0.0\\text{ A}$."
        ],
        "deducao": r"""
<h5>Passo 1: Equação Característica do Circuito RLC Paralelo</h5>
Pela Lei das Correntes de Kirchhoff aplicada ao nó único para a resposta natural (sem fontes independentes):
\[
\frac{v(t)}{R} + \frac{1}{L} \int_{0}^{t} v(\tau) d\tau + i_L(0) + C \frac{dv(t)}{dt} = 0
\]
Diferenciando em relação ao tempo e dividindo por $C$:
\[
\frac{d^2 v(t)}{dt^2} + \frac{1}{R C} \frac{dv(t)}{dt} + \frac{1}{L C} v(t) = 0 \implies \frac{d^2 v(t)}{dt^2} + 2\alpha \frac{dv(t)}{dt} + \omega_0^2 v(t) = 0
\]
Para um circuito RLC paralelo:
\[
\alpha = \frac{1}{2 R C}, \quad \omega_0 = \frac{1}{\sqrt{L C}}
\]

<h5>Passo 2: Condição de Amortecimento Crítico</h5>
Para que o sistema seja criticamente amortecido ($\alpha = \omega_0$), as raízes da equação característica $s^2 + 2\alpha s + \omega_0^2 = 0$ devem ser reais e repetidas:
\[
s_1 = s_2 = -\alpha = -\omega_0
\]
Calculando $\omega_0$ com $L = 20\text{ mH} = 0.02\text{ H}$ e $C = 2\ \mu\text{F} = 2.0 \times 10^{-6}\text{ F}$:
\[
\omega_0 = \frac{1}{\sqrt{0.02 \cdot 2 \times 10^{-6}}} = \frac{1}{\sqrt{4 \times 10^{-8}}} = \frac{1}{2 \times 10^{-4}} = 5000.000\text{ rad/s}
\]
Impondo $\alpha = \omega_0 = 5000\text{ s}^{-1}$:
\[
R = \frac{1}{2 \alpha C} = \frac{1}{2 \cdot 5000 \cdot 2.0 \times 10^{-6}} = \frac{1}{0.02} = 50.000\ \Omega
\]

<h5>Passo 3: Solução Geral com Raízes Degeneradas</h5>
A forma analítica exata da resposta homogênea com raiz dupla $s = -\alpha$ é:
\[
v(t) = \left( A_1 + A_2 \cdot t \right) e^{-\alpha t} = \left( A_1 + A_2 \cdot t \right) e^{-5000 t}
\]
Determinando $A_1$ a partir de $v(0)$:
\[
v(0) = A_1 = 12.000\text{ V}
\]
Determinando $A_2$ a partir da derivada $v'(0)$:
Pela LKC no instante $t = 0^+$:
\[
i_C(0) = - i_R(0) - i_L(0) = -\frac{v(0)}{R} - 0 = -\frac{12.0}{50.0} = -0.240\text{ A}
\]
Logo:
\[
\left. \frac{dv}{dt} \right|_{t=0} = \frac{i_C(0)}{C} = \frac{-0.240}{2.0 \times 10^{-6}} = -120\,000.000\text{ V/s}
\]
Derivando a expressão geral da tensão:
\[
\frac{dv}{dt} = A_2 e^{-\alpha t} - \alpha (A_1 + A_2 t) e^{-\alpha t} \implies \left. \frac{dv}{dt} \right|_{t=0} = A_2 - \alpha A_1
\]
\[
-120\,000 = A_2 - 5000 \cdot 12.0 \implies A_2 = -120\,000 + 60\,000 = -60\,000.000\text{ V/s}
\]

<h5>Passo 4: Expressão Fechada e Cruzamento por Zero</h5>
\[
v(t) = (12.0 - 60\,000 \cdot t) \, e^{-5000 t}\quad [\text{V}]
\]
Propriedade fundamental do amortecimento crítico com condições iniciais de polaridades opostas:
\[
12.0 - 60\,000 \cdot t_{zero} = 0 \implies t_{zero} = \frac{12.0}{60\,000} = 0.000200\text{ s} = 0.200\text{ ms}
\]
A forma de onda cruza o zero exatamente em $t = 0.200\text{ ms}$, atinge um mínimo negativo e decai assintoticamente para zero sem qualquer oscilação periódica.
""",
        "python_code": (
            "from circuit_solver import Circuit, Resistor, Inductor, Capacitor, solve_transient\n\n"
            "circuit = Circuit([\n"
            "    Resistor('R', '1', '0', 50.0),\n"
            "    Capacitor('C', '1', '0', 2.0e-6, ic=12.0),\n"
            "    Inductor('L', '1', '0', 0.020, ic=0.0),\n"
            "])\n"
            "res = solve_transient(circuit, t_stop=0.0015, dt=2e-6, uic=True)\n"
            "print(f'v(0.1ms) = {res.V(\"1\")[50]:.3f} V, v(0.2ms zero) = {res.V(\"1\")[100]:.3f} V')"
        ),
        "run": run_ex46,
    },
    {
        "id": 47,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª/6ª Edição)",
        "chapter": "Capítulo 12 — Circuitos Trifásicos",
        "title": "Medição de Potência Trifásica pelo Método dos Dois Wattímetros (Teorema de Blondel)",
        "category": "Sistemas Trifásicos",
        "enunciado": (
            "Determine analiticamente e comprove computacionalmente as leituras de potência ativa $W_1$ e $W_2$ do Método dos Dois Wattímetros, "
            "bem como a potência ativa total $P_{3\\phi}$, potência reativa total $Q_{3\\phi}$, potência aparente $S_{3\\phi}$ e o fator de potência "
            "global em um sistema trifásico a 3 fios equilibrado (gerador Y conectado a carga em $\\Delta$), com tensão de fase da fonte "
            "$V_{LN} = 120.0\\text{ V}$ rms a $60\\text{ Hz}$ (sequência ABC) e carga em triângulo com impedância de ramo "
            "$Z_\\Delta = 24.0 + j18.0\\ \\Omega$."
        ),
        "topologia": [
            "Fonte trifásica balanceada em Estrela (Y) com neutro aterrado: $V_{LN} = 120.0\\text{ V}$ rms, $f = 60\\text{ Hz}$, sequência direta ABC.",
            "Carga trifásica balanceada em Triângulo (Δ) conectada às linhas A, B, C.",
            "Cada ramo de carga é composto por associação série $R = 24.0\\ \\Omega$ e $L = 18.0 / (2\\pi \\cdot 60) \\approx 47.75\\text{ mH}$.",
            "Wattímetro 1 instalado na linha A com bobina de potencial conectada entre linha A e C ($V_{ac}$).",
            "Wattímetro 2 instalado na linha B com bobina de potencial conectada entre linha B e C ($V_{bc}$)."
        ],
        "deducao": r"""
<h5>Passo 1: Teorema de Blondel e Método dos Dois Wattímetros</h5>
O Teorema de André Blondel (1893) estabelece que o número mínimo de wattímetros necessários para medir a potência total de um sistema polifásico de $N$ condutores é $N - 1$. Em um sistema trifásico a 3 fios sem neutro distribuído ($N = 3$), bastam exatamente 2 wattímetros conectados com bobinas de tensão referenciadas a uma linha comum (neste caso, linha C).

<h5>Passo 2: Tensões e Correntes Fasoriais de Linha e Fase</h5>
Sendo a sequência direta ABC e adotando $V_{an}$ como referência fasorial ($120.0 \angle 0^\circ\text{ V}$ rms):
\[
\mathbf{V}_{an} = 120.0 \angle 0^\circ\text{ V}, \quad \mathbf{V}_{bn} = 120.0 \angle -120^\circ\text{ V}, \quad \mathbf{V}_{cn} = 120.0 \angle 120^\circ\text{ V}
\]
As tensões de linha ($\mathbf{V}_{LL} = \sqrt{3} V_{LN} \angle +30^\circ$) são:
\[
\mathbf{V}_{ab} = 207.846 \angle 30^\circ\text{ V}, \quad \mathbf{V}_{bc} = 207.846 \angle -90^\circ\text{ V}, \quad \mathbf{V}_{ca} = 207.846 \angle 150^\circ\text{ V}
\]
A tensão entre linha A e linha C é dada por $\mathbf{V}_{ac} = -\mathbf{V}_{ca} = 207.846 \angle -30^\circ\text{ V}$.

A impedância de cada ramo da carga em triângulo:
\[
Z_\Delta = 24.0 + j18.0\ \Omega \implies |Z_\Delta| = \sqrt{24^2 + 18^2} = \sqrt{576 + 324} = \sqrt{900} = 30.000\ \Omega
\]
\[
\theta_\Delta = \arctan\left(\frac{18.0}{24.0}\right) = \arctan(0.75) \approx 36.8699^\circ \implies \mathbf{Z}_\Delta = 30.0 \angle 36.87^\circ\ \Omega
\]

Correntes de fase nos ramos delta:
\[
\mathbf{I}_{ab} = \frac{\mathbf{V}_{ab}}{Z_\Delta} = \frac{207.846 \angle 30^\circ}{30.0 \angle 36.87^\circ} = 6.9282 \angle -6.87^\circ\text{ A}
\]
\[
\mathbf{I}_{bc} = \frac{\mathbf{V}_{bc}}{Z_\Delta} = \frac{207.846 \angle -90^\circ}{30.0 \angle 36.87^\circ} = 6.9282 \angle -126.87^\circ\text{ A}
\]
\[
\mathbf{I}_{ca} = \frac{\mathbf{V}_{ca}}{Z_\Delta} = \frac{207.846 \angle 150^\circ}{30.0 \angle 36.87^\circ} = 6.9282 \angle 113.13^\circ\text{ A}
\]

Correntes de linha (LKC nos vértices da carga):
\[
\mathbf{I}_a = \mathbf{I}_{ab} - \mathbf{I}_{ca} = \sqrt{3} \cdot 6.9282 \angle (-6.87^\circ - 30^\circ) = 12.000 \angle -36.87^\circ\text{ A}
\]
\[
\mathbf{I}_b = \mathbf{I}_{bc} - \mathbf{I}_{ab} = 12.000 \angle -156.87^\circ\text{ A}
\]

<h5>Passo 3: Equações das Leituras dos Dois Wattímetros</h5>
A potência média indicada por cada wattímetro corresponde ao produto da tensão eficaz pela corrente eficaz e pelo cosseno do ângulo entre esses fasores:
\[
W_1 = \text{Re}(\mathbf{V}_{ac} \cdot \mathbf{I}_a^*) = |\mathbf{V}_{ac}| \cdot |\mathbf{I}_a| \cos(\theta_{Vac} - \theta_{Ia})
\]
\[
\theta_{Vac} - \theta_{Ia} = -30.0^\circ - (-36.8699^\circ) = +6.8699^\circ
\]
\[
W_1 = 207.846 \cdot 12.0 \cdot \cos(6.8699^\circ) = 2494.15 \cdot 0.9928 = 2476.3\text{ W}
\]

Para o Wattímetro 2:
\[
W_2 = \text{Re}(\mathbf{V}_{bc} \cdot \mathbf{I}_b^*) = |\mathbf{V}_{bc}| \cdot |\mathbf{I}_b| \cos(\theta_{Vbc} - \theta_{Ib})
\]
\[
\theta_{Vbc} - \theta_{Ib} = -90.0^\circ - (-156.8699^\circ) = +66.8699^\circ
\]
\[
W_2 = 207.846 \cdot 12.0 \cdot \cos(66.8699^\circ) = 2494.15 \cdot 0.3928 = 979.7\text{ W}
\]

<h5>Passo 4: Síntese de Potências Totais do Sistema</h5>
Potência Ativa Total:
\[
P_{3\phi} = W_1 + W_2 = 2476.3 + 979.7 = 3456.000\text{ W}
\]
Verificação direta per-fase: $P = 3 \cdot |\mathbf{I}_\Delta|^2 \cdot R_\Delta = 3 \cdot (6.9282)^2 \cdot 24.0 = 3 \cdot 48.0 \cdot 24.0 = 3456.000\text{ W}$!

Potência Reativa Total:
\[
Q_{3\phi} = \sqrt{3} (W_1 - W_2) = \sqrt{3} \cdot (2476.3 - 979.7) = 1.73205 \cdot 1496.6 = 2592.000\text{ VAR}
\]
Verificação direta: $Q = 3 \cdot |\mathbf{I}_\Delta|^2 \cdot X_\Delta = 3 \cdot 48.0 \cdot 18.0 = 2592.000\text{ VAR}$!

Potência Aparente Total e Fator de Potência:
\[
S_{3\phi} = \sqrt{P_{3\phi}^2 + Q_{3\phi}^2} = \sqrt{3456^2 + 2592^2} = 4320.000\text{ VA}
\]
\[
\text{FP} = \frac{P_{3\phi}}{S_{3\phi}} = \frac{3456.0}{4320.0} = 0.8000\quad (\text{indutivo})
\]
""",
        "python_code": (
            "from circuit_solver import (\n"
            "    Circuit, build_three_phase_source, build_delta_load,\n"
            "    solve_ac, analyze_two_wattmeters\n"
            ")\n\n"
            "sources = build_three_phase_source('Y', V_rms=120.0, sequence='ABC', node_a='s_a', node_b='s_b', node_c='s_c', node_neutral='0')\n"
            "load = build_delta_load(R=24.0, L=18.0/(2*3.14159*60), node_a='s_a', node_b='s_b', node_c='s_c')\n"
            "circuit = Circuit(sources + load)\n"
            "sol = solve_ac(circuit, frequency=60.0)\n"
            "wm = analyze_two_wattmeters(sol.V('s_a')-sol.V('s_c'), sol.V('s_b')-sol.V('s_c'), (sol.V('s_a')-sol.V('s_b'))/(24+18j), (sol.V('s_b')-sol.V('s_c'))/(24+18j))\n"
            "print(f'P_tot = {wm.P_total:.1f} W, Q_tot = {wm.Q_total:.1f} VAR, FP = {wm.power_factor:.3f}')"
        ),
        "run": run_ex47,
    },
    {
        "id": 48,
        "book": "Hayt, Kemmerly & Durbin",
        "book_title": "Análise de Circuitos em Engenharia (8ª/9ª Edição)",
        "chapter": "Capítulo 17 — Redes de Duas Portas (Two-Port Networks)",
        "title": "Extração Completa dos Parâmetros de Impedância (Z) e Transmissão (ABCD) de Quadripolo em T",
        "category": "Quadripolos & Redes de Duas Portas",
        "enunciado": (
            "Determine analiticamente e extraia computacionalmente as matrizes completas de parâmetros de Impedância ($Z$) e "
            "parâmetros de Transmissão ($ABCD$ ou $T$) de uma rede de duas portas puramente resistiva em topologia T assimétrica "
            "com $R_1 = 20.0\\ \\Omega$ (braço série de entrada), $R_2 = 40.0\\ \\Omega$ (braço série de saída) e $R_3 = 30.0\\ \\Omega$ "
            "(braço shunt comum). Demonstre as propriedades de reciprocidade e assimetria, e calcule a impedância de entrada $Z_{in}$ "
            "quando terminada em uma carga resistiva $R_L = 50.0\\ \\Omega$."
        ),
        "topologia": [
            "Porta 1 (Entrada): terminais 'n1' (+) e '0' (-).",
            "Porta 2 (Saída): terminais 'n2' (+) e '0' (-).",
            "Braço série 1: Resistor linear $R_1 = 20.0\\ \\Omega$ entre nó 'n1' e nó intermediário 'mid'.",
            "Braço série 2: Resistor linear $R_2 = 40.0\\ \\Omega$ entre nó 'n2' e nó intermediário 'mid'.",
            "Braço shunt: Resistor linear $R_3 = 30.0\\ \\Omega$ entre nó intermediário 'mid' e terra comum '0'."
        ],
        "deducao": r"""
<h5>Passo 1: Fundamentação das Equações de Parâmetros Z</h5>
Um quadripolo linear e invariante no tempo é inteiramente descrito em termos de suas tensões e correntes de porta pelas equações canônicas de impedância de malha:
\[
\begin{bmatrix} V_1 \\ V_2 \end{bmatrix} = \begin{bmatrix} z_{11} & z_{12} \\ z_{21} & z_{22} \end{bmatrix} \begin{bmatrix} I_1 \\ I_2 \end{bmatrix}
\]
Onde as definições operacionais com portas em circuito aberto são:
\[
z_{11} = \left. \frac{V_1}{I_1} \right|_{I_2 = 0}, \quad z_{12} = \left. \frac{V_1}{I_2} \right|_{I_1 = 0}, \quad z_{21} = \left. \frac{V_2}{I_1} \right|_{I_2 = 0}, \quad z_{22} = \left. \frac{V_2}{I_2} \right|_{I_1 = 0}
\]

<h5>Passo 2: Dedução Analítica dos Parâmetros Z</h5>
1. Com a Porta 2 aberta ($I_2 = 0$) e corrente $I_1$ injetada na Porta 1:
   Toda a corrente $I_1$ flui através de $R_1$ e desce por $R_3$ para a referência:
   \[
   V_1 = (R_1 + R_3) \cdot I_1 = (20.0 + 30.0) \cdot I_1 = 50.0 \cdot I_1 \implies z_{11} = 50.000\ \Omega
   \]
   Como não há corrente em $R_2$ ($I_2 = 0$), a tensão na Porta 2 é exatamente a tensão no nó intermediário 'mid':
   \[
   V_2 = V_{mid} = R_3 \cdot I_1 = 30.0 \cdot I_1 \implies z_{21} = 30.000\ \Omega
   \]

2. Com a Porta 1 aberta ($I_1 = 0$) e corrente $I_2$ injetada na Porta 2:
   Toda a corrente $I_2$ flui através de $R_2$ e desce por $R_3$:
   \[
   V_2 = (R_2 + R_3) \cdot I_2 = (40.0 + 30.0) \cdot I_2 = 70.0 \cdot I_2 \implies z_{22} = 70.000\ \Omega
   \]
   A tensão na Porta 1 aberta é a tensão de 'mid':
   \[
   V_1 = V_{mid} = R_3 \cdot I_2 = 30.0 \cdot I_2 \implies z_{12} = 30.000\ \Omega
   \]

Matriz de Impedância Resultante:
\[
\mathbf{Z} = \begin{bmatrix} 50.0 & 30.0 \\ 30.0 & 70.0 \end{bmatrix}\ \Omega
\]
Determinante de $\mathbf{Z}$:
\[
\det(\mathbf{Z}) = z_{11} z_{22} - z_{12} z_{21} = 50.0 \cdot 70.0 - 30.0 \cdot 30.0 = 3500.0 - 900.0 = 2600.000\ \Omega^2
\]

Auditoria de Simetria e Reciprocidade:
- <strong>Reciprocidade</strong>: Como $z_{12} = z_{21} = 30.0\ \Omega$, o circuito é estritamente recíproco (composto apenas por elementos bilaterais passivos).
- <strong>Simetria</strong>: Como $z_{11} = 50.0\ \Omega \neq z_{22} = 70.0\ \Omega$, o circuito é geometricamente e eletricamente <em>assimétrico</em>.

<h5>Passo 3: Conversão para Parâmetros de Transmissão / ABCD (Matriz T)</h5>
As relações de transmissão expressam as grandezas da porta 1 em função da porta 2 (com convenção de corrente de saída saindo do quadripolo):
\[
\begin{bmatrix} V_1 \\ I_1 \end{bmatrix} = \begin{bmatrix} A & B \\ C & D \end{bmatrix} \begin{bmatrix} V_2 \\ -I_2 \end{bmatrix}
\]
Fórmulas canônicas de conversão de $\mathbf{Z}$ para $\mathbf{T}$:
\[
A = \frac{z_{11}}{z_{21}} = \frac{50.0}{30.0} = \frac{5}{3} \approx 1.666667\quad [\text{adimensional}]
\]
\[
B = \frac{\det(\mathbf{Z})}{z_{21}} = \frac{2600.0}{30.0} = \frac{260}{3} \approx 86.666667\ \Omega
\]
\[
C = \frac{1}{z_{21}} = \frac{1}{30.0} \approx 0.033333\text{ S}
\]
\[
D = \frac{z_{22}}{z_{21}} = \frac{70.0}{30.0} = \frac{7}{3} \approx 2.333333\quad [\text{adimensional}]
\]
Condição de reciprocidade nos parâmetros ABCD:
\[
A D - B C = \left(\frac{5}{3}\right) \left(\frac{7}{3}\right) - \left(\frac{260}{3}\right) \left(\frac{1}{30}\right) = \frac{35}{9} - \frac{26}{9} = \frac{9}{9} = 1.000000\quad (\text{Comprovado!})
\]

<h5>Passo 4: Cálculo da Impedância de Entrada sob Carga $R_L = 50.0\ \Omega$</h5>
A impedância de entrada $Z_{in} = \frac{V_1}{I_1}$ quando a porta 2 é terminada por $R_L$ ($V_2 = -I_2 \cdot R_L$) é calculada matricialmente por:
\[
Z_{in} = \frac{A R_L + B}{C R_L + D} = \frac{\frac{5}{3} \cdot 50.0 + \frac{260}{3}}{\frac{1}{30} \cdot 50.0 + \frac{7}{3}} = \frac{\frac{250 + 260}{3}}{\frac{5}{3} + \frac{7}{3}} = \frac{\frac{510}{3}}{\frac{12}{3}} = \frac{510.0}{12.0} = 42.500\ \Omega
\]
Verificação direta por redução elementar série-paralelo:
\[
Z_{in} = R_1 + [R_3 \parallel (R_2 + R_L)] = 20.0 + [30.0 \parallel (40.0 + 50.0)] = 20.0 + [30.0 \parallel 90.0]
\]
\[
30.0 \parallel 90.0 = \frac{30.0 \cdot 90.0}{30.0 + 90.0} = \frac{2700.0}{120.0} = 22.500\ \Omega \implies Z_{in} = 20.0 + 22.5 = 42.500\ \Omega
\]
A equivalência analítica e computacional é exata até a última casa decimal.
""",
        "python_code": (
            "from circuit_solver import Circuit, Resistor, analyze_two_port\n\n"
            "circuit = Circuit([\n"
            "    Resistor('R1', 'n1', 'mid', 20.0),\n"
            "    Resistor('R2', 'n2', 'mid', 40.0),\n"
            "    Resistor('R3', 'mid', '0', 30.0),\n"
            "], allowed_open_nodes=['n1', 'n2'])\n"
            "res = analyze_two_port(circuit, port1=('n1', '0'), port2=('n2', '0'))\n"
            "print(f'Z = {res.Z}')\n"
            "print(f'ABCD = {res.ABCD}')\n"
            "print(f'Zin(RL=50) = {res.input_impedance(Z_load=50.0):.3f} ohms')"
        ),
        "run": run_ex48,
    },
]

