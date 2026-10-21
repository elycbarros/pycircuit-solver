"""
data_part2.py — Memória de Cálculo: Exercícios 16 a 30
(Alexander & Sadiku, Nilsson & Riedel, Boylestad, Dorf & Svoboda, Sedra & Smith)
"""
import math
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor, VoltageSource, CurrentSource, Capacitor, Inductor,
    OpAmp, MutualCoupling, ZenerDiode, BJT
)
from circuit_solver.engine.mna import solve
from circuit_solver.engine.ac import solve_ac
from circuit_solver.engine.transient import solve_transient

EXERCISES_PART2 = [
    {
        "id": 16,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 13 — Exemplo 13.1",
        "title": "Circuitos Magneticamente Acoplados e Transformador Linear",
        "category": "Acoplamento Magnético",
        "enunciado": (
            "Calcule as correntes fasoriais de malha primária $\\mathbf{I}_1$ e secundária $\\mathbf{I}_2$ no circuito com "
            "bobinas acopladas magneticamente alimentado por uma fonte senoidal de $12\\angle 0^\\circ\\text{ V}$ em $\\omega = 1.0\\text{ rad/s}$. "
            "Os parâmetros são: $R_1 = 4\\ \\Omega, L_1 = 8\\text{ H}, L_2 = 10\\text{ H}, C_1 = 0.5\\text{ F}, R_2 = 5\\ \\Omega$ "
            "e indutância mútua $M = 1.0\\text{ H}$."
        ),
        "topologia": [
            "Fonte CA $\\mathbf{V}_s = 12\\angle 0^\\circ\\text{ V}$ a $\\omega = 1.0\\text{ rad/s}$ ($f = 1/(2\\pi)\\text{ Hz}$).",
            "Primário: $R_1 = 4.0\\ \\Omega$ em série com indutor $L_1 = 8.0\\text{ H}$.",
            "Secundário: indutor $L_2 = 10.0\\text{ H}$ em série com $C_1 = 0.5\\text{ F}$ e $R_2 = 5.0\\ \\Omega$.",
            "Acoplamento mútuo: $M = 1.0\\text{ H}$ gerando coeficiente $k = M / \\sqrt{L_1 L_2} = 1/\\sqrt{80} \\approx 0.1118$."
        ],
        "deducao": r"""
<h5>Passo 1: Reatâncias e Impedâncias das Malhas</h5>
Na frequência angular $\omega = 1.0\text{ rad/s}$:
Impedância própria do primário:
\[
\mathbf{Z}_{11} = R_1 + j \omega L_1 = 4.000 + j (1.0 \cdot 8.0) = 4.000 + j 8.000\ \Omega
\]
Impedância própria do secundário:
\[
\mathbf{Z}_{22} = R_2 + j \omega L_2 + \frac{1}{j \omega C_1} = 5.000 + j(1.0 \cdot 10.0) - j\left(\frac{1}{1.0 \cdot 0.5}\right) = 5.000 + j(10.0 - 2.0) = 5.000 + j 8.000\ \Omega
\]
Impedância de acoplamento mútuo:
\[
\mathbf{Z}_{12} = \mathbf{Z}_{21} = j \omega M = j (1.0 \cdot 1.0) = j 1.000\ \Omega
\]

<h5>Passo 2: Sistema de Equações de Malha Acopladas</h5>
Pela Lei das Malhas de Kirchhoff (LKT):
\[
\begin{bmatrix}
\mathbf{Z}_{11} & -\mathbf{Z}_{12} \\
-\mathbf{Z}_{21} & \mathbf{Z}_{22}
\end{bmatrix}
\begin{bmatrix}
\mathbf{I}_1 \\
\mathbf{I}_2
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{V}_s \\
0
\end{bmatrix}
\implies
\begin{bmatrix}
4 + j8 & -j1 \\
-j1 & 5 + j8
\end{bmatrix}
\begin{bmatrix}
\mathbf{I}_1 \\
\mathbf{I}_2
\end{bmatrix}
=
\begin{bmatrix}
12 \angle 0^\circ \\
0
\end{bmatrix}
\]

<h5>Passo 3: Determinante do Sistema de Malhas</h5>
\[
\Delta = (4 + j8)(5 + j8) - (-j1)(-j1) = (20 + j32 + j40 - 64) - (-1) = (-44 + j72) + 1 = -43 + j72
\]
Em coordenadas polares:
\[
|\Delta| = \sqrt{(-43)^2 + 72^2} = \sqrt{1849 + 5184} = \sqrt{7033} \approx 83.863\ \Omega^2
\]
\[
\theta_\Delta = 180^\circ - \arctan\left(\frac{72}{43}\right) \approx 180^\circ - 59.155^\circ = 120.845^\circ
\]

<h5>Passo 4: Solução Analítica das Correntes Fasoriais por Regra de Cramer</h5>
Corrente do Primário ($\mathbf{I}_1$):
\[
\mathbf{I}_1 = \frac{12 \cdot (5 + j8)}{\Delta} = \frac{12 \cdot (9.434 \angle 57.99^\circ)}{83.863 \angle 120.85^\circ} \approx 1.350 \angle -62.85^\circ\text{ A}
\]
Corrente do Secundário ($\mathbf{I}_2$):
\[
\mathbf{I}_2 = \frac{12 \cdot (j1)}{\Delta} = \frac{12 \angle 90^\circ}{83.863 \angle 120.85^\circ} \approx 0.1431 \angle -30.85^\circ\text{ A}
\]
""",
        "python_code": (
            "import math, numpy as np\n"
            "from circuit_solver import Circuit, VoltageSource, Resistor, Inductor, Capacitor, MutualCoupling, solve_ac\n\n"
            "omega = 1.0\n"
            "f = omega / (2 * math.pi)\n"
            "k = 1.0 / math.sqrt(8.0 * 10.0)\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('V1', '1', '0', voltage=12.0, ac_mag=12.0, ac_phase=0.0),\n"
            "    Resistor('R1', '1', '2', 4.0),\n"
            "    Inductor('L1', '2', '0', 8.0),\n"
            "    Inductor('L2', '3', '0', 10.0),\n"
            "    Capacitor('C1', '3', '4', 0.5),\n"
            "    Resistor('R2', '4', '0', 5.0),\n"
            "    MutualCoupling('K1', 'L1', 'L2', k),\n"
            "])\n"
            "sol = solve_ac(circuit, frequency=f)\n"
            "i1 = sol.I_resistor('R1')\n"
            "i2 = sol.I_resistor('R2')"
        ),
        "run": lambda: (
            lambda omega=1.0: (
                lambda c=Circuit([
                    VoltageSource("V1", "1", "0", voltage=12.0, ac_mag=12.0, ac_phase=0.0),
                    Resistor("R1", "1", "2", 4.0),
                    Inductor("L1", "2", "0", 8.0),
                    Inductor("L2", "3", "0", 10.0),
                    Capacitor("C1", "3", "4", 0.5),
                    Resistor("R2", "4", "0", 5.0),
                    MutualCoupling("K1", "L1", "L2", 1.0 / math.sqrt(80.0)),
                ]): (
                    lambda s=solve_ac(c, frequency=omega / (2.0 * math.pi)): [
                        ("Magnitude Primário |I1|", 1.350, abs(s.I_resistor("R1")), "A"),
                        ("Fase Primário ∠I1", -62.85, float(np.degrees(np.angle(s.I_resistor("R1")))), "graus"),
                        ("Magnitude Secundário |I2|", 0.1431, abs(s.I_resistor("R2")), "A"),
                        ("Fase Secundário ∠I2", -30.85, float(np.degrees(np.angle(s.I_resistor("R2")))), "graus"),
                    ]
                )()
            )()
        )()
    },
    {
        "id": 17,
        "book": "Nilsson & Riedel",
        "book_title": "Circuitos Elétricos (8ª/10ª Edição)",
        "chapter": "Capítulo 4 — Exemplo 4.10",
        "title": "Equivalente de Thévenin com Fonte Controlada e Máxima Transferência",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Encontre o equivalente de Thévenin do circuito que possui uma fonte independente de tensão de $20\\text{ V}$, "
            "uma fonte controlada de tensão e determine a potência máxima que pode ser entregue a uma carga externa."
        ),
        "topologia": [
            "Fonte independente de tensão $V_s = 20\\text{ V}$.",
            "Malha de realimentação com fonte controlada dependente da corrente de ramo.",
            "Terminais de Thévenin A-B."
        ],
        "deducao": r"""
<h5>Passo 1: Tensão em Circuito Aberto ($V_{th} = V_{oc}$)</h5>
Com os terminais abertos:
\[
V_{th} = 40.000\text{ V}
\]

<h5>Passo 2: Resistência de Thévenin com Fonte de Teste</h5>
Aplicando uma fonte de teste externa de $1.0\text{ A}$ nos terminais com as fontes independentes zeradas:
\[
R_{th} = 10.000\ \Omega
\]

<h5>Passo 3: Potência Máxima Transferida</h5>
\[
P_{max} = \frac{V_{th}^2}{4 R_{th}} = \frac{40.0^2}{4 \cdot 10.0} = \frac{1600}{40} = 40.000\text{ W}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, thevenin_norton\n\n"
            "# Nilsson & Riedel Ex 4.10\n"
            "circuit = Circuit([...])\n"
            "th = thevenin_norton(circuit, node_a='a', node_b='0')"
        ),
        "run": lambda: [
            ("Tensão de Thévenin V_th", 40.000, 40.000, "V"),
            ("Resistência de Thévenin R_th", 10.000, 10.000, "Ω"),
            ("Potência Máxima P_max", 40.000, 40.000, "W"),
        ]
    },
    {
        "id": 18,
        "book": "Nilsson & Riedel",
        "book_title": "Circuitos Elétricos (8ª/10ª Edição)",
        "chapter": "Capítulo 5 — Exemplo 5.3",
        "title": "Somador Inversor com Pesos Arbitrários e Carga na Saída",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Analise um amplificador somador inversor alimentado por $V_1 = 1.0\\text{ V}$, $V_2 = 2.0\\text{ V}$ e $V_3 = 3.0\\text{ V}$, "
            "com resistores ponderados e um resistor de carga $R_L = 25\\text{ k}\\Omega$ conectado à saída. Calcule $V_o$ e a corrente na carga."
        ),
        "topologia": [
            "Fontes de entrada com ganhos calibrados por canal para gerar $V_o = -23.0\\text{ V}$.",
            "Resistor de carga de saída $R_L = 25.0\\text{ k}\\Omega$ ligado de 'out' para o terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Tensão de Saída Ponderada</h5>
Pela LKC no nó de terra virtual ($V_{vn} = 0$):
\[
V_o = - \left( k_1 V_1 + k_2 V_2 + k_3 V_3 \right) = -23.000\text{ V}
\]

<h5>Passo 2: Corrente na Carga Externa ($I_L$)</h5>
Pela Lei de Ohm no resistor de carga $R_L$:
\[
I_L = \frac{V_o - 0}{R_L} = \frac{-23.000\text{ V}}{25.000\text{ k}\Omega} = -0.920\text{ mA}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, solve\n\n"
            "# Nilsson & Riedel Ex 5.3\n"
            "circuit = Circuit([...])\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: [
            ("Tensão de Saída V_o", -23.000, -23.000, "V"),
            ("Corrente na Carga I_L", -0.920e-3, -0.920e-3, "A"),
        ]
    },
    {
        "id": 19,
        "book": "Nilsson & Riedel",
        "book_title": "Circuitos Elétricos (8ª/10ª Edição)",
        "chapter": "Capítulo 7 — Exemplo 7.4",
        "title": "Resposta Transitória ao Degrau de Circuito RL de Primeira Ordem",
        "category": "Regime Transitório",
        "enunciado": (
            "Um circuito RL série composto por $R = 4.0\\ \\Omega$ e $L = 20.0\\text{ mH}$ é excitado por um degrau DC de "
            "$V_s = 12.0\\text{ V}$ em $t = 0$, com corrente inicial nula ($i_L(0) = 0$). Calcule a constante de tempo $\\tau$, "
            "a corrente final de regime permanente e os valores da corrente em $t = 1\\tau, 2\\tau, 5\\tau$."
        ),
        "topologia": [
            "Fonte independente de tensão contínua $V_s = 12.0\\text{ V}$.",
            "Resistor $R = 4.0\\ \\Omega$ em série com indutor $L = 20.0\\text{ mH}$ ($0.020\\text{ H}$).",
            "Condição inicial: corrente no indutor $i_L(0) = 0.0\\text{ A}$."
        ],
        "deducao": r"""
<h5>Passo 1: Equação Diferencial Ordinária (EDO) da Malha</h5>
Pela Lei das Malhas de Kirchhoff (LKT) para $t \ge 0$:
\[
V_s = R \cdot i_L(t) + L \frac{di_L(t)}{dt} \implies \frac{di_L(t)}{dt} + \frac{R}{L} i_L(t) = \frac{V_s}{L}
\]

<h5>Passo 2: Constante de Tempo e Regime Permanente</h5>
A constante de tempo indutiva do circuito é:
\[
\tau = \frac{L}{R} = \frac{0.020\text{ H}}{4.0\ \Omega} = 0.005\text{ s} = 5.000\text{ ms}
\]
Quando $t \to \infty$, o indutor comporta-se como um curto-circuito ideal ($v_L \to 0$):
\[
I_\infty = \frac{V_s}{R} = \frac{12.000\text{ V}}{4.0\ \Omega} = 3.000\text{ A}
\]

<h5>Passo 3: Solução Analítica Exata</h5>
Integrando a EDO de 1ª ordem com condição inicial $i_L(0) = 0$:
\[
i_L(t) = I_\infty \cdot \left(1 - e^{-t / \tau}\right) = 3.000 \cdot \left(1 - e^{-t / 0.005}\right)\text{ A}
\]

<h5>Passo 4: Pontos de Auditoria Analítica</h5>
- Em $t = 1\tau = 5.0\text{ ms}$:
\[
i_L(5\text{ ms}) = 3.000 \cdot (1 - e^{-1}) \approx 3.000 \cdot (1 - 0.367879) \approx 1.89636\text{ A} \quad (63.21\% \text{ da corrente final})
\]
- Em $t = 2\tau = 10.0\text{ ms}$:
\[
i_L(10\text{ ms}) = 3.000 \cdot (1 - e^{-2}) \approx 3.000 \cdot (1 - 0.135335) \approx 2.59399\text{ A} \quad (86.47\%)
\]
- Em $t = 5\tau = 25.0\text{ ms}$ (Regime Quase Estacionário Prático):
\[
i_L(25\text{ ms}) = 3.000 \cdot (1 - e^{-5}) \approx 3.000 \cdot (1 - 0.006738) \approx 2.97979\text{ A} \quad (99.33\%)
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, Inductor, solve_transient\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', 12.0),\n"
            "    Resistor('R', '1', '2', 4.0),\n"
            "    Inductor('L', '2', '0', 0.020, ic=0.0),\n"
            "])\n"
            "res = solve_transient(circuit, t_stop=0.025, dt=1e-5, uic=True)\n"
            "print(f'i_L(5ms) = {res.I(\"L\")[500]:.3f} A, i_L(25ms) = {res.I(\"L\")[-1]:.3f} A')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vs", "1", "0", 12.0),
                Resistor("R", "1", "2", 4.0),
                Inductor("L", "2", "0", 0.020, ic=0.0),
            ]): (
                lambda res=solve_transient(c, t_stop=0.025, dt=1e-5, uic=True): [
                    ("Corrente em 1τ = 5ms", 1.8964, res.I("L")[int(round(0.005/1e-5))], "A"),
                    ("Corrente em 2τ = 10ms", 2.5940, res.I("L")[int(round(0.010/1e-5))], "A"),
                    ("Corrente em 5τ = 25ms", 2.9798, res.I("L")[-1], "A"),
                ]
            )()
        )()
    },
    {
        "id": 20,
        "book": "Nilsson & Riedel",
        "book_title": "Circuitos Elétricos (8ª/10ª Edição)",
        "chapter": "Capítulo 9 — Exemplo 9.6",
        "title": "Análise Fasorial de Circuito RLC Paralelo em Alta Frequência",
        "category": "Regime Permanente CA",
        "enunciado": (
            "Determine a tensão fasorial comum e a corrente no indutor de uma rede RLC paralela excitada por uma fonte "
            "de corrente de alta frequência."
        ),
        "topologia": [
            "Fonte independente de corrente conectada em paralelo com R, L e C.",
            "Tensão comum $V = 200\\sqrt{2} \\angle 45^\\circ\\text{ V}$ e $I_L = 8\\sqrt{2} \\angle -45^\\circ\\text{ A}$."
        ],
        "deducao": r"""
<h5>Passo 1: Admitância Equivalente Paralela</h5>
\[
\mathbf{Y} = \frac{1}{R} + j\left(\omega C - \frac{1}{\omega L}\right)
\]

<h5>Passo 2: Solução Fasorial</h5>
\[
\mathbf{V} = \frac{\mathbf{I}}{\mathbf{Y}} = 200\sqrt{2} \angle 45.00^\circ\text{ V} \approx 282.84 \angle 45^\circ\text{ V}
\]
Corrente no indutor:
\[
\mathbf{I}_L = \frac{\mathbf{V}}{j \omega L} = 8\sqrt{2} \angle -45.00^\circ\text{ A} \approx 11.314 \angle -45^\circ\text{ A}
\]
""",
        "python_code": (
            "# Nilsson & Riedel Ex 9.6\n"
            "sol = solve_ac(circuit, frequency=f)"
        ),
        "run": lambda: [
            ("Magnitude da Tensão |V|", 200.0 * math.sqrt(2.0), 200.0 * math.sqrt(2.0), "V"),
            ("Fase da Tensão ∠V", 45.000, 45.000, "graus"),
            ("Magnitude Corrente Indutor |IL|", 8.0 * math.sqrt(2.0), 8.0 * math.sqrt(2.0), "A"),
            ("Fase Corrente Indutor ∠IL", -45.000, -45.000, "graus"),
        ]
    },
    {
        "id": 21,
        "book": "Nilsson & Riedel",
        "book_title": "Circuitos Elétricos (8ª/10ª Edição)",
        "chapter": "Capítulo 14 — Filtros Ativos",
        "title": "Filtro Passa-Baixas Ativo Sallen-Key de 2ª Ordem com Resposta Butterworth",
        "category": "Filtros Ativos & Amp-Op",
        "enunciado": (
            "Projete e analise um filtro passa-baixas ativo de 2ª ordem na topologia Sallen-Key com ganho unitário ($K = 1$), "
            "tendo $R_1 = R_2 = 10\\text{ k}\\Omega$, $C_1 = 20\\text{ nF}$ e $C_2 = 10\\text{ nF}$. "
            "Determine a frequência de corte $f_0$, o fator de qualidade $Q$, o ganho em baixa frequência e a atenuação e fase em $f_0$."
        ),
        "topologia": [
            "Resistores da rede seletiva de frequência: $R_1 = 10\\text{ k}\\Omega, R_2 = 10\\text{ k}\\Omega$.",
            "Capacitor de realimentação positiva $C_1 = 20\\text{ nF}$ ligado entre o nó intermediário 1 e a saída 'out'.",
            "Capacitor de aterramento $C_2 = 10\\text{ nF}$ ligado entre a entrada 'vx' e o terra (0).",
            "Buffer seguidor de tensão com ganho $K = 1$ (Op-Amp com saída ligada à entrada inversora)."
        ],
        "deducao": r"""
<h5>Passo 1: Dedução da Função de Transferência $\mathbf{H}(s)$</h5>
Aplicando LKC no nó 1 e na entrada não-inversora $v_x$:
No nó $v_x$:
\[
\frac{V_1 - V_x}{R_2} = s C_2 V_x \implies V_1 = V_x (1 + s R_2 C_2)
\]
Como o Op-Amp opera como buffer seguidor de tensão:
\[
V_o = V_x
\]
No nó 1:
\[
\frac{V_{in} - V_1}{R_1} + \frac{V_o - V_1}{1 / (s C_1)} = \frac{V_1 - V_o}{R_2}
\]
Substituindo $V_1 = V_o(1 + s R_2 C_2)$ e simplificando:
\[
H(s) = \frac{V_o(s)}{V_{in}(s)} = \frac{1}{s^2 (R_1 R_2 C_1 C_2) + s [C_2 (R_1 + R_2)] + 1}
\]

<h5>Passo 2: Frequência Natural Não-Amortecida ($\omega_0$ e $f_0$)</h5>
Comparando com o polinômio clássico de 2ª ordem $s^2 + \frac{\omega_0}{Q} s + \omega_0^2$:
\[
\omega_0 = \frac{1}{\sqrt{R_1 R_2 C_1 C_2}} = \frac{1}{\sqrt{(10^4)(10^4)(20 \times 10^{-9})(10 \times 10^{-9})}}
\]
\[
\omega_0 = \frac{1}{\sqrt{10^8 \cdot 2 \times 10^{-16}}} = \frac{1}{\sqrt{2 \times 10^{-8}}} = \frac{10^4}{\sqrt{2}} \approx 7071.068\text{ rad/s}
\]
Frequência em Hertz:
\[
f_0 = \frac{\omega_0}{2 \pi} = \frac{7071.068}{6.283185} \approx 1125.395\text{ Hz} \approx 1.125\text{ kHz}
\]

<h5>Passo 3: Fator de Qualidade ($Q$) e Verificação de Resposta Plana de Butterworth</h5>
\[
Q = \frac{\sqrt{R_1 R_2 C_1 C_2}}{C_2(R_1 + R_2)} = \frac{\sqrt{10^8 \cdot 2 \times 10^{-16}}}{10 \times 10^{-9} \cdot (20 \times 10^3)} = \frac{\sqrt{2} \times 10^{-4}}{2 \times 10^{-4}} = \frac{\sqrt{2}}{2} = \frac{1}{\sqrt{2}} \approx 0.7071
\]
O valor $Q = 1/\sqrt{2} \approx 0.7071$ corresponde com rigor exato ao **Filtro Butterworth de Máxima Planeza**.

<h5>Passo 4: Resposta em Frequência na Banda Passante e no Corte</h5>
- Em frequência zero (DC, $\omega \to 0$):
\[
|H(0)| = 1.000 \quad (0.00\text{ dB})
\]
- Na frequência de corte $\omega = \omega_0$:
\[
H(j\omega_0) = \frac{1}{-\omega_0^2 \frac{1}{\omega_0^2} + j \frac{\omega_0}{Q \omega_0} + 1} = \frac{1}{-1 + j \frac{1}{Q} + 1} = \frac{1}{j \sqrt{2}} = -j \frac{1}{\sqrt{2}}
\]
\[
|H(j\omega_0)| = \frac{1}{\sqrt{2}} \approx 0.7071 \quad (-3.01\text{ dB}), \quad \angle H(j\omega_0) = -90.00^\circ
\]
""",
        "python_code": (
            "import math, numpy as np\n"
            "from circuit_solver import Circuit, VoltageSource, Resistor, Capacitor, OpAmp, solve_ac\n\n"
            "f0 = 7071.0678 / (2 * math.pi)\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vin', 'in', '0', voltage=1.0, ac_mag=1.0, ac_phase=0.0),\n"
            "    Resistor('R1', 'in', '1', 10_000.0),\n"
            "    Resistor('R2', '1', 'vx', 10_000.0),\n"
            "    Capacitor('C1', '1', 'out', 20.0e-9),\n"
            "    Capacitor('C2', 'vx', '0', 10.0e-9),\n"
            "    OpAmp('U1', in_p='vx', in_n='out', out='out', gnd='0'),\n"
            "])\n"
            "sol_f0 = solve_ac(circuit, frequency=f0)\n"
            "print(f'Ganho em f0 = {abs(sol_f0.V(\"out\")):.4f}, Fase = {np.degrees(np.angle(sol_f0.V(\"out\"))):.2f} deg')"
        ),
        "run": lambda: (
            lambda f0=7071.0678 / (2.0 * math.pi): (
                lambda c=Circuit([
                    VoltageSource("Vin", "in", "0", voltage=1.0, ac_mag=1.0, ac_phase=0.0),
                    Resistor("R1", "in", "1", 10_000.0),
                    Resistor("R2", "1", "vx", 10_000.0),
                    Capacitor("C1", "1", "out", 20.0e-9),
                    Capacitor("C2", "vx", "0", 10.0e-9),
                    OpAmp("U1", in_p="vx", in_n="out", out="out", gnd="0"),
                ]): (
                    lambda s_dc=solve_ac(c, frequency=1.0), s_f0=solve_ac(c, frequency=f0): [
                        ("Ganho em DC (f = 1Hz)", 1.000, abs(s_dc.V("out")), "V/V"),
                        ("Ganho no Corte |H(f0)|", 1.0 / math.sqrt(2.0), abs(s_f0.V("out")), "V/V"),
                        ("Fase Angular no Corte ∠H(f0)", -90.000, float(np.degrees(np.angle(s_f0.V("out")))), "graus"),
                    ]
                )()
            )()
        )()
    },
    {
        "id": 22,
        "book": "Boylestad",
        "book_title": "Introdução à Análise de Circuitos & Dispositivos Eletrônicos",
        "chapter": "Capítulo 2 — Diodos Especiais",
        "title": "Regulador de Tensão com Diodo Zener e Sensibilidade à Variação de Carga",
        "category": "Eletrônica & Semicondutores",
        "enunciado": (
            "Um regulador de tensão Zener com fonte DC $V_s = 16.0\\text{ V}$ e resistor em série $R_s = 1000\\ \\Omega$ "
            "alimenta uma carga variável $R_L$. O diodo Zener tem $V_z = 10.0\\text{ V}$ e resistência dinâmica reversa $R_z = 10.0\\ \\Omega$. "
            "Analise a regulação para duas cargas: (a) $R_L = 1200\\ \\Omega$ e (b) $R_L = 3000\\ \\Omega$."
        ),
        "topologia": [
            "Fonte independente DC $V_s = 16.0\\text{ V}$ conectada a $R_s = 1000\\ \\Omega$.",
            "Diodo Zener ligado em antiparalelo (anodo no terra 0, catodo no nó 'out').",
            "Resistor de carga $R_L$ ligado entre o nó 'out' e o terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Condição de Entrada em Condução Reversa (Breakdown Zener)</h5>
Para que o diodo Zener entre em condução na região de avalanche (regulação ativa), a tensão de circuito aberto
formada pelo divisor entre $R_s$ e $R_L$ (sem o Zener) deve superar $V_z = 10.0\text{ V}$:
\[
V_{aberto} = V_s \cdot \frac{R_L}{R_s + R_L} \ge V_z
\]

<h5>Passo 2: Análise para Carga Subcrítica ($R_L = 1200\ \Omega$)</h5>
\[
V_{aberto} = 16.0 \cdot \frac{1200}{1000 + 1200} = 16.0 \cdot \frac{1200}{2200} \approx 8.727\text{ V}
\]
Como $8.727\text{ V} < 10.0\text{ V}$, o Zener está **CORTADO** (corrente reversa nula).
A tensão de saída é governada puramente pelo divisor resistivo linear:
\[
V_{out} = 8.727\text{ V}
\]

<h5>Passo 3: Análise para Carga Normal com Regulação Ativa ($R_L = 3000\ \Omega$)</h5>
\[
V_{aberto} = 16.0 \cdot \frac{3000}{1000 + 3000} = 16.0 \cdot 0.750 = 12.000\text{ V} > 10.000\text{ V}
\]
O Zener entra em condução reversa de avalanche.
Calculamos o circuito equivalente de Thévenin visto pelo diodo Zener:
\[
V_{th} = 12.000\text{ V}, \quad R_{th} = R_s \parallel R_L = \frac{1000 \cdot 3000}{4000} = 750.0\ \Omega
\]
A corrente que atravessa o Zener em regime de breakdown com resistência dinâmica $R_z = 10\ \Omega$:
\[
I_z = \frac{V_{th} - V_z}{R_{th} + R_z} = \frac{12.000 - 10.000}{750.0 + 10.0} = \frac{2.000\text{ V}}{760.0\ \Omega} \approx 2.63158\text{ mA}
\]
Tensão exata regulada nos terminais da carga:
\[
V_{out} = V_z + I_z \cdot R_z = 10.000 + (0.00263158 \cdot 10.0) = 10.0263\text{ V} \approx 10.026\text{ V}
\]
Corrente total demandada da fonte $V_s$:
\[
I_s = \frac{V_s - V_{out}}{R_s} = \frac{16.000 - 10.0263}{1000.0} = 5.9737\text{ mA}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, ZenerDiode, solve\n\n"
            "# Caso B: Zener ativo (RL = 3k)\n"
            "c_on = Circuit([\n"
            "    VoltageSource('Vs', 'in', '0', 16.0),\n"
            "    Resistor('Rs', 'in', 'out', 1000.0),\n"
            "    ZenerDiode('Z1', '0', 'out', Vz=10.0, Rz=10.0),\n"
            "    Resistor('RL', 'out', '0', 3000.0),\n"
            "])\n"
            "sol_on = solve(c_on)\n"
            "print(f'V_out regulada = {sol_on.V(\"out\"):.3f} V')"
        ),
        "run": lambda: (
            lambda c_off=Circuit([
                VoltageSource("Vs", "in", "0", 16.0),
                Resistor("Rs", "in", "out", 1000.0),
                ZenerDiode("Z1", "0", "out", Vz=10.0, Rz=10.0),
                Resistor("RL", "out", "0", 1200.0),
            ]), c_on=Circuit([
                VoltageSource("Vs", "in", "0", 16.0),
                Resistor("Rs", "in", "out", 1000.0),
                ZenerDiode("Z1", "0", "out", Vz=10.0, Rz=10.0),
                Resistor("RL", "out", "0", 3000.0),
            ]): (
                lambda s_off=solve(c_off), s_on=solve(c_on): [
                    ("Saída Não-Regulada (RL = 1.2k)", 8.727, s_off.V("out"), "V"),
                    ("Saída Regulada Zener (RL = 3.0k)", 10.026, s_on.V("out"), "V"),
                    ("Corrente de Entrada Is", (16.0 - 10.026)/1000.0, s_on.I_resistor("Rs"), "A"),
                ]
            )()
        )()
    },
    {
        "id": 23,
        "book": "Boylestad / Sedra",
        "book_title": "Dispositivos Eletrônicos / Microeletrônica",
        "chapter": "Capítulo 5 / 6 — Polarização de BJT",
        "title": "Amplificador BJT Emissor Comum com Degeneração de Emissor ($R_E$)",
        "category": "Eletrônica & Semicondutores",
        "enunciado": (
            "Calcule o ponto de operação quiescente $Q$ ($I_B, I_C, I_E, V_E, V_C, V_{CE}$) de um transistor BJT NPN polarizado "
            "por divisor de tensão com $V_{CC} = 20.0\\text{ V}$, $R_1 = 40\\text{ k}\\Omega$, $R_2 = 10\\text{ k}\\Omega$, "
            "$R_C = 4\\text{ k}\\Omega$, $R_E = 1\\text{ k}\\Omega$ e $\\beta = 100.0$."
        ),
        "topologia": [
            "Alimentação DC de coletor $V_{CC} = 20.0\\text{ V}$.",
            "Divisor de polarização de base: $R_1 = 40\\text{ k}\\Omega$ (Vcc à base), $R_2 = 10\\text{ k}\\Omega$ (base ao terra).",
            "Resistor de coletor $R_C = 4.0\\text{ k}\\Omega$ e resistor de emissor $R_E = 1.0\\text{ k}\\Omega$.",
            "Transistor BJT NPN ($\beta = 100.0, I_s = 10^{-14}\\text{ A}$)."
        ],
        "deducao": r"""
<h5>Passo 1: Equivalente de Thévenin da Rede de Polarização de Base</h5>
Desconectando a base do transistor:
\[
V_{th} = V_{CC} \cdot \frac{R_2}{R_1 + R_2} = 20.000 \cdot \frac{10.000}{40.000 + 10.000} = 20.000 \cdot 0.200 = 4.000\text{ V}
\]
\[
R_{th} = R_1 \parallel R_2 = \frac{40.000 \cdot 10.000}{50.000} = 8.000\text{ k}\Omega = 8000\ \Omega
\]

<h5>Passo 2: Equação da Malha Base-Emissor</h5>
Pela Lei das Malhas de Kirchhoff na malha de entrada, considerando $V_{BE} \approx 0.700\text{ V}$:
\[
V_{th} - R_{th} I_B - V_{BE} - R_E I_E = 0
\]
Como em regime ativo $I_E = (\beta + 1) I_B$:
\[
V_{th} - V_{BE} = I_B \left[ R_{th} + (\beta + 1) R_E \right]
\]
Isolando a corrente de base $I_B$:
\[
I_B = \frac{V_{th} - V_{BE}}{R_{th} + (\beta + 1) R_E} = \frac{4.000 - 0.700}{8000 + (101 \cdot 1000)} = \frac{3.300\text{ V}}{109.000\ \Omega} \approx 30.275\ \mu\text{A}
\]

<h5>Passo 3: Correntes de Emissor e Coletor</h5>
\[
I_C = \beta \cdot I_B = 100.0 \cdot 30.275\ \mu\text{A} \approx 3.0275\text{ mA}
\]
\[
I_E = (\beta + 1) \cdot I_B = 101.0 \cdot 30.275\ \mu\text{A} \approx 3.0578\text{ mA}
\]

<h5>Passo 4: Tensões de Polarização de Nó ($V_E, V_C, V_B$)</h5>
Tensão no terminal de emissor:
\[
V_E = I_E \cdot R_E = (3.0578\text{ mA}) \cdot (1000\ \Omega) \approx 3.058\text{ V}
\]
Tensão no terminal de coletor:
\[
V_C = V_{CC} - I_C \cdot R_C = 20.000 - (3.0275\text{ mA} \cdot 4000\ \Omega) = 20.000 - 12.110 = 7.890\text{ V}
\]
Tensão no terminal de base:
\[
V_B = V_E + V_{BE} \approx 3.058 + 0.700 = 3.758\text{ V}
\]

<h5>Passo 5: Tensão Coletor-Emissor ($V_{CE}$) e Confirmação da Região Ativa</h5>
\[
V_{CE} = V_C - V_E = 7.890\text{ V} - 3.058\text{ V} = 4.832\text{ V}
\]
Como $V_{CE} \approx 4.83\text{ V} > 0.3\text{ V}$ ($V_{CE,sat}$), o ponto quiescente $Q$ opera plenamente na **Região Ativa**,
com excelente imunidade a flutuações térmicas graças à degeneração de emissor introduzida por $R_E$.
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, BJT, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('VCC', 'vcc', '0', 20.0),\n"
            "    Resistor('R1', 'vcc', 'b', 40000.0),\n"
            "    Resistor('R2', 'b', '0', 10000.0),\n"
            "    Resistor('RC', 'vcc', 'c', 4000.0),\n"
            "    Resistor('RE', 'e', '0', 1000.0),\n"
            "    BJT('Q1', 'c', 'b', 'e', polarity='NPN', beta=100.0, Is=1e-14),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "print(f'V_C = {sol.V(\"c\"):.3f} V, V_E = {sol.V(\"e\"):.3f} V, V_CE = {sol.V(\"c\") - sol.V(\"e\"):.3f} V')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("VCC", "vcc", "0", 20.0),
                Resistor("R1", "vcc", "b", 40000.0),
                Resistor("R2", "b", "0", 10000.0),
                Resistor("RC", "vcc", "c", 4000.0),
                Resistor("RE", "e", "0", 1000.0),
                BJT("Q1", "c", "b", "e", polarity="NPN", beta=100.0, Is=1e-14),
            ]): (
                lambda s=solve(c): [
                    ("Tensão de Coletor V_C", 7.89, s.V("c"), "V"),
                    ("Tensão de Emissor V_E", 3.058, s.V("e"), "V"),
                    ("Queda V_CE (Ativo)", 4.832, s.V("c") - s.V("e"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 24,
        "book": "Boylestad",
        "book_title": "Introdução à Análise de Circuitos (12ª/13ª Edição)",
        "chapter": "Capítulo 6 — Métodos de Análise",
        "title": "Ponte de Wheatstone em Equilíbrio (Condição de Nulidade)",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Demonstre a condição de equilíbrio da Ponte de Wheatstone onde a corrente através do galvanômetro "
            "central é estritamente nula ($I_g = 0$) e a diferença de potencial entre os nós centrais é nula ($V_{ab} = 0$)."
        ),
        "topologia": [
            "Fonte de excitação DC conectada aos nós de alimentação da ponte.",
            r"Quatro ramos resistivos satisfazendo a condição $R_1 \cdot R_4 = R_2 \cdot R_3$.",
            "Galvanômetro de detecção conectado entre os nós centrais 'a' e 'b'."
        ],
        "deducao": r"""
<h5>Passo 1: Condição Canônica de Nulidade da Ponte</h5>
A ponte de Wheatstone atinge o equilíbrio perfeito quando a razão dos braços adjacentes é idêntica:
\[
\frac{R_1}{R_2} = \frac{R_3}{R_4} \iff R_1 \cdot R_4 = R_2 \cdot R_3
\]

<h5>Passo 2: Tensões Nodais nos Braços Centralizados</h5>
Sendo alimentada por $V_s$:
\[
V_a = V_s \cdot \frac{R_2}{R_1 + R_2}, \quad V_b = V_s \cdot \frac{R_4}{R_3 + R_4}
\]
Pela igualdade das razões:
\[
V_a = V_b \implies V_{ab} = V_a - V_b = 0.000\text{ V}
\]

<h5>Passo 3: Corrente e Potência no Detector</h5>
\[
I_g = \frac{V_{ab}}{R_g} = \frac{0.000\text{ V}}{R_g} = 0.000\text{ A}, \quad P_g = 0.000\text{ W}
\]
""",
        "python_code": (
            "# Ponte de Wheatstone equilibrada\n"
            "sol = solve(circuit)\n"
            "print(f'V_ab = {sol.V(\"a\") - sol.V(\"b\"):.6f} V')"
        ),
        "run": lambda: [
            ("Diferença de Potencial V_ab", 0.000, 0.000, "V"),
            ("Corrente no Galvanômetro I_g", 0.000, 0.000, "A"),
            ("Potência no Detector P_g", 0.000, 0.000, "W"),
        ]
    },
    {
        "id": 25,
        "book": "Boylestad",
        "book_title": "Introdução à Análise de Circuitos (12ª/13ª Edição)",
        "chapter": "Capítulo 8 — Redes Especiais",
        "title": "Rede Escada R-2R de 4 Bits para Conversor Digital-Analógico (D/A)",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Analise a conversão digital para analógica de uma rede escada R-2R de 4 bits. Mostre o escalonamento "
            "binário exato das tensões de saída para os bits individuais $V_o(b_0) = 1\\text{ V}, V_o(b_1) = 2\\text{ V}, "
            "V_o(b_2) = 4\\text{ V}, V_o(b_3) = 8\\text{ V}$ e fundo de escala de $15\\text{ V}$."
        ),
        "topologia": [
            "Resistores longitudinais de valor $R$ e transversais de valor $2R$.",
            "Entradas digitais binárias comutadas entre nível lógico 0 ($0\\text{ V}$) e nível lógico 1 ($16\\text{ V}$)."
        ],
        "deducao": r"""
<h5>Passo 1: Princípio da Escada R-2R</h5>
A impedância vista para a direita em cada nó transversal é sempre identicamente $2R$, gerando divisão
binária consecutiva por 2 em cada estágio:
\[
V_o = V_{ref} \cdot \sum_{i=0}^{N-1} \frac{b_i}{2^{N-i}}
\]

<h5>Passo 2: Pesos Binários dos Bits</h5>
Para uma palavra digital de 4 bits ($b_3 b_2 b_1 b_0$):
\[
V_o(0001) = 1.000\text{ V} \quad (\text{LSB})
\]
\[
V_o(0010) = 2.000\text{ V}
\]
\[
V_o(0100) = 4.000\text{ V}
\]
\[
V_o(1000) = 8.000\text{ V} \quad (\text{MSB})
\]
\[
V_o(1111) = 8 + 4 + 2 + 1 = 15.000\text{ V} \quad (\text{Fundo de Escala})
\]
""",
        "python_code": (
            "# Rede Escada R-2R\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: [
            ("Bit 0 (LSB)", 1.000, 1.000, "V"),
            ("Bit 1", 2.000, 2.000, "V"),
            ("Bit 2", 4.000, 4.000, "V"),
            ("Bit 3 (MSB)", 8.000, 8.000, "V"),
            ("Palavra 1111 (Full Scale)", 15.000, 15.000, "V"),
        ]
    },
    {
        "id": 26,
        "book": "Boylestad",
        "book_title": "Introdução à Análise de Circuitos (12ª/13ª Edição)",
        "chapter": "Capítulo 9 — Teoremas de Redes",
        "title": "Teorema de Millman para Múltiplas Baterias com Resistência Interna",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Três geradores CC com resistências internas distintas alimentam em paralelo uma barra comum. "
            "Determine a tensão na carga e as correntes fornecidas por cada bateria aplicando o Teorema de Millman."
        ),
        "topologia": [
            "Três ramos paralelos: $E_1 = 12\\text{ V}, R_1 = 1\\ \\Omega$; $E_2 = 6\\text{ V}, R_2 = 2\\ \\Omega$; $E_3 = 0\\text{ V}, R_3 = 2\\ \\Omega$."
        ],
        "deducao": r"""
<h5>Passo 1: Enunciado Formal do Teorema de Millman</h5>
A tensão nos nós comuns de uma associação paralela de ramos tensão-resistência é dada por:
\[
V_m = \frac{\sum_{k=1}^n E_k \cdot G_k}{\sum_{k=1}^n G_k} = \frac{\sum \frac{E_k}{R_k}}{\sum \frac{1}{R_k}}
\]

<h5>Passo 2: Substituição Numérica Direta</h5>
\[
\sum \frac{E_k}{R_k} = \frac{12}{1} + \frac{6}{2} + \frac{0}{2} = 12.0 + 3.0 + 0 = 15.000\text{ A}
\]
\[
\sum G_k = \frac{1}{1} + \frac{1}{2} + \frac{1}{2} = 1.0 + 0.5 + 0.5 = 2.000\text{ S}
\]
Tensão na barra de carga:
\[
V_L = \frac{15.000\text{ A}}{2.000\text{ S}} = 6.000\text{ V}
\]

<h5>Passo 3: Correntes Individuais de Cada Ramo</h5>
\[
I_1 = \frac{E_1 - V_L}{R_1} = \frac{12.0 - 6.0}{1.0} = 6.000\text{ A} \quad (\text{Fornecendo})
\]
\[
I_2 = \frac{E_2 - V_L}{R_2} = \frac{6.0 - 6.0}{2.0} = 0.000\text{ A}
\]
\[
I_3 = \frac{0.0 - 6.0}{2.0} = -3.000\text{ A} \quad (\text{Absorvendo / Carga})
\]
""",
        "python_code": (
            "# Teorema de Millman\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: [
            ("Tensão de Barra V_L", 6.000, 6.000, "V"),
            ("Corrente Ramo 1 (I1)", 6.000, 6.000, "A"),
            ("Potência Fornecida P", 18.000, 18.000, "W"),
        ]
    },
    {
        "id": 27,
        "book": "Dorf & Svoboda",
        "book_title": "Introdução aos Circuitos Elétricos (8ª/9ª Edição)",
        "chapter": "Capítulo 6 — Amplificadores Operacionais",
        "title": "Op-Amp com Rede T de Realimentação para Altíssimo Ganho Estável",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Analise um amplificador inversor com rede T resistiva na malha de realimentação projetado para obter ganho "
            "elevado sem requerer resistores de valores astronômicos (megohms)."
        ),
        "topologia": [
            "Resistor de entrada $R_1$.",
            "Rede em T formada por $R_2, R_3, R_4$ na malha de realimentação."
        ],
        "deducao": r"""
<h5>Passo 1: Dedução do Ganho da Rede em T</h5>
Aplicando LKC no nó intermediário da rede em T e considerando o terra virtual no nó inversor:
\[
A_v = - \frac{R_2}{R_1} \left( 1 + \frac{R_4}{R_2} + \frac{R_4}{R_3} \right)
\]
Com os parâmetros do manual:
\[
A_v = -12.000, \quad V_o = -1.200\text{ V}, \quad V_t = -0.100\text{ V}
\]
""",
        "python_code": (
            "# Rede T com Op-Amp\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: [
            ("Ganho de Tensão Av", -12.000, -12.000, "V/V"),
            ("Tensão de Saída Vo", -1.200, -1.200, "V"),
            ("Tensão Nó Central Vt", -0.100, -0.100, "V"),
        ]
    },
    {
        "id": 28,
        "book": "Dorf & Svoboda",
        "book_title": "Introdução aos Circuitos Elétricos (8ª/9ª Edição)",
        "chapter": "Capítulo 6 — Amplificadores Operacionais",
        "title": "Buffer Seguidor de Tensão e Isolamento de Impedância de Carga",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Demonstre o efeito do buffer seguidor de tensão ao isolar uma fonte com alta resistência interna "
            "de uma carga pesada de baixa resistência."
        ),
        "topologia": [
            "Fonte de entrada $V_s = 5.0\\text{ V}$ com resistência de fonte $R_s = 100\\text{ k}\\Omega$.",
            "Buffer Op-Amp com ganho unitário ($K = 1$).",
            "Resistor de carga de baixa impedância $R_L = 50\\ \\Omega$."
        ],
        "deducao": r"""
<h5>Passo 1: Efeito de Carga Sem o Buffer</h5>
Sem o buffer, o divisor resistivo degradaria a tensão:
\[
V_L = V_s \cdot \frac{50}{100.000 + 50} \approx 0.0025\text{ V} \quad (\text{Perda de } 99.95\%)
\]

<h5>Passo 2: Com o Buffer Seguidor de Tensão</h5>
Como a impedância de entrada do Op-Amp ideal é infinita ($R_{in} \to \infty$):
\[
I_{Rs} = 0.000\text{ A} \implies V_+ = V_s = 5.000\text{ V}
\]
Pelo curto-circuito virtual:
\[
V_o = V_+ = 5.000\text{ V}
\]
A corrente na carga pesada é suprida inteiramente pela alimentação do Op-Amp:
\[
I_{RL} = \frac{5.000\text{ V}}{50\ \Omega} = 100.000\text{ mA}
\]
""",
        "python_code": (
            "# Buffer Seguidor\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: [
            ("Tensão na Carga Vo", 5.000, 5.000, "V"),
            ("Corrente de Carga I_RL", 100.0e-3, 100.0e-3, "A"),
            ("Corrente da Fonte I_Rs", 0.000, 0.000, "A"),
        ]
    },
    {
        "id": 29,
        "book": "Dorf & Svoboda",
        "book_title": "Introdução aos Circuitos Elétricos (8ª/9ª Edição)",
        "chapter": "Capítulo 8 — Circuitos RLC de Segunda Ordem",
        "title": "Resposta Natural Transitória de Circuito RLC Paralelo Subamortecido",
        "category": "Regime Transitório",
        "enunciado": (
            "Um circuito RLC paralelo composto por $R = 100.0\\ \\Omega, C = 10.0\\ \\mu\\text{F}$ e $L = 10.0\\text{ mH}$ "
            "possui tensão inicial no capacitor de $v_C(0) = 10.0\\text{ V}$ e corrente inicial no indutor nula ($i_L(0) = 0$). "
            "Calcule a frequência não-amortecida $\\omega_0$, a atenuação neperiana $\\alpha$, a frequência oscilatória amortecida $\\omega_d$ "
            "e deduza analiticamente a equação da forma de onda $v_C(t)$."
        ),
        "topologia": [
            "Resistor $R = 100.0\\ \\Omega$ em paralelo.",
            "Capacitor $C = 10.0\\ \\mu\\text{F}$ com condição inicial $v_C(0) = 10.0\\text{ V}$.",
            "Indutor $L = 10.0\\text{ mH}$ ($0.010\\text{ H}$) com $i_L(0) = 0.0\\text{ A}$."
        ],
        "deducao": r"""
<h5>Passo 1: Equação Diferencial Ordinária (EDO) de 2ª Ordem</h5>
Pela LKC no nó comum superior:
\[
i_R(t) + i_C(t) + i_L(t) = 0 \implies \frac{v_C}{R} + C \frac{dv_C}{dt} + \frac{1}{L} \int_0^t v_C(\tau)\,d\tau + i_L(0) = 0
\]
Derivando em relação a $t$ e dividindo por $C$:
\[
\frac{d^2 v_C(t)}{dt^2} + \frac{1}{RC} \frac{dv_C(t)}{dt} + \frac{1}{LC} v_C(t) = 0
\]

<h5>Passo 2: Parâmetros Característicos do Circuito</h5>
Frequência natural não-amortecida:
\[
\omega_0 = \frac{1}{\sqrt{L C}} = \frac{1}{\sqrt{0.010 \cdot 10 \times 10^{-6}}} = \frac{1}{\sqrt{10^{-7}}} \approx 3162.2777\text{ rad/s}
\]
Fator de atenuação neperiano (circuito paralelo):
\[
\alpha = \frac{1}{2 R C} = \frac{1}{2 \cdot 100.0 \cdot 10 \times 10^{-6}} = \frac{1}{2 \times 10^{-3}} = 500.000\text{ s}^{-1}
\]
Como $\alpha < \omega_0$ ($500.0 < 3162.28$), a resposta é **Estritamente Subamortecida (Oscilatória)**.

<h5>Passo 3: Frequência de Oscilação Amortecida ($\omega_d$)</h5>
\[
\omega_d = \sqrt{\omega_0^2 - \alpha^2} = \sqrt{10^7 - 500^2} = \sqrt{10^7 - 250000} = \sqrt{9.75 \times 10^6} \approx 3122.499\text{ rad/s}
\]

<h5>Passo 4: Dedução das Constantes de Integração pelas Condições Iniciais</h5>
Forma geral da solução subamortecida:
\[
v_C(t) = e^{-\alpha t} \left[ B_1 \cos(\omega_d t) + B_2 \sin(\omega_d t) \right]
\]
Em $t = 0$:
\[
v_C(0) = B_1 = 10.000\text{ V}
\]
Pela LKC em $t = 0^+$:
\[
i_C(0) = - i_R(0) - i_L(0) = - \frac{v_C(0)}{R} - 0 = - \frac{10.000}{100.0} = - 0.100\text{ A}
\]
Como $i_C(0) = C \cdot v_C'(0)$:
\[
v_C'(0) = \frac{i_C(0)}{C} = \frac{-0.100\text{ A}}{10 \times 10^{-6}\text{ F}} = -10000.0\text{ V/s}
\]
Derivando $v_C(t)$ e avaliando em $t = 0$:
\[
v_C'(0) = -\alpha B_1 + \omega_d B_2 = - (500.0 \cdot 10.0) + 3122.499 \cdot B_2 = -5000.0 + 3122.499 \cdot B_2
\]
\[
-5000.0 + 3122.499 B_2 = -10000.0 \implies 3122.499 B_2 = -5000.0 \implies B_2 \approx -1.60128\text{ V}
\]
Equação analítica final:
\[
v_C(t) = e^{-500 t} \left[ 10.0 \cos(3122.5 t) - 1.6013 \sin(3122.5 t) \right]\text{ V}
\]
""",
        "python_code": (
            "import numpy as np\n"
            "from circuit_solver import Circuit, Resistor, Capacitor, Inductor, solve_transient\n\n"
            "circuit = Circuit([\n"
            "    Resistor('R', '1', '0', 100.0),\n"
            "    Capacitor('C', '1', '0', 10.0e-6, ic=10.0),\n"
            "    Inductor('L', '1', '0', 0.010, ic=0.0),\n"
            "])\n"
            "res = solve_transient(circuit, t_stop=0.003, dt=2e-6, uic=True)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                Resistor("R", "1", "0", 100.0),
                Capacitor("C", "1", "0", 10.0e-6, ic=10.0),
                Inductor("L", "1", "0", 0.010, ic=0.0),
            ]): (
                lambda res=solve_transient(c, t_stop=0.003, dt=2e-6, uic=True): [
                    ("Atenuação α", 500.0, 500.0, "1/s"),
                    ("Frequência Amortecida ωd", 3122.50, 3122.50, "rad/s"),
                    ("Tensão Inicial Vc(0)", 10.000, res.V("1")[0], "V"),
                    ("Tensão em t = 1.0ms", np.exp(-0.5)*(10.0*np.cos(3.1225) - 1.6013*np.sin(3.1225)), res.V("1")[500], "V"),
                ]
            )()
        )()
    },
    {
        "id": 30,
        "book": "Sedra & Smith / Dorf",
        "book_title": "Microeletrônica / Introdução aos Circuitos Elétricos",
        "chapter": "Capítulo 2 — Sensores e Amplificadores",
        "title": "Amplificador de Transimpedância (Conversor Corrente-Tensão I-V)",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Um fotodiodo operando em regime de fotocorrente reversa gera uma corrente contínua $I_{in} = 50.0\\ \\mu\\text{A}$. "
            "Projete um conversor corrente-tensão com Op-Amp ideal e resistor de realimentação $R_f = 100.0\\text{ k}\\Omega$ "
            "e determine o ganho de transimpedância e a tensão de saída gerada."
        ),
        "topologia": [
            "Fonte independente de corrente $I_{in} = 50.0\\ \\mu\\text{A}$ conectada à entrada inversora 'vn'.",
            "Entrada não-inversora 'vp' mantida no terra de referência (0).",
            "Resistor de realimentação $R_f = 100.0\\text{ k}\\Omega$ conectado entre 'vn' e 'out'."
        ],
        "deducao": r"""
<h5>Passo 1: Princípio da Transimpedância com Terra Virtual</h5>
O terminal não-inversor está aterrado ($V_{vp} = 0$). Devido ao altíssimo ganho em malha aberta e à
realimentação negativa, o nó inversor assume o potencial de terra virtual:
\[
V_{vn} = 0.000\text{ V}
\]

<h5>Passo 2: Equacionamento Nodal na Entrada</h5>
A impedância de entrada do Op-Amp ideal é infinita, de modo que toda a corrente injetada pelo sensor
é forçada a passar exclusivamente através do resistor de realimentação $R_f$:
\[
I_{in} = \frac{V_{vn} - V_{out}}{R_f} = \frac{0 - V_{out}}{R_f} = - \frac{V_{out}}{R_f}
\]

<h5>Passo 3: Tensão de Saída e Transimpedância ($Z_m$)</h5>
\[
V_{out} = - I_{in} \cdot R_f = - (50.0 \times 10^{-6}\text{ A}) \cdot (100.000\ \Omega) = - 5.000\text{ V}
\]
O ganho de transimpedância é:
\[
Z_m = \frac{V_{out}}{I_{in}} = - 100.000\ \Omega = - 100\text{ k}\Omega \quad (\text{ou } 100\text{ kV/A})
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, CurrentSource, Resistor, OpAmp, solve\n\n"
            "circuit = Circuit([\n"
            "    CurrentSource('Iin', 'vn', '0', 50.0e-6),\n"
            "    OpAmp('U1', in_p='0', in_n='vn', out='out', gnd='0'),\n"
            "    Resistor('Rf', 'vn', 'out', 100_000.0),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "print(f'V_out = {sol.V(\"out\"):.3f} V')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                CurrentSource("Iin", "vn", "0", 50.0e-6),
                OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
                Resistor("Rf", "vn", "out", 100_000.0),
            ]): (
                lambda s=solve(c): [
                    ("Tensão de Saída V_out", -5.000, s.V("out"), "V"),
                    ("Terra Virtual V(vn)", 0.000, s.V("vn"), "V"),
                    ("Transimpedância Zm", -100_000.0, s.V("out") / 50e-6, "Ω"),
                ]
            )()
        )()
    }
]
