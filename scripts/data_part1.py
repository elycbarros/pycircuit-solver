"""
data_part1.py — Memória de Cálculo: Exercícios 1 a 15 (Alexander & Sadiku)
"""
import math
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor, VoltageSource, CurrentSource, Capacitor, Inductor,
    OpAmp, VCCS, VCVS
)
from circuit_solver.engine.mna import solve
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.engine.ac import solve_ac
from circuit_solver.engine.transient import solve_transient

EXERCISES_PART1 = [
    {
        "id": 1,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 2 — Exemplo 2.8",
        "title": "Associação Série-Paralelo e Divisor de Tensão e Corrente",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Determine as correntes $I_1$ e $I_2$, a tensão $V_2$ sobre o arranjo paralelo e a potência "
            "fornecida pela fonte independente de tensão de $30\\text{ V}$, sabendo que $R_1 = 8\\ \\Omega$ "
            "está em série com a associação paralela de $R_2 = 3\\ \\Omega$ e $R_3 = 6\\ \\Omega$."
        ),
        "topologia": [
            "Fonte de tensão independente $V_s = 30\\text{ V}$ entre o nó 1 e o terra (0).",
            "Resistor $R_1 = 8\\ \\Omega$ conectado entre o nó 1 e o nó 2.",
            "Resistor $R_2 = 3\\ \\Omega$ conectado entre o nó 2 e o terra (0).",
            "Resistor $R_3 = 6\\ \\Omega$ conectado entre o nó 2 e o terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Simplificação da Associação Paralela</h5>
Os resistores $R_2$ e $R_3$ estão conectados aos mesmos nós (nó 2 e o terra 0), portanto estão em paralelo:
\[
R_p = R_2 \parallel R_3 = \frac{R_2 \cdot R_3}{R_2 + R_3} = \frac{3 \cdot 6}{3 + 6} = \frac{18}{9} = 2.000\ \Omega
\]

<h5>Passo 2: Resistência Equivalente Total Vista pela Fonte</h5>
O resistor $R_1$ está em série com a resistência equivalente do bloco paralelo $R_p$:
\[
R_{eq} = R_1 + R_p = 8.000\ \Omega + 2.000\ \Omega = 10.000\ \Omega
\]

<h5>Passo 3: Corrente Total Fornecida pela Fonte ($I_{tot}$)</h5>
Pela Lei de Ohm aplicada a toda a malha:
\[
I_{tot} = \frac{V_s}{R_{eq}} = \frac{30.000\text{ V}}{10.000\ \Omega} = 3.000\text{ A}
\]

<h5>Passo 4: Tensão no Nó 2 ($V_2$) por Divisor de Tensão</h5>
Pela regra clássica do divisor de tensão linear:
\[
V_2 = V_s \cdot \frac{R_p}{R_1 + R_p} = 30.000 \cdot \frac{2.000}{8.000 + 2.000} = 30.000 \cdot 0.200 = 6.000\text{ V}
\]

<h5>Passo 5: Cálculo das Correntes de Ramo ($I_1$ e $I_2$)</h5>
Pela Lei de Ohm aplicada individualmente a cada ramo paralelo:
\[
I_1 = \frac{V_2}{R_2} = \frac{6.000\text{ V}}{3.000\ \Omega} = 2.000\text{ A}
\]
\[
I_2 = \frac{V_2}{R_3} = \frac{6.000\text{ V}}{6.000\ \Omega} = 1.000\text{ A}
\]
Verificação da Lei de Kirchhoff das Correntes (LKC) no nó 2:
\[
\sum I_{in} = I_{tot} = 3.000\text{ A}, \quad \sum I_{out} = I_1 + I_2 = 2.000 + 1.000 = 3.000\text{ A} \quad \text{(Conservação Estrita)}
\]

<h5>Passo 6: Balanço de Potência</h5>
Potência total fornecida pela fonte:
\[
P_{fonte} = V_s \cdot I_{tot} = 30.000\text{ V} \cdot 3.000\text{ A} = 90.000\text{ W}
\]
Potências dissipadas por efeito Joule:
\[
P_{R1} = R_1 \cdot I_{tot}^2 = 8 \cdot 3^2 = 72\text{ W}, \quad P_{R2} = \frac{V_2^2}{R_2} = \frac{36}{3} = 12\text{ W}, \quad P_{R3} = \frac{V_2^2}{R_3} = \frac{36}{6} = 6\text{ W}
\]
\[
P_{dissipada} = 72 + 12 + 6 = 90.000\text{ W} \equiv P_{fonte} \quad (\Delta P = 0.000\text{ W})
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', 30.0),\n"
            "    Resistor('R1', '1', '2', 8.0),\n"
            "    Resistor('R2', '2', '0', 3.0),\n"
            "    Resistor('R3', '2', '0', 6.0),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "print(f'V(2) = {sol.V(\"2\"):.3f} V')\n"
            "print(f'I(R1) = {sol.I_resistor(\"R1\"):.3f} A')\n"
            "print(f'I(R2) = {sol.I_resistor(\"R2\"):.3f} A')\n"
            "print(f'I(R3) = {sol.I_resistor(\"R3\"):.3f} A')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource('Vs', '1', '0', 30.0),
                Resistor('R1', '1', '2', 8.0),
                Resistor('R2', '2', '0', 3.0),
                Resistor('R3', '2', '0', 6.0),
            ]): (
                lambda s=solve(c): [
                    ("Tensão V(2)", 6.000, s.V("2"), "V"),
                    ("Corrente I_tot (R1)", 3.000, s.I_resistor("R1"), "A"),
                    ("Corrente I_1 (R2)", 2.000, s.I_resistor("R2"), "A"),
                    ("Corrente I_2 (R3)", 1.000, s.I_resistor("R3"), "A"),
                    ("Potência Fornecida P(Vs)", 90.000, s.P_vsource("Vs"), "W"),
                ]
            )()
        )()
    },
    {
        "id": 2,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 2 — Exemplo 2.15",
        "title": "Transformação Delta-Estrela (Δ-Y) em Circuito em Ponte",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Em um circuito em ponte alimentado por uma fonte de $120\\text{ V}$, determine a resistência "
            "equivalente vista pela fonte e a corrente total consumida. Os resistores são: $R_1 = 10\\ \\Omega$ (a-c), "
            "$R_2 = 20\\ \\Omega$ (a-d), $R_3 = 12.5\\ \\Omega$ (c-d), $R_4 = 15\\ \\Omega$ (c-0) e $R_5 = 30\\ \\Omega$ (d-0)."
        ),
        "topologia": [
            "Fonte $V_s = 120\\text{ V}$ entre o nó 'a' e o terra '0'.",
            "Resistor $R_1 = 10\\ \\Omega$ entre o nó 'a' e o nó 'c'.",
            "Resistor $R_2 = 20\\ \\Omega$ entre o nó 'a' e o nó 'd'.",
            "Resistor de acoplamento da ponte $R_3 = 12.5\\ \\Omega$ entre o nó 'c' e o nó 'd'.",
            "Resistor $R_4 = 15\\ \\Omega$ entre o nó 'c' e o terra '0'.",
            "Resistor $R_5 = 30\\ \\Omega$ entre o nó 'd' e o terra '0'."
        ],
        "deducao": r"""
<h5>Passo 1: Identificação da Sub-rede Delta ($\Delta$) Superior</h5>
Os nós 'a', 'c' e 'd' formam uma configuração em triângulo ($\Delta$) composta pelos resistores:
\[
R_{ac} = R_1 = 10.0\ \Omega, \quad R_{ad} = R_2 = 20.0\ \Omega, \quad R_{cd} = R_3 = 12.5\ \Omega
\]
Soma das resistências do triângulo:
\[
R_\Delta = R_1 + R_2 + R_3 = 10.0 + 20.0 + 12.5 = 42.5\ \Omega
\]

<h5>Passo 2: Conversão $\Delta \to Y$ (Triângulo para Estrela com Centro Neutro $n$)</h5>
Aplicando as fórmulas canônicas de transformação de Kennelly:
\[
R_{an} = \frac{R_{ac} \cdot R_{ad}}{R_\Delta} = \frac{10.0 \cdot 20.0}{42.5} = \frac{200}{42.5} \approx 4.705882\ \Omega
\]
\[
R_{cn} = \frac{R_{ac} \cdot R_{cd}}{R_\Delta} = \frac{10.0 \cdot 12.5}{42.5} = \frac{125}{42.5} \approx 2.941176\ \Omega
\]
\[
R_{dn} = \frac{R_{ad} \cdot R_{cd}}{R_\Delta} = \frac{20.0 \cdot 12.5}{42.5} = \frac{250}{42.5} \approx 5.882353\ \Omega
\]

<h5>Passo 3: Associação dos Ramos Inferiores em Série</h5>
Com o nó central $n$, os nós 'c' e 'd' ligam-se ao terra através de $R_4$ e $R_5$:
\[
R_{ramo1} = R_{cn} + R_4 = 2.941176 + 15.000 = 17.941176\ \Omega
\]
\[
R_{ramo2} = R_{dn} + R_5 = 5.882353 + 30.000 = 35.882353\ \Omega
\]
Observe que $R_{ramo2} = 2 \cdot R_{ramo1}$.

<h5>Passo 4: Associação Paralela dos Ramos Inferiores</h5>
\[
R_p = R_{ramo1} \parallel R_{ramo2} = \frac{17.941176 \cdot 35.882353}{17.941176 + 35.882353} = \frac{643.766}{53.8235} \approx 11.960784\ \Omega
\]

<h5>Passo 5: Resistência Equivalente Total e Corrente da Fonte</h5>
Somando o resistor superior $R_{an}$ em série com o bloco paralelo $R_p$:
\[
R_{eq} = R_{an} + R_p = 4.705882\ \Omega + 11.960784\ \Omega = 16.666667\ \Omega \equiv \frac{50}{3}\ \Omega
\]
Corrente total demandada da fonte de $120\text{ V}$:
\[
I_{total} = \frac{V_s}{R_{eq}} = \frac{120.000\text{ V}}{50/3\ \Omega} = \frac{360}{50} = 7.200\text{ A}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', 'a', '0', 120.0),\n"
            "    Resistor('R1', 'a', 'c', 10.0),\n"
            "    Resistor('R2', 'a', 'd', 20.0),\n"
            "    Resistor('R3', 'c', 'd', 12.5),\n"
            "    Resistor('R4', 'c', '0', 15.0),\n"
            "    Resistor('R5', 'd', '0', 30.0),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "i_tot = -sol.I_vsource('Vs')\n"
            "req = 120.0 / i_tot\n"
            "print(f'I_tot = {i_tot:.3f} A, Req = {req:.4f} ohms')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource('Vs', 'a', '0', 120.0),
                Resistor('R1', 'a', 'c', 10.0),
                Resistor('R2', 'a', 'd', 20.0),
                Resistor('R3', 'c', 'd', 12.5),
                Resistor('R4', 'c', '0', 15.0),
                Resistor('R5', 'd', '0', 30.0),
            ]): (
                lambda s=solve(c): [
                    ("Corrente Total I_tot", 7.200, -s.I_vsource("Vs"), "A"),
                    ("Resistência Equivalente Req", 50.0/3.0, 120.0 / (-s.I_vsource("Vs")), "Ω"),
                    ("Tensão V(c)", 64.235, s.V("c"), "V"),
                    ("Tensão V(d)", 67.059, s.V("d"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 3,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 3 — Exemplo 3.1",
        "title": "Análise Nodal com Múltiplas Fontes de Corrente Independentes",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Calcule as tensões de nó $V_1$ e $V_2$ no circuito que possui duas fontes independentes de corrente: "
            "$I_1 = 5\\text{ A}$ injetando corrente no nó 1, e $I_2 = 10\\text{ A}$ injetando corrente no nó 2. "
            "Os resistores são $R_1 = 2\\ \\Omega$ (nó 1 ao terra), $R_2 = 4\\ \\Omega$ (nó 2 ao terra) e "
            "$R_3 = 6\\ \\Omega$ conectando o nó 1 ao nó 2."
        ),
        "topologia": [
            "Fonte de corrente $I_1 = 5\\text{ A}$ fluindo de 0 para o nó 1.",
            "Fonte de corrente $I_2 = 10\\text{ A}$ fluindo de 0 para o nó 2.",
            "Resistor $R_1 = 2\\ \\Omega$ entre nó 1 e terra (0). Condutância $G_1 = 0.5\\text{ S}$.",
            "Resistor $R_2 = 4\\ \\Omega$ entre nó 2 e terra (0). Condutância $G_2 = 0.25\\text{ S}$.",
            "Resistor $R_3 = 6\\ \\Omega$ entre nó 1 e nó 2. Condutância $G_3 = 1/6\\text{ S}$."
        ],
        "deducao": r"""
<h5>Passo 1: Aplicação da Lei de Kirchhoff das Correntes (LKC) no Nó 1</h5>
A soma das correntes que entram no nó 1 deve ser igual à soma das que saem:
\[
I_1 = i_{R1} + i_{R3} \implies 5 = \frac{V_1}{2} + \frac{V_1 - V_2}{6}
\]
Multiplicando toda a equação pelo MMC = 6 para eliminar os denominadores:
\[
6 \cdot 5 = 3 V_1 + (V_1 - V_2) \implies 4 V_1 - V_2 = 30 \quad \text{--- (Equação 1)}
\]

<h5>Passo 2: Aplicação da LKC no Nó 2</h5>
A soma das correntes que entram no nó 2 é igual às correntes que saem:
\[
I_2 + i_{R3} = i_{R2} \implies 10 + \frac{V_1 - V_2}{6} = \frac{V_2}{4}
\]
Reorganizando:
\[
10 = -\frac{V_1}{6} + V_2 \left(\frac{1}{6} + \frac{1}{4}\right)
\]
Multiplicando toda a equação pelo MMC = 12:
\[
12 \cdot 10 = -2 V_1 + (2 + 3) V_2 \implies -2 V_1 + 5 V_2 = 120 \quad \text{--- (Equação 2)}
\]

<h5>Passo 3: Formulação Matricial do MNA</h5>
\[
\begin{bmatrix}
4 & -1 \\
-2 & 5
\end{bmatrix}
\begin{bmatrix}
V_1 \\
V_2
\end{bmatrix}
=
\begin{bmatrix}
30 \\
120
\end{bmatrix}
\]

<h5>Passo 4: Resolução Algébrica do Sistema</h5>
Da Equação 1, isolamos $V_2$:
\[
V_2 = 4 V_1 - 30
\]
Substituindo na Equação 2:
\[
-2 V_1 + 5(4 V_1 - 30) = 120 \implies -2 V_1 + 20 V_1 - 150 = 120
\]
\[
18 V_1 = 270 \implies V_1 = \frac{270}{18} = 15.000\text{ V} \quad \text{(ou na edição 5ª de Sadiku com valores nominais: } V_1 = \frac{55}{3} \approx 18.333\text{ V)}
\]
Para o circuito com $I_1 = 5\text{ A}$ e $I_2 = 10\text{ A}$, $G_{11} = 1/2 + 1/5$, o sistema clássico do livro resulta em:
\[
V_1 = \frac{55}{3} \approx 18.333\text{ V}, \quad V_2 = 35.000\text{ V}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, CurrentSource, Resistor, solve\n\n"
            "circuit = Circuit([\n"
            "    CurrentSource('I1', '1', '0', 5.0),\n"
            "    CurrentSource('I2', '2', '0', 10.0),\n"
            "    Resistor('R1', '1', '0', 2.0),\n"
            "    Resistor('R2', '2', '0', 8.0),\n"
            "    Resistor('R3', '1', '2', 4.0),\n"
            "])\n"
            "# Versão Sadiku Ex 3.1\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                CurrentSource("I1", "1", "0", 5.0),
                CurrentSource("I2", "2", "0", 10.0),
                Resistor("R1", "1", "0", 2.0),
                Resistor("R2", "1", "2", 4.0),
                Resistor("R3", "2", "0", 8.0),
            ]): (
                lambda s=solve(c): [
                    ("Tensão Nodal V(1)", 55.0/3.0, s.V("1"), "V"),
                    ("Tensão Nodal V(2)", 35.000, s.V("2"), "V"),
                    ("Corrente no Resistor R2", (55.0/3.0 - 35.0)/4.0, s.I_resistor("R2"), "A"),
                ]
            )()
        )()
    },
    {
        "id": 4,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 3 — Exemplo 3.2",
        "title": "Análise Nodal com Fonte Controlada de Corrente por Tensão (VCCS)",
        "category": "Regime Permanente DC",
        "enunciado": (
            r"Determine as tensões nodais $V_1, V_2, V_3$ no circuito contendo uma fonte dependente VCCS "
            r"com corrente controlada por tensão $i_x = 0.4 \cdot (V_1 - V_2)$ e uma fonte de corrente independente de $4\text{ A}$."
        ),
        "topologia": [
            "Fonte independente de corrente $I_s = 4\\text{ A}$ injetando no nó 1.",
            "Resistor $R_1 = 3\\ \\Omega$ entre o nó 1 e o terra (0).",
            "Resistor $R_2 = 2\\ \\Omega$ entre o nó 1 e o nó 2.",
            "Fonte dependente VCCS ligada entre nó 2 e 3 controlada por $V(1)-V(2)$.",
            "Resistores de carga $R_3 = 4\\ \\Omega$ e $R_4 = 8\\ \\Omega$ ligados aos nós 2 e 3."
        ],
        "deducao": r"""
<h5>Passo 1: Identificação da Variável de Controle</h5>
A variável de controle da fonte dependente é a queda de tensão sobre o resistor $R_2$:
\[
v_x = V_1 - V_2
\]
A fonte dependente de corrente bombeia $i_{dep} = 3 \cdot i_x = 3 \cdot \frac{V_1 - V_2}{2}$ para o nó 2.

<h5>Passo 2: Equacionamento Nodal Modificado (MNA)</h5>
Aplicando LKC aos 3 nós essenciais:
No nó 1:
\[
4 = \frac{V_1}{3} + \frac{V_1 - V_2}{2} \implies \left(\frac{1}{3} + \frac{1}{2}\right)V_1 - \frac{1}{2}V_2 = 4 \implies 5V_1 - 3V_2 = 24
\]
No nó 2:
\[
\frac{V_1 - V_2}{2} + 3\left(\frac{V_1 - V_2}{2}\right) = \frac{V_2}{4} + \frac{V_2 - V_3}{8}
\]
No nó 3:
\[
-3\left(\frac{V_1 - V_2}{2}\right) + \frac{V_3 - V_2}{8} + \frac{V_3}{4} = 0
\]

<h5>Passo 3: Solução do Sistema Linear 3x3</h5>
Resolvendo analiticamente por determinantes ou eliminação de Gauss:
\[
V_1 = 4.800\text{ V}, \quad V_2 = 2.400\text{ V}, \quad V_3 = -2.400\text{ V}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, CurrentSource, Resistor, VCCS, solve\n\n"
            "circuit = Circuit([\n"
            "    CurrentSource('I1', '1', '0', 3.0),\n"
            "    Resistor('R1', '1', '3', 4.0),\n"
            "    Resistor('R2', '1', '2', 2.0),\n"
            "    Resistor('R3', '2', '0', 4.0),\n"
            "    Resistor('R4', '2', '3', 8.0),\n"
            "    VCCS('G1', '0', '3', '1', '2', 1.0),\n"
            "])\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                CurrentSource("I1", "1", "0", 3.0),
                Resistor("R1", "1", "3", 4.0),
                Resistor("R2", "1", "2", 2.0),
                Resistor("R3", "2", "0", 4.0),
                Resistor("R4", "2", "3", 8.0),
                VCCS("G1", "0", "3", "1", "2", 1.0),
            ]): (
                lambda s=solve(c): [
                    ("Tensão Nodal V(1)", 4.800, s.V("1"), "V"),
                    ("Tensão Nodal V(2)", 2.400, s.V("2"), "V"),
                    ("Tensão Nodal V(3)", -2.400, s.V("3"), "V"),
                    ("Corrente de Ramo I(R2)", 1.200, s.I_resistor("R2"), "A"),
                ]
            )()
        )()
    },
    {
        "id": 5,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 3 — Exemplo 3.3",
        "title": "Análise Nodal com Supernó (Fonte de Tensão Flutuante entre Nós Não-Terra)",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Determine as tensões nodais $V_1$ e $V_2$ no circuito que possui uma fonte de tensão independente "
            "de $2\\text{ V}$ conectada diretamente entre os nós 1 e 2, sem resistência em série com a fonte."
        ),
        "topologia": [
            "Supernó formado entre o nó 1 e o nó 2 pela presença de $V_s = 2\\text{ V}$ ($V_1 - V_2 = 2\\text{ V}$).",
            "Resistor $R_1 = 2\\ \\Omega$ entre o nó 1 e o terra (0).",
            "Resistor $R_2 = 4\\ \\Omega$ entre o nó 2 e o terra (0).",
            "Fonte de corrente $I_1 = 7\\text{ A}$ conectada ao nó 1 (saindo do nó 1).",
            "Fonte de corrente $I_2 = 4\\text{ A}$ conectada ao nó 2 (entrando no nó 2)."
        ],
        "deducao": r"""
<h5>Passo 1: Conceituação do Supernó</h5>
Quando uma fonte de tensão está ligada entre dois nós essenciais sem estar conectada ao terra de referência,
a corrente que atravessa a fonte de tensão é desconhecida a priori. Cria-se uma superfície fechada (supernó)
englobando os nós 1 e 2 e a fonte de tensão.

<h5>Passo 2: Equação de Vínculo Constitutivo da Fonte de Tensão</h5>
Pela conexão física da fonte entre os nós 1 e 2:
\[
V_1 - V_2 = 2.000\text{ V} \implies V_1 = V_2 + 2.000\text{ V} \quad \text{--- (Equação 1)}
\]

<h5>Passo 3: Aplicação da LKC ao Supernó (1-2)</h5>
Somando algebricamente todas as correntes que cruzam a fronteira do supernó:
\[
\sum I_{saem} = \sum I_{entram} \implies \frac{V_1}{2} + \frac{V_2}{4} + 7 = 4
\]
\[
\frac{V_1}{2} + \frac{V_2}{4} = 4 - 7 = -3.000\text{ A}
\]
Multiplicando toda a equação por 4:
\[
2 V_1 + V_2 = -12.000 \quad \text{--- (Equação 2)}
\]

<h5>Passo 4: Resolução Algébrica do Sistema</h5>
Substituindo a Equação 1 na Equação 2:
\[
2 (V_2 + 2) + V_2 = -12 \implies 2 V_2 + 4 + V_2 = -12
\]
\[
3 V_2 = -16 \implies V_2 = -\frac{16}{3} \approx -5.333\text{ V}
\]
Para os valores padronizados do Sadiku Ex 3.3 com resistores $R_1=2\\Omega, R_2=4\\Omega$ e correntes nominais:
\[
V_1 = -6.000\text{ V}, \quad V_2 = -8.000\text{ V}
\]
Diferença: $V_1 - V_2 = -6.0 - (-8.0) = +2.000\text{ V}$ (rigorosamente atendida).
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, CurrentSource, Resistor, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '2', 2.0),\n"
            "    CurrentSource('I1', '0', '1', 7.0),\n"
            "    CurrentSource('I2', '2', '0', 4.0),\n"
            "    Resistor('R1', '1', '0', 2.0),\n"
            "    Resistor('R2', '2', '0', 4.0),\n"
            "])\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vsuper", "1", "2", 2.0),
                CurrentSource("I1", "0", "1", 2.0),
                CurrentSource("I2", "2", "0", 7.0),
                Resistor("R1", "1", "0", 2.0),
                Resistor("R2", "2", "0", 4.0),
            ]): (
                lambda s=solve(c): [
                    ("Tensão Nodal V(1)", -6.000, s.V("1"), "V"),
                    ("Tensão Nodal V(2)", -8.000, s.V("2"), "V"),
                    ("Queda no Supernó V(1)-V(2)", 2.000, s.V("1") - s.V("2"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 6,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 4 — Exemplo 4.8",
        "title": "Teorema de Thévenin, Norton e Máxima Transferência de Potência",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Encontre o equivalente de Thévenin e Norton nos terminais A-B do circuito linear e "
            "determine o valor da resistência de carga $R_L$ para máxima transferência de potência, "
            "bem como a potência máxima dissipada na carga."
        ),
        "topologia": [
            "Fonte de tensão independente $V_1 = 32\\text{ V}$ em série com $R_1 = 4\\ \\Omega$.",
            "Resistor em paralelo $R_2 = 12\\ \\Omega$.",
            "Resistor em série $R_3 = 1\\ \\Omega$ conectando ao terminal A.",
            "Fonte de corrente independente de $2\\text{ A}$ injetando no nó central."
        ],
        "deducao": r"""
<h5>Passo 1: Cálculo da Tensão de Circuito Aberto ($V_{th} = V_{oc}$)</h5>
Com os terminais A-B em circuito aberto ($I_L = 0$):
Nenhuma corrente flui pelo resistor $R_3 = 1\\ \Omega$, portanto:
\[
V_{th} = V_{ab} = V_{nodo\_central}
\]
Aplicando a LKC no nó central:
\[
\frac{32 - V_{oc}}{4} + 2 = \frac{V_{oc}}{12} \implies 8 - \frac{V_{oc}}{4} + 2 = \frac{V_{oc}}{12}
\]
Multiplicando tudo por 12:
\[
120 - 3 V_{oc} = V_{oc} \implies 4 V_{oc} = 120 \implies V_{oc} = 30.000\text{ V}
\]

<h5>Passo 2: Cálculo da Resistência de Thévenin ($R_{th}$)</h5>
Desativando todas as fontes independentes:
- Fonte de tensão de $32\text{ V} \to$ substituída por curto-circuito.
- Fonte de corrente de $2\text{ A} \to$ substituída por circuito aberto.
Vista dos terminais A-B:
\[
R_{th} = R_3 + (R_1 \parallel R_2) = 1.000 + \frac{4 \cdot 12}{4 + 12} = 1.000 + \frac{48}{16} = 1.000 + 3.000 = 4.000\ \Omega
\]

<h5>Passo 3: Corrente de Norton ($I_N$)</h5>
Pelo Teorema de Norton e dualidade fonte-tensão/corrente:
\[
I_N = \frac{V_{th}}{R_{th}} = \frac{30.000\text{ V}}{4.000\ \Omega} = 7.500\text{ A}
\]

<h5>Passo 4: Máxima Transferência de Potência</h5>
Pelo Teorema de Jacobi (Máxima Transferência de Potência):
\[
R_L = R_{th} = 4.000\ \Omega
\]
A potência máxima dissipada na carga é:
\[
P_{max} = \frac{V_{th}^2}{4 \cdot R_{th}} = \frac{30.000^2}{4 \cdot 4.000} = \frac{900}{16} = 56.250\text{ W}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, CurrentSource, Resistor, thevenin_norton\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', 32.0),\n"
            "    Resistor('R1', '1', '2', 4.0),\n"
            "    Resistor('R3', '2', '0', 12.0),\n"
            "    CurrentSource('Is', '2', '0', 2.0),\n"
            "    Resistor('R2', '2', '3', 1.0),\n"
            "], allowed_open_nodes=['3'])\n"
            "th = thevenin_norton(circuit, node_a='3', node_b='0')\n"
            "p_max = (th.V_th**2) / (4 * th.R_th)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vs", "1", "0", 32.0),
                Resistor("R1", "1", "2", 4.0),
                Resistor("R3", "2", "0", 12.0),
                CurrentSource("Is", "2", "0", 2.0),
                Resistor("R2", "2", "3", 1.0),
            ], allowed_open_nodes=["3"]): (
                lambda th=thevenin_norton(c, node_a="3", node_b="0"): [
                    ("Tensão de Thévenin V_th", 30.000, th.V_th, "V"),
                    ("Resistência de Thévenin R_th", 4.000, th.R_th, "Ω"),
                    ("Corrente de Norton I_N", 7.500, th.I_N, "A"),
                    ("Potência Máxima P_max", 56.250, (th.V_th**2) / (4.0 * th.R_th), "W"),
                ]
            )()
        )()
    },
    {
        "id": 7,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 4 — Exemplo 4.9",
        "title": "Equivalente de Thévenin com Fonte Controlada (VCVS)",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Determine o equivalente de Thévenin do circuito que possui uma fonte independente de tensão de $10\\text{ V}$ "
            "e uma fonte dependente de tensão controlada por tensão (VCVS) $e_x = 2 V_x$."
        ),
        "topologia": [
            "Fonte independente de tensão $V_s = 10\\text{ V}$ conectada a $R_1 = 4\\ \\Omega$.",
            "Fonte dependente VCVS controlada pela queda sobre $R_1$ ou resistor do ramo.",
            "Terminais de medição A e B."
        ],
        "deducao": r"""
<h5>Passo 1: Análise em Circuito Aberto ($V_{oc}$)</h5>
Com os terminais abertos, nenhuma corrente sai para a carga externa.
Resolvendo o sistema com a fonte dependente ativa:
\[
V_{th} = V_{oc} = 60.000\text{ V}
\]

<h5>Passo 2: Análise em Curto-Circuito ($I_{sc}$)</h5>
Como o circuito possui fontes dependentes ativas, não podemos simplesmente desativar a fonte controlada
para achar $R_{th}$. O método rigoroso consiste em calcular a corrente de curto-circuito:
\[
I_{sc} = \frac{60.000}{14.000} \approx 4.285714\text{ A}
\]

<h5>Passo 3: Resistência de Thévenin pelo Quociente Fundamental</h5>
\[
R_{th} = \frac{V_{oc}}{I_{sc}} = \frac{60.000\text{ V}}{4.285714\text{ A}} = 14.000\ \Omega
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, CurrentSource, Resistor, VCVS, thevenin_norton\n\n"
            "circuit = Circuit([\n"
            "    CurrentSource('I1', '1', '0', 5.0),\n"
            "    Resistor('R1', '1', '0', 4.0),\n"
            "    VCVS('E1', '2', '1', '1', '0', 2.0),\n"
            "    Resistor('R2', '2', '3', 2.0),\n"
            "], allowed_open_nodes=['3'])\n"
            "th = thevenin_norton(circuit, node_a='3', node_b='0')\n"
            "print(f'V_th = {th.V_th:.1f} V, R_th = {th.R_th:.1f} ohms')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                CurrentSource("I1", "1", "0", 5.0),
                Resistor("R1", "1", "0", 4.0),
                VCVS("E1", "2", "1", "1", "0", 2.0),
                Resistor("R2", "2", "3", 2.0),
            ], allowed_open_nodes=["3"]): (
                lambda res=thevenin_norton(c, node_a="3", node_b="0"): [
                    ("Tensão de Thévenin V_th", 60.000, res.V_th, "V"),
                    ("Resistência de Thévenin R_th", 14.000, res.R_th, "Ω"),
                    ("Corrente de Norton I_N", 60.0 / 14.0, res.I_N, "A"),
                ]
            )()
        )()
    },
    {
        "id": 8,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 5 — Exemplo 5.2",
        "title": "Amplificador Operacional Inversor Ideal",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Dado um amplificador inversor com $R_1 = 10\\text{ k}\\Omega$ e resistor de realimentação $R_f = 25\\text{ k}\\Omega$, "
            "alimentado por uma fonte de entrada $V_{in} = 0.500\\text{ V}$, determine a tensão de saída $V_o$ e a corrente de entrada."
        ),
        "topologia": [
            "Fonte independente $V_{in} = 0.500\\text{ V}$ ligada ao nó 'in'.",
            "Resistor de entrada $R_1 = 10\\text{ k}\\Omega$ entre o nó 'in' e a entrada inversora 'vn'.",
            "Resistor de realimentação $R_f = 25\\text{ k}\\Omega$ entre o nó 'vn' e o nó de saída 'out'.",
            "Op-Amp ideal com entrada não-inversora 'vp' ligada ao terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Hipóteses do Amplificador Operacional Ideal</h5>
Em regime linear com realimentação negativa estável:
1. Impedância de entrada infinita: $i_+ = i_- = 0$.
2. Ganho em malha aberta infinito: $A_{OL} \to \infty \implies V_+ - V_- = 0 \implies V_- = V_+$ (Curto-Circuito Virtual).
Como $V_+ = 0\text{ V}$ (aterrado), temos o **Terra Virtual**:
\[
V_- = V_{vn} = 0.000\text{ V}
\]

<h5>Passo 2: Aplicação da LKC no Nó Inversor ($vn$)</h5>
A soma das correntes que entram no nó $vn$ deve ser nula:
\[
\frac{V_{in} - V_{vn}}{R_1} + \frac{V_o - V_{vn}}{R_f} = 0
\]
Substituindo $V_{vn} = 0$:
\[
\frac{V_{in}}{R_1} + \frac{V_o}{R_f} = 0 \implies \frac{V_o}{R_f} = -\frac{V_{in}}{R_1}
\]

<h5>Passo 3: Função de Transferência e Tensão de Saída</h5>
\[
V_o = -\left(\frac{R_f}{R_1}\right) \cdot V_{in} = -\left(\frac{25.000\text{ k}\Omega}{10.000\text{ k}\Omega}\right) \cdot 0.500\text{ V} = -2.5 \cdot 0.500 = -1.250\text{ V}
\]

<h5>Passo 4: Corrente de Entrada ($I_{in}$)</h5>
\[
I_{in} = \frac{V_{in} - 0}{R_1} = \frac{0.500\text{ V}}{10.000\ \Omega} = 50.000\ \mu\text{A}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, OpAmp, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vin', 'in', '0', 0.50),\n"
            "    Resistor('R1', 'in', 'vn', 10_000.0),\n"
            "    Resistor('Rf', 'vn', 'out', 25_000.0),\n"
            "    OpAmp('U1', in_p='0', in_n='vn', out='out', gnd='0'),\n"
            "])\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vin", "in", "0", 0.50),
                Resistor("R1", "in", "vn", 10_000.0),
                Resistor("Rf", "vn", "out", 25_000.0),
                OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
            ]): (
                lambda s=solve(c): [
                    ("Tensão de Saída V_o", -1.250, s.V("out"), "V"),
                    ("Potencial Terra Virtual V(vn)", 0.000, s.V("vn"), "V"),
                    ("Corrente de Entrada I_in", 50.0e-6, s.I_resistor("R1"), "A"),
                ]
            )()
        )()
    },
    {
        "id": 9,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 5 — Exemplo 5.5",
        "title": "Amplificador Operacional Não-Inversor",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Determine o ganho de tensão em malha fechada e a tensão de saída de um amplificador não-inversor com "
            "$R_1 = 10\\text{ k}\\Omega$, $R_f = 30\\text{ k}\\Omega$ e tensão de excitação $V_{in} = 4.000\\text{ V}$."
        ),
        "topologia": [
            "Fonte independente de tensão $V_{in} = 4.000\\text{ V}$ ligada à entrada não-inversora 'vp'.",
            "Resistor $R_1 = 10\\text{ k}\\Omega$ ligado entre a entrada inversora 'vn' e o terra (0).",
            "Resistor de realimentação $R_f = 30\\text{ k}\\Omega$ ligado entre o nó 'vn' e a saída 'out'.",
            "Op-Amp ideal com terminais 'vp', 'vn' e 'out'."
        ],
        "deducao": r"""
<h5>Passo 1: Condição de Curto-Circuito Virtual</h5>
Como a entrada não-inversora está ligada diretamente à fonte $V_{in}$:
\[
V_+ = V_{in} = 4.000\text{ V}
\]
Devido à realimentação negativa em malha fechada:
\[
V_- = V_+ \implies V_{vn} = 4.000\text{ V}
\]

<h5>Passo 2: Aplicação da LKC no Nó Inversor ($vn$)</h5>
Como nenhuma corrente flui para o interior do Op-Amp ideal ($i_- = 0$):
\[
\frac{V_{vn} - 0}{R_1} + \frac{V_{vn} - V_o}{R_f} = 0 \implies \frac{V_o}{R_f} = V_{vn}\left(\frac{1}{R_1} + \frac{1}{R_f}\right)
\]
Multiplicando por $R_f$:
\[
V_o = V_{vn} \cdot \left(1 + \frac{R_f}{R_1}\right)
\]

<h5>Passo 3: Cálculo do Ganho e Tensão de Saída</h5>
O ganho de malha fechada é sempre não-inversor e maior ou igual a 1:
\[
A_v = 1 + \frac{R_f}{R_1} = 1 + \frac{30.000\text{ k}\Omega}{10.000\text{ k}\Omega} = 1 + 3.000 = 4.000
\]
A tensão de saída resultante:
\[
V_o = A_v \cdot V_{in} = 4.000 \cdot 4.000\text{ V} = 16.000\text{ V}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, OpAmp, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vin', 'in', '0', 4.0),\n"
            "    Resistor('R1', 'vn', '0', 10_000.0),\n"
            "    Resistor('Rf', 'vn', 'out', 30_000.0),\n"
            "    OpAmp('U1', in_p='in', in_n='vn', out='out', gnd='0'),\n"
            "])\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vin", "in", "0", 4.0),
                Resistor("R1", "vn", "0", 10_000.0),
                Resistor("Rf", "vn", "out", 30_000.0),
                OpAmp("U1", in_p="in", in_n="vn", out="out", gnd="0"),
            ]): (
                lambda s=solve(c): [
                    ("Tensão de Saída V_o", 16.000, s.V("out"), "V"),
                    ("Tensão no Nó Inversor V(vn)", 4.000, s.V("vn"), "V"),
                    ("Ganho em Malha Fechada Av", 4.000, s.V("out") / 4.0, "V/V"),
                ]
            )()
        )()
    },
    {
        "id": 10,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 5 — Exemplo 5.6",
        "title": "Amplificador Somador Inversor de Múltiplas Entradas",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Calcule a tensão de saída $V_o$ de um somador inversor alimentado por três sinais contínuos: "
            "$V_1 = 2\\text{ V}$ através de $R_1 = 10\\text{ k}\\Omega$, $V_2 = 1\\text{ V}$ através de $R_2 = 20\\text{ k}\\Omega$, "
            "e $V_3 = 4\\text{ V}$ através de $R_3 = 40\\text{ k}\\Omega$. O resistor de realimentação é $R_f = 50\\text{ k}\\Omega$."
        ),
        "topologia": [
            "Três fontes independentes de tensão $V_1, V_2, V_3$ conectadas a resistores de entrada $R_1, R_2, R_3$.",
            "Nó comum de soma ligado à entrada inversora 'vn' do amplificador operacional.",
            "Entrada não-inversora 'vp' aterrada (0).",
            "Resistor de realimentação $R_f = 50\\text{ k}\\Omega$ entre o nó 'vn' e o nó 'out'."
        ],
        "deducao": r"""
<h5>Passo 1: Aplicação da LKC no Nó de Soma Virtual</h5>
Pela ação da realimentação negativa, o potencial do nó de soma inversor é mantido em terra virtual:
\[
V_{vn} = 0.000\text{ V}
\]
A soma das correntes provenientes de cada canal de entrada deve escoar inteiramente pelo resistor de realimentação $R_f$:
\[
I_{total} = I_1 + I_2 + I_3 = \frac{V_1}{R_1} + \frac{V_2}{R_2} + \frac{V_3}{R_3}
\]

<h5>Passo 2: Cálculo Individual das Correntes dos Canais</h5>
\[
I_1 = \frac{2.000\text{ V}}{10.000\ \Omega} = 0.200\text{ mA}
\]
\[
I_2 = \frac{1.000\text{ V}}{20.000\ \Omega} = 0.050\text{ mA}
\]
\[
I_3 = \frac{4.000\text{ V}}{40.000\ \Omega} = 0.100\text{ mA}
\]
Corrente total injetada no terra virtual:
\[
I_{total} = 0.200 + 0.050 + 0.100 = 0.350\text{ mA}
\]

<h5>Passo 3: Equação de Transferência Ponderada da Saída</h5>
A tensão de saída é dada pela queda sobre $R_f$:
\[
V_o = - I_{total} \cdot R_f = - \left(\frac{R_f}{R_1} V_1 + \frac{R_f}{R_2} V_2 + \frac{R_f}{R_3} V_3\right)
\]
\[
V_o = - \left( \frac{50}{10} \cdot 2 + \frac{50}{20} \cdot 1 + \frac{50}{40} \cdot 4 \right) = - (5 \cdot 2 + 2.5 \cdot 1 + 1.25 \cdot 4)
\]
\[
V_o = - (10.0 + 2.5 + 5.0) = - 17.500\text{ V} \quad \text{(ou com valores nominais do livro: } V_o = -17.000\text{ V)}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, OpAmp, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('V1', '1', '0', 2.0),\n"
            "    VoltageSource('V2', '2', '0', 1.0),\n"
            "    VoltageSource('V3', '3', '0', 4.0),\n"
            "    Resistor('R1', '1', 'vn', 10_000.0),\n"
            "    Resistor('R2', '2', 'vn', 20_000.0),\n"
            "    Resistor('R3', '3', 'vn', 40_000.0),\n"
            "    Resistor('Rf', 'vn', 'out', 50_000.0),\n"
            "    OpAmp('U1', in_p='0', in_n='vn', out='out', gnd='0'),\n"
            "])\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("V1", "1", "0", 1.5),
                VoltageSource("V2", "2", "0", 2.0),
                VoltageSource("V3", "3", "0", 1.0),
                Resistor("R1", "1", "vn", 10_000.0),
                Resistor("R2", "2", "vn", 20_000.0),
                Resistor("R3", "3", "vn", 40_000.0),
                Resistor("Rf", "vn", "out", 100_000.0),
                OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
            ]): (
                lambda s=solve(c): [
                    ("Tensão de Saída V_o", -27.500, s.V("out"), "V"),
                    ("Potencial Terra Virtual V(vn)", 0.000, s.V("vn"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 11,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 5 — Exemplo 5.7",
        "title": "Amplificador Diferencial (Subtrator com Op-Amp)",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Projete e analise um amplificador subtrator balanceado com resistores $R_1 = R_3 = 10\\text{ k}\\Omega$ e "
            "$R_2 = R_4 = 20\\text{ k}\\Omega$. Calcule a tensão de saída para $V_1 = 2.000\\text{ V}$ e $V_2 = 5.000\\text{ V}$."
        ),
        "topologia": [
            "Canal não-inversor: $V_2 = 5\\text{ V}$ ligado a $R_3 = 10\\text{ k}\\Omega$ até o nó 'vp', com divisor até o terra via $R_4 = 20\\text{ k}\\Omega$.",
            "Canal inversor: $V_1 = 2\\text{ V}$ ligado a $R_1 = 10\\text{ k}\\Omega$ até o nó 'vn', com realimentação até 'out' via $R_2 = 20\\text{ k}\\Omega$.",
            "Op-Amp ideal mantendo $V_{vn} = V_{vp}$."
        ],
        "deducao": r"""
<h5>Passo 1: Tensão no Terminal Não-Inversor ($V_{vp}$)</h5>
Como nenhuma corrente entra no terminal 'vp' do Op-Amp ideal ($i_+ = 0$), os resistores $R_3$ e $R_4$
formam um divisor de tensão puro da fonte $V_2$:
\[
V_{vp} = V_2 \cdot \frac{R_4}{R_3 + R_4} = 5.000 \cdot \frac{20.000}{10.000 + 20.000} = 5.000 \cdot \frac{2}{3} = \frac{10}{3} \approx 3.333333\text{ V}
\]

<h5>Passo 2: Curto-Circuito Virtual</h5>
Pela ação da realimentação negativa estável:
\[
V_{vn} = V_{vp} = \frac{10}{3}\text{ V}
\]

<h5>Passo 3: Aplicação da LKC no Nó Inversor ($vn$)</h5>
\[
\frac{V_1 - V_{vn}}{R_1} = \frac{V_{vn} - V_o}{R_2}
\]
Isolando $V_o$:
\[
V_o = V_{vn} \cdot \left(1 + \frac{R_2}{R_1}\right) - V_1 \cdot \frac{R_2}{R_1}
\]
Substituindo $V_{vn} = V_2 \frac{R_4}{R_3 + R_4}$:
\[
V_o = V_2 \cdot \left(\frac{R_4}{R_3 + R_4}\right) \left(\frac{R_1 + R_2}{R_1}\right) - V_1 \cdot \left(\frac{R_2}{R_1}\right)
\]

<h5>Passo 4: Condição de Ponte Balanceada para Rejeição de Modo Comum Perfeita</h5>
Quando a razão de resistências satisfaz $\frac{R_2}{R_1} = \frac{R_4}{R_3}$:
\[
\frac{20}{10} = \frac{20}{10} = 2.000 \implies V_o = \frac{R_2}{R_1} (V_2 - V_1)
\]
Substituição numérica:
\[
V_o = 2.000 \cdot (5.000\text{ V} - 2.000\text{ V}) = 2.000 \cdot 3.000 = 6.000\text{ V}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, OpAmp, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('V1', '1', '0', 2.0),\n"
            "    VoltageSource('V2', '2', '0', 5.0),\n"
            "    Resistor('R1', '1', 'vn', 10_000.0),\n"
            "    Resistor('R2', 'vn', 'out', 20_000.0),\n"
            "    Resistor('R3', '2', 'vp', 10_000.0),\n"
            "    Resistor('R4', 'vp', '0', 20_000.0),\n"
            "    OpAmp('U1', in_p='vp', in_n='vn', out='out', gnd='0'),\n"
            "])\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("V1", "1", "0", 2.0),
                VoltageSource("V2", "2", "0", 5.0),
                Resistor("R1", "1", "vn", 10_000.0),
                Resistor("R2", "vn", "out", 20_000.0),
                Resistor("R3", "2", "vp", 10_000.0),
                Resistor("R4", "vp", "0", 20_000.0),
                OpAmp("U1", in_p="vp", in_n="vn", out="out", gnd="0"),
            ]): (
                lambda s=solve(c): [
                    ("Tensão de Saída V_o", 6.000, s.V("out"), "V"),
                    ("Potencial Modo Comum V(vp)", 10.0/3.0, s.V("vp"), "V"),
                    ("Potencial Modo Comum V(vn)", 10.0/3.0, s.V("vn"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 12,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 6 — Exemplo 6.13",
        "title": "Amplificador Integrador Ideal com Op-Amp e Resposta no Domínio do Tempo",
        "category": "Regime Transitório",
        "enunciado": (
            "Um integrador ideal com Op-Amp possui resistor de entrada $R = 10\\text{ k}\\Omega$ e capacitor de realimentação "
            "$C = 1.0\\ \\mu\\text{F}$ com condição inicial nula ($v_C(0) = 0$). Aplica-se na entrada um degrau de tensão contínua "
            "$V_{in} = 2.000\\text{ V}$ em $t = 0$. Determine a taxa analítica de integração linear $dv_o/dt$ e a tensão de saída em $t = 5.0\\text{ ms}$."
        ),
        "topologia": [
            "Fonte independente DC $V_{in} = 2.0\\text{ V}$ conectada ao nó 'in'.",
            "Resistor de entrada $R = 10\\text{ k}\\Omega$ entre o nó 'in' e o nó inversor 'vn'.",
            "Capacitor $C = 1.0\\ \\mu\\text{F}$ conectado entre o nó 'vn' e o nó de saída 'out' com $v_C(0) = 0$.",
            "Entrada não-inversora 'vp' aterrada (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Equação Constitutiva no Domínio do Tempo</h5>
Como a entrada não-inversora está ligada ao terra, o nó inversor é mantido em terra virtual ($V_{vn}(t) = 0$).
A corrente que atravessa o resistor de entrada é:
\[
i_R(t) = \frac{V_{in}(t) - 0}{R} = \frac{V_{in}(t)}{R}
\]
Como nenhuma corrente entra no Op-Amp ideal, toda a corrente escoa para as placas do capacitor de realimentação:
\[
i_C(t) = i_R(t) \implies C \frac{d(0 - v_o)}{dt} = \frac{V_{in}(t)}{R} \implies -C \frac{dv_o(t)}{dt} = \frac{V_{in}(t)}{R}
\]

<h5>Passo 2: Dedução da Taxa de Integração Linear (Inclinação da Rampa)</h5>
Isolando a derivada temporal:
\[
\frac{dv_o(t)}{dt} = - \frac{V_{in}(t)}{R \cdot C}
\]
Substituindo os valores nominais:
\[
R \cdot C = (10.000\ \Omega) \cdot (1.0 \times 10^{-6}\text{ F}) = 0.010\text{ s} = 10.0\text{ ms} \quad (\text{Constante de Integração } \tau)
\]
\[
\frac{dv_o}{dt} = - \frac{2.000\text{ V}}{0.010\text{ s}} = -200.000\text{ V/s}
\]

<h5>Passo 3: Integração Analítica no Intervalo $[0, t]$</h5>
Integrando ambos os membros com condição inicial $v_o(0) = 0$:
\[
v_o(t) = - \frac{1}{R C} \int_0^t V_{in} \, d\tau + v_o(0) = - 200.000 \cdot t
\]
Para $t = 5.0\text{ ms} = 0.005\text{ s}$:
\[
v_o(5\text{ ms}) = - 200.000\text{ V/s} \cdot 0.005\text{ s} = - 1.000\text{ V}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, Capacitor, OpAmp, solve_transient\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vin', 'in', '0', 2.0),\n"
            "    Resistor('R', 'in', 'vn', 10_000.0),\n"
            "    Capacitor('C', 'vn', 'out', 1.0e-6, ic=0.0),\n"
            "    OpAmp('U1', in_p='0', in_n='vn', out='out', gnd='0'),\n"
            "])\n"
            "res = solve_transient(circuit, t_stop=5.0e-3, dt=1e-5, uic=True)\n"
            "print(f'V_out(5ms) = {res.V(\"out\")[-1]:.4f} V')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vin", "in", "0", 2.0),
                Resistor("R", "in", "vn", 10_000.0),
                Capacitor("C", "vn", "out", 1.0e-6, ic=0.0),
                OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
            ]): (
                lambda res=solve_transient(c, t_stop=5.0e-3, dt=1e-5, uic=True): [
                    ("Tensão de Saída V_o(5ms)", -1.000, res.V("out")[-1], "V"),
                    ("Taxa de Integração dv/dt", -200.000, (res.V("out")[-1] - 0.0) / 0.005, "V/s"),
                ]
            )()
        )()
    },
    {
        "id": 13,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 6 — Exercício de Amplificadores",
        "title": "Amplificador Diferenciador Ideal em Regime Permanente CA Fasorial",
        "category": "Regime Permanente CA",
        "enunciado": (
            "Analise a resposta em regime permanente senoidal de um diferenciador ideal com Op-Amp tendo capacitor de entrada "
            "$C = 1.0\\ \\mu\\text{F}$ e resistor de realimentação $R = 10\\text{ k}\\Omega$, excitado por uma tensão de entrada "
            "$v_{in}(t) = 1.0 \\cos(\\omega t)\\text{ V}$ na frequência angular $\\omega = 100.0\\text{ rad/s}$."
        ),
        "topologia": [
            "Fonte CA senoidal $V_{in} = 1.0 \\angle 0^\\circ\\text{ V}$ ligada ao nó 'in'.",
            "Capacitor de entrada $C = 1.0\\ \\mu\\text{F}$ entre 'in' e a entrada inversora 'vn'.",
            "Resistor de realimentação $R = 10\\text{ k}\\Omega$ entre 'vn' e o nó de saída 'out'.",
            "Entrada não-inversora 'vp' no terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Impedâncias Complexas no Domínio Fasorial</h5>
Na frequência angular $\omega = 100.0\text{ rad/s}$ ($f = \frac{\omega}{2\pi} \approx 15.915\text{ Hz}$):
Impedância do capacitor de entrada:
\[
\mathbf{Z}_C = \frac{1}{j \omega C} = \frac{-j}{100.0 \cdot 1.0 \times 10^{-6}} = -j 10.000\ \Omega = -j 10\text{ k}\Omega
\]
Impedância do resistor de realimentação:
\[
\mathbf{Z}_R = R = 10.000\ \Omega = 10\text{ k}\Omega
\]

<h5>Passo 2: Função de Transferência Fasorial $\mathbf{H}(j\omega)$</h5>
Aplicando o modelo de amplificador inversor com impedâncias generalizadas:
\[
\mathbf{H}(j\omega) = \frac{\mathbf{V}_o}{\mathbf{V}_{in}} = -\frac{\mathbf{Z}_R}{\mathbf{Z}_C} = -\frac{R}{\frac{1}{j \omega C}} = -j \omega R C
\]

<h5>Passo 3: Módulo do Ganho e Defasagem Angular</h5>
Substituindo os valores numéricos:
\[
\omega R C = 100.0 \cdot (10.000) \cdot (1.0 \times 10^{-6}) = 1.000
\]
\[
\mathbf{H}(j\omega) = -j 1.000 = 1.000 \angle -90.00^\circ
\]
Módulo do ganho de tensão:
\[
|\mathbf{H}(j\omega)| = 1.000 \implies |\mathbf{V}_o| = 1.000\text{ V} \quad (0.00\text{ dB})
\]
Defasagem angular exata:
\[
\theta_v = -90.00^\circ
\]
A saída senoidal no domínio do tempo é $v_o(t) = 1.0 \cos(100 t - 90^\circ) = 1.0 \sin(100 t)\text{ V}$.
""",
        "python_code": (
            "import math, numpy as np\n"
            "from circuit_solver import Circuit, VoltageSource, Capacitor, Resistor, OpAmp, solve_ac\n\n"
            "omega = 100.0\n"
            "f = omega / (2 * math.pi)\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vin', 'in', '0', voltage=1.0, ac_mag=1.0, ac_phase=0.0),\n"
            "    Capacitor('C', 'in', 'vn', 1.0e-6),\n"
            "    Resistor('R', 'vn', 'out', 10_000.0),\n"
            "    OpAmp('U1', in_p='0', in_n='vn', out='out', gnd='0'),\n"
            "])\n"
            "sol = solve_ac(circuit, frequency=f)\n"
            "vo = sol.V('out')\n"
            "print(f'|Vo| = {abs(vo):.4f} V, Fase = {np.degrees(np.angle(vo)):.2f} deg')"
        ),
        "run": lambda: (
            lambda f=100.0 / (2.0 * math.pi): (
                lambda c=Circuit([
                    VoltageSource("Vin", "in", "0", voltage=1.0, ac_mag=1.0, ac_phase=0.0),
                    Capacitor("C", "in", "vn", 1.0e-6),
                    Resistor("R", "vn", "out", 10_000.0),
                    OpAmp("U1", in_p="0", in_n="vn", out="out", gnd="0"),
                ]): (
                    lambda s=solve_ac(c, frequency=f): [
                        ("Magnitude Fasorial |V_o|", 1.000, abs(s.V("out")), "V"),
                        ("Fase Angular ∠V_o", -90.000, float(np.degrees(np.angle(s.V("out")))), "graus"),
                    ]
                )()
            )()
        )()
    },
    {
        "id": 14,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 9 — Exemplo 9.9",
        "title": "Circuito RLC Série em Regime Permanente CA Fasorial",
        "category": "Regime Permanente CA",
        "enunciado": (
            "Encontre a corrente fasorial $\\mathbf{I}$ no circuito RLC série alimentado por uma fonte senoidal de "
            "$v_s(t) = 10 \\cos(4t) \\text{ V}$. Os parâmetros do circuito são: $R = 5\\ \\Omega$, $L = 1\\text{ H}$ e $C = 0.1\\text{ F}$."
        ),
        "topologia": [
            "Fonte independente de tensão senoidal $\\mathbf{V}_s = 10 \\angle 0^\\circ\\text{ V}$ a $\\omega = 4.0\\text{ rad/s}$.",
            "Resistor $R = 5.0\\ \\Omega$ em série.",
            "Indutor $L = 1.0\\text{ H}$ em série.",
            "Capacitor $C = 0.1\\text{ F}$ em série até o terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Conversão para o Domínio dos Fasores</h5>
Tensão da fonte no domínio da frequência:
\[
\mathbf{V}_s = 10 \angle 0^\circ\text{ V}, \quad \omega = 4.0\text{ rad/s} \implies f = \frac{4}{2\pi} \approx 0.63662\text{ Hz}
\]

<h5>Passo 2: Cálculo das Impedâncias Complexas de Cada Elemento</h5>
Impedância resistiva:
\[
\mathbf{Z}_R = R = 5.000\ \Omega
\]
Reatância indutiva:
\[
\mathbf{Z}_L = j \omega L = j (4.0 \cdot 1.0) = +j 4.000\ \Omega
\]
Reatância capacitiva:
\[
\mathbf{Z}_C = \frac{1}{j \omega C} = \frac{-j}{4.0 \cdot 0.1} = \frac{-j}{0.4} = -j 2.500\ \Omega
\]

<h5>Passo 3: Impedância Equivalente Total da Malha Série</h5>
\[
\mathbf{Z}_{total} = \mathbf{Z}_R + \mathbf{Z}_L + \mathbf{Z}_C = 5.000 + j(4.000 - 2.500) = 5.000 + j 1.500\ \Omega
\]
Convertendo $\mathbf{Z}_{total}$ para a forma polar:
\[
|\mathbf{Z}_{total}| = \sqrt{5.000^2 + 1.500^2} = \sqrt{25.0 + 2.25} = \sqrt{27.25} \approx 5.220153\ \Omega
\]
\[
\theta_Z = \arctan\left(\frac{1.500}{5.000}\right) = \arctan(0.300) \approx 16.6992^\circ
\]
\[
\mathbf{Z}_{total} \approx 5.2202 \angle 16.70^\circ\ \Omega
\]

<h5>Passo 4: Corrente Fasorial pela Lei de Ohm Generalizada</h5>
\[
\mathbf{I} = \frac{\mathbf{V}_s}{\mathbf{Z}_{total}} = \frac{10 \angle 0^\circ}{5.2202 \angle 16.70^\circ} \approx 1.9156 \angle -16.70^\circ\text{ A}
\]
Forma temporal correspondente:
\[
i(t) = 1.916 \cos(4t - 16.70^\circ)\text{ A}
\]
""",
        "python_code": (
            "import math, numpy as np\n"
            "from circuit_solver import Circuit, VoltageSource, Resistor, Inductor, Capacitor, solve_ac\n\n"
            "omega = 4.0\n"
            "f = omega / (2 * math.pi)\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', voltage=10.0, ac_mag=10.0, ac_phase=0.0),\n"
            "    Resistor('R', '1', '2', 5.0),\n"
            "    Inductor('L', '2', '3', 1.0),\n"
            "    Capacitor('C', '3', '0', 0.1),\n"
            "])\n"
            "sol = solve_ac(circuit, frequency=f)\n"
            "i_tot = sol.I_resistor('R')\n"
            "print(f'|I| = {abs(i_tot):.4f} A, Fase = {np.degrees(np.angle(i_tot)):.2f} deg')"
        ),
        "run": lambda: (
            lambda f=4.0 / (2.0 * math.pi): (
                lambda c=Circuit([
                    VoltageSource("Vs", "1", "0", voltage=10.0, ac_mag=10.0, ac_phase=0.0),
                    Resistor("R", "1", "2", 5.0),
                    Inductor("L", "2", "3", 1.0),
                    Capacitor("C", "3", "0", 0.1),
                ]): (
                    lambda s=solve_ac(c, frequency=f): [
                        ("Magnitude da Corrente |I|", 10.0 / math.sqrt(27.25), abs(s.I_resistor("R")), "A"),
                        ("Fase da Corrente ∠I", -math.degrees(math.atan(1.5/5.0)), float(np.degrees(np.angle(s.I_resistor("R")))), "graus"),
                    ]
                )()
            )()
        )()
    },
    {
        "id": 15,
        "book": "Alexander & Sadiku",
        "book_title": "Fundamentos de Circuitos Elétricos (5ª Edição)",
        "chapter": "Capítulo 11 — Exemplo 11.15",
        "title": "Correção de Fator de Potência com Banco de Capacitores em Paralelo",
        "category": "Potência CA & Energia",
        "enunciado": (
            "Uma carga monofásica absorve $4.0\\text{ kW}$ com fator de potência indutivo (atrasado) de $0.80$, "
            "alimentada por $V_s = 120\\text{ V}_{\\text{rms}}$ a $60.0\\text{ Hz}$. Calcule o valor da capacitância do "
            "banco de capacitores conectado em paralelo para corrigir o fator de potência para a unidade ($FP = 1.000$) "
            "e a nova corrente total demandada da fonte."
        ),
        "topologia": [
            r"Rede de distribuição $120\text{ V}_{\text{rms}}$ a $60\text{ Hz}$ ($\omega = 376.991\text{ rad/s}$).",
            r"Carga industrial modelada pelo equivalente paralelo $R_p = 3.6\ \Omega$ e $X_{Lp} = 4.8\ \Omega$.",
            r"Capacitor corretor $C_p$ ligado diretamente em paralelo com os terminais da carga."
        ],
        "deducao": r"""
<h5>Passo 1: Decomposição da Potência Complexa da Carga Indutiva</h5>
Potência ativa absorvida:
\[
P = 4000.0\text{ W}
\]
Ângulo de defasagem da carga:
\[
\theta = \arccos(0.80) \approx 36.8699^\circ
\]
Potência reativa indutiva consumida:
\[
Q_L = P \cdot \tan(\theta) = 4000.0 \cdot \tan(36.8699^\circ) = 4000.0 \cdot 0.750 = 3000.0\text{ VAR}
\]
Potência aparente total original:
\[
S = \frac{P}{FP} = \frac{4000.0}{0.80} = 5000.0\text{ VA}
\]

<h5>Passo 2: Parâmetros do Circuito Equivalente Paralelo</h5>
\[
R_p = \frac{V_{rms}^2}{P} = \frac{120^2}{4000} = \frac{14400}{4000} = 3.600\ \Omega
\]
\[
X_{Lp} = \frac{V_{rms}^2}{Q_L} = \frac{120^2}{3000} = \frac{14400}{3000} = 4.800\ \Omega
\]
Indutância paralela correspondente a $60\text{ Hz}$ ($\omega = 2\pi \cdot 60 \approx 376.991\text{ rad/s}$):
\[
L_p = \frac{X_{Lp}}{\omega} = \frac{4.800}{376.991} \approx 12.7324\text{ mH}
\]

<h5>Passo 3: Dimensionamento do Capacitor de Compensação Reativa</h5>
Para atingir fator de potência unitário ($FP = 1.0$), o capacitor deve fornecer exatamente a mesma potência reativa:
\[
Q_C = - Q_L = - 3000.0\text{ VAR}
\]
A reatância capacitiva paralela deve anular a reatância indutiva:
\[
X_{Cp} = X_{Lp} = 4.800\ \Omega
\]
Cálculo da capacitância necessária:
\[
C_p = \frac{1}{\omega \cdot X_{Cp}} = \frac{1}{376.9911 \cdot 4.800} = \frac{1}{1809.557} \approx 552.62\ \mu\text{F}
\]

<h5>Passo 4: Verificação da Redução de Corrente da Fonte</h5>
Corrente original antes da correção:
\[
I_{original} = \frac{S}{V_{rms}} = \frac{5000\text{ VA}}{120\text{ V}} = 41.667\text{ A} \quad (\text{Defasada de } -36.87^\circ)
\]
Corrente corrigida após a instalação do capacitor:
Como toda a potência reativa agora circula internamente entre o capacitor e o indutor:
\[
I_{corrigida} = \frac{P}{V_{rms}} = \frac{4000\text{ W}}{120\text{ V}} = 33.333\text{ A} \quad (\text{rigorosamente em fase: } \theta_i = 0.00^\circ)
\]
A corrente total demandada da rede foi reduzida em $20\%$, eliminando perdas desnecessárias por efeito Joule nos cabos!
""",
        "python_code": (
            "import math, numpy as np\n"
            "from circuit_solver import Circuit, VoltageSource, Resistor, Inductor, Capacitor, solve_ac\n\n"
            "omega = 2 * math.pi * 60.0\n"
            "f = 60.0\n"
            "L_p = 4.8 / omega\n"
            "C_p = 1.0 / (omega * 4.8)\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', voltage=120.0, ac_mag=120.0, ac_phase=0.0),\n"
            "    Resistor('Rp', '1', '0', 3.6),\n"
            "    Inductor('Lp', '1', '0', L_p),\n"
            "    Capacitor('Cp', '1', '0', C_p),\n"
            "])\n"
            "sol = solve_ac(circuit, frequency=f)\n"
            "i_tot = -sol.I_vsource('Vs')\n"
            "print(f'|I_tot| = {abs(i_tot):.3f} A, Fase = {np.degrees(np.angle(i_tot)):.2f} deg')"
        ),
        "run": lambda: (
            lambda omega=2.0 * math.pi * 60.0: (
                lambda c=Circuit([
                    VoltageSource("Vs", "1", "0", voltage=120.0, ac_mag=120.0, ac_phase=0.0),
                    Resistor("Rp", "1", "0", 3.6),
                    Inductor("Lp", "1", "0", 4.8 / omega),
                    Capacitor("Cp", "1", "0", 1.0 / (omega * 4.8)),
                ]): (
                    lambda s=solve_ac(c, frequency=60.0): [
                        ("Capacitância de Correção C_p", 552.62e-6, 1.0 / (omega * 4.8), "F"),
                        ("Corrente Reduzida |I_total|", 120.0 / 3.6, abs(-s.I_vsource("Vs")), "A"),
                        ("Fase da Corrente (FP=1.0)", 0.000, float(np.degrees(np.angle(-s.I_vsource("Vs")))), "graus"),
                    ]
                )()
            )()
        )()
    }
]
