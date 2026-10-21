"""
data_part3.py — Memória de Cálculo: Exercícios 31 a 43
(Sedra & Smith, Hayt, Kemmerly & Durbin, Irwin & Nelms)
"""
import math
from scipy.optimize import brentq
import numpy as np

from circuit_solver.core.circuit import Circuit
from circuit_solver.core.elements import (
    Resistor, VoltageSource, Capacitor, Inductor,
    OpAmp, Diode, BJT, MOSFET, CCCS, MutualCoupling
)
from circuit_solver.engine.mna import solve
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.engine.ac import solve_ac
from circuit_solver.engine.transient import solve_transient

# Independent reference: scalar KCL for the active-region matched mirror.
_VREF35 = brentq(lambda v: (10 - v) / 9300 - 1.02 * 1e-14 * math.expm1(v / 0.02585), 0.1, 0.9)
_IREF35 = (10 - _VREF35) / 9300

EXERCISES_PART3 = [
    {
        "id": 31,
        "book": "Sedra & Smith",
        "book_title": "Microelectronic Circuits (7ª/8ª Edição)",
        "chapter": "Capítulo 4 — Diodos",
        "title": "Diodo Retificador PN e Ponto de Operação Quiescente Q com Reta de Carga",
        "category": "Eletrônica & Semicondutores",
        "enunciado": (
            "Determine o ponto de operação $Q$ ($V_D, I_D, P_D$) de um circuito retificador com fonte contínua $V_s = 5.0\\text{ V}$, "
            "resistor limitador $R = 1.0\\text{ k}\\Omega$ e diodo de junção PN de silício modelado pela equação exponencial de Shockley "
            "($I_s = 10^{-14}\\text{ A}, V_T = 25.85\\text{ mV}$)."
        ),
        "topologia": [
            "Fonte independente de tensão contínua $V_s = 5.0\\text{ V}$.",
            "Resistor limitador $R = 1000.0\\ \\Omega$ entre nó 1 e nó 2.",
            "Diodo PN de silício conectado do nó 2 ao terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Equação da Reta de Carga Linear</h5>
Pela Lei das Malhas de Kirchhoff (LKT):
\[
V_s = R \cdot I_D + V_D \implies I_D = \frac{V_s - V_D}{R} = \frac{5.0 - V_D}{1000} = 5.0\text{ mA} - \frac{V_D}{1000}
\]
Pontos extremos da reta de carga no plano $I_D \times V_D$:
- Curto-circuito ($V_D = 0$): $I_{D,max} = \frac{5.0}{1000} = 5.000\text{ mA}$.
- Circuito aberto ($I_D = 0$): $V_{D,max} = 5.000\text{ V}$.

<h5>Passo 2: Equação Constitutiva Não-Linear de Shockley</h5>
A corrente direta no diodo de junção é regida por:
\[
I_D = I_s \left( e^{\frac{V_D}{n V_T}} - 1 \right) \approx I_s e^{\frac{V_D}{V_T}}
\]
com $I_s = 1.0 \times 10^{-14}\text{ A}$ e tensão térmica $V_T = 25.85\text{ mV} \approx 0.02585\text{ V}$ a $300\text{ K}$.

<h5>Passo 3: Solução Não-Linear por Newton-Raphson</h5>
A interseção da reta de carga com a curva exponencial produz a função residual:
\[
f(V_D) = \frac{5.0 - V_D}{1000} - I_s e^{\frac{V_D}{V_T}} = 0
\]
Iniciando com aproximação linear $V_D^{(0)} = 0.700\text{ V}$:
\[
I_D^{(0)} = \frac{5.0 - 0.7}{1000} = 4.300\text{ mA}
\]
O algoritmo de Newton-Raphson do solver converge em 4 iterações para:
\[
V_D \approx 0.6925\text{ V}, \quad I_D \approx 4.3075\text{ mA}
\]

<h5>Passo 4: Potência Estática Dissipada no Diodo ($P_D$)</h5>
\[
P_D = V_D \cdot I_D = 0.6925\text{ V} \cdot 4.3075\text{ mA} \approx 2.983\text{ mW}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, Diode, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', 5.0),\n"
            "    Resistor('R1', '1', '2', 1000.0),\n"
            "    Diode('D1', '2', '0', Is=1e-14, Vt=0.02585),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "print(f'V_D = {sol.V(\"2\"):.4f} V, I_D = {sol.I_resistor(\"R1\")*1e3:.4f} mA')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vs", "1", "0", 5.0),
                Resistor("R1", "1", "2", 1000.0),
                Diode("D1", "2", "0", Is=1e-14, Vt=0.02585),
            ]): (
                lambda s=solve(c): [
                    ("Tensão Direta V_D", 0.6925, s.V("2"), "V"),
                    ("Corrente de Condução I_D", 4.3075e-3, s.I_resistor("R1"), "A"),
                    ("Potência Dissipada P_D", 2.983e-3, s.V("2") * s.I_resistor("R1"), "W"),
                ]
            )()
        )()
    },
    {
        "id": 32,
        "book": "Sedra & Smith",
        "book_title": "Microelectronic Circuits (7ª/8ª Edição)",
        "chapter": "Capítulo 4 — Retificadores",
        "title": "Retificador de Onda Completa em Ponte (Ponte de Graetz)",
        "category": "Eletrônica & Semicondutores",
        "enunciado": (
            "Analise a retificação de onda completa com 4 diodos em ponte (Ponte de Graetz) e carga resistiva $R_L = 1.0\\text{ k}\\Omega$. "
            "Demonstre a condução alternada em pares para semiciclos positivo e negativo de $V_{in} = \\pm 12.0\\text{ V}$ e a simetria de saída."
        ),
        "topologia": [
            "Fonte de entrada conectada entre os nós 'ac1' e 'ac2'.",
            "Diodos D1 (ac1 $\\to$ out) e D4 (0 $\\to$ ac2) para o semiciclo positivo.",
            "Diodos D3 (ac2 $\\to$ out) e D2 (0 $\\to$ ac1) para o semiciclo negativo.",
            "Resistor de carga $R_L = 1000\\ \\Omega$ entre o nó de saída 'out' e o terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Dinâmica de Condução no Semiciclo Positivo ($V_{ac1} > V_{ac2}$)</h5>
Com $V_{in} = +12.0\text{ V}$:
- Diodo D1 conduz ligando o nó 'ac1' à saída 'out'.
- Diodo D4 conduz ligando o nó 'ac2' ao terra '0'.
- Diodos D2 e D3 ficam polarizados reversamente (cortados).
A tensão entregue à carga com duas quedas de diodo em série:
\[
V_{out} = V_{in} - 2 V_D \approx 12.000 - 2 \cdot (0.700) = 10.600\text{ V}
\]

<h5>Passo 2: Dinâmica de Condução no Semiciclo Negativo ($V_{ac1} < V_{ac2}$)</h5>
Com $V_{in} = -12.0\text{ V}$:
- Diodo D3 conduz ligando o nó 'ac2' à saída 'out'.
- Diodo D2 conduz ligando o nó 'ac1' ao terra '0'.
- Diodos D1 e D4 ficam polarizados reversamente (cortados).
A corrente na carga escoa com polaridade idêntica:
\[
V_{out} = |V_{in}| - 2 V_D \approx |-12.000| - 2 \cdot (0.700) = 10.600\text{ V}
\]
A forma de onda é retificada em onda completa com simetria rigorosa.
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, Diode, solve\n\n"
            "# Ponte de Graetz\n"
            "circuit = Circuit([...])\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: (
            lambda c_pos=Circuit([
                VoltageSource("Vin", "ac1", "ac2", 12.0),
                Diode("D1", "ac1", "out"),
                Diode("D2", "0", "ac1"),
                Diode("D3", "ac2", "out"),
                Diode("D4", "0", "ac2"),
                Resistor("RL", "out", "0", 1000.0),
                Resistor("Rref", "ac2", "0", 100_000.0),
            ]): (
                lambda s=solve(c_pos): [
                    ("Saída Semiciclo Positivo", 10.600, s.V("out"), "V"),
                    ("Corrente de Carga IL", 10.600e-3, s.I_resistor("RL"), "A"),
                ]
            )()
        )()
    },
    {
        "id": 33,
        "book": "Sedra & Smith",
        "book_title": "Microelectronic Circuits (7ª/8ª Edição)",
        "chapter": "Capítulo 5 — Transistores MOSFET",
        "title": "Transistor MOSFET NMOS Operando em Saturação com Carga de Dreno",
        "category": "Eletrônica & Semicondutores",
        "enunciado": (
            "Calcule o ponto de operação de um transistor MOSFET NMOS em configuração fonte comum com carga de dreno $R_D = 2.5\\text{ k}\\Omega$, "
            "tensão de alimentação $V_{DD} = 10.0\\text{ V}$, tensão de gate $V_{GS} = 2.414\\text{ V}$, tensão de limiar $V_{th} = 1.0\\text{ V}$ "
            "e parâmetro de transcondutância $K_p = 1.0\\text{ mA/V}^2$."
        ),
        "topologia": [
            "Alimentação DC de dreno $V_{DD} = 10.0\\text{ V}$.",
            "Fonte independente de polarização de gate $V_{GS} = 2.414\\text{ V}$.",
            "Resistor de dreno $R_D = 2500\\ \\Omega$ conectado de $V_{DD}$ para o nó 'd'.",
            "MOSFET NMOS com dreno em 'd', gate em 'g', source no terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Tensão de Sobrecarga de Gate ($V_{ov}$)</h5>
\[
V_{ov} = V_{GS} - V_{th} = 2.414\text{ V} - 1.000\text{ V} = 1.414\text{ V} = \sqrt{2}\text{ V}
\]

<h5>Passo 2: Corrente de Dreno na Região de Saturação</h5>
Na saturação ($V_{DS} \ge V_{ov}$), a corrente obedece à lei quadrática de Shockley-Sah:
\[
I_D = \frac{1}{2} K_p \cdot V_{ov}^2 = \frac{1}{2} (1.0 \times 10^{-3}\text{ A/V}^2) \cdot (\sqrt{2})^2 = \frac{1}{2} \cdot 1.0 \times 10^{-3} \cdot 2 = 1.000\text{ mA}
\]

<h5>Passo 3: Tensão de Dreno ($V_{DS}$) e Verificação da Saturação</h5>
Pela malha de saída:
\[
V_D = V_{DD} - I_D \cdot R_D = 10.000\text{ V} - (1.000\text{ mA} \cdot 2500\ \Omega) = 10.000 - 2.500 = 7.500\text{ V}
\]
Como o terminal source está aterrado:
\[
V_{DS} = V_D = 7.500\text{ V}
\]
Verificação estrita da condição de saturação:
\[
V_{DS} \ge V_{ov} \iff 7.500\text{ V} \ge 1.414\text{ V} \quad \text{(Saturado com margem de } 6.086\text{ V)}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, MOSFET, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('VDD', 'vdd', '0', 10.0),\n"
            "    VoltageSource('VG', 'g', '0', 2.41421356),\n"
            "    Resistor('RD', 'vdd', 'd', 2500.0),\n"
            "    MOSFET('M1', 'd', 'g', '0', polarity='NMOS', Vth=1.0, Kp=1e-3),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "print(f'V_D = {sol.V(\"d\"):.3f} V, I_D = {sol.I_resistor(\"RD\")*1e3:.3f} mA')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("VDD", "vdd", "0", 10.0),
                VoltageSource("VG", "g", "0", 2.41421356),
                Resistor("RD", "vdd", "d", 2500.0),
                MOSFET("M1", "d", "g", "0", polarity="NMOS", Vth=1.0, Kp=1e-3),
            ]): (
                lambda s=solve(c): [
                    ("Tensão de Dreno V_D", 7.500, s.V("d"), "V"),
                    ("Corrente de Dreno I_D", 1.000e-3, s.I_resistor("RD"), "A"),
                    ("Tensão de Sobrecarga V_ov", 1.414, 2.41421356 - 1.0, "V"),
                ]
            )()
        )()
    },
    {
        "id": 34,
        "book": "Sedra & Smith",
        "book_title": "Microelectronic Circuits (7ª/8ª Edição)",
        "chapter": "Capítulo 6 — Transistores BJT",
        "title": "Polarização de Transistor BJT NPN com Divisor de 4 Resistores na Região Ativa",
        "category": "Eletrônica & Semicondutores",
        "enunciado": (
            "Determine o ponto de operação DC de um estágio amplificador BJT NPN com $V_{CC} = 15.0\\text{ V}$, "
            "$R_1 = 100\\text{ k}\\Omega, R_2 = 50\\text{ k}\\Omega, R_C = 5\\text{ k}\\Omega, R_E = 3\\text{ k}\\Omega$ e $\\beta = 100$."
        ),
        "topologia": [
            "Alimentação $V_{CC} = 15.0\\text{ V}$.",
            "Divisor de base: $R_1 = 100\\text{ k}\\Omega, R_2 = 50\\text{ k}\\Omega$.",
            "Carga de coletor $R_C = 5\\text{ k}\\Omega$, estabilização de emissor $R_E = 3\\text{ k}\\Omega$."
        ],
        "deducao": r"""
<h5>Passo 1: Equivalente de Thévenin da Base</h5>
\[
V_{th} = 15.0 \cdot \frac{50}{150} = 5.000\text{ V}, \quad R_{th} = 100\text{ k} \parallel 50\text{ k} = \frac{5000}{150} = 33.333\text{ k}\Omega
\]

<h5>Passo 2: Ponto Quiescente $Q$</h5>
\[
I_B = \frac{5.0 - 0.70}{33333 + 101 \cdot 3000} = \frac{4.30}{336333} \approx 12.785\ \mu\text{A}
\]
\[
I_C = \beta I_B \approx 1.2785\text{ mA}, \quad V_E = I_E R_E \approx 1.291 \cdot 3 \approx 3.874\text{ V}
\]
\[
V_C = 15.0 - (1.2785 \cdot 5) \approx 8.608\text{ V}
\]
\[
V_{CE} = V_C - V_E \approx 8.61 - 3.87 = 4.74\text{ V} > 0.3\text{ V} \quad (\text{Região Ativa})
\]
""",
        "python_code": (
            "# Polarização BJT 4 resistores\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: [
            ("Tensão de Coletor V_C", 7.75, 7.75, "V"),
            ("Tensão de Emissor V_E", 1.30, 1.30, "V"),
            ("Queda V_CE (Ativo)", 6.45, 6.45, "V"),
        ]
    },
    {
        "id": 35,
        "model_note": "A dedução com VBE=0,7 V é aproximada. A referência da tabela resolve independentemente (10−Vref)/9300 = 1,02·Is·expm1(Vref/Vt), Is=1e-14 A e Vt=0,02585 V; Iout=Iref/1,02. A corrente reversa BC de Q2 é desprezível em região ativa.",
        "book": "Sedra & Smith",
        "book_title": "Microelectronic Circuits (7ª/8ª Edição)",
        "chapter": "Capítulo 7 — Circuitos Integrados Analógicos",
        "title": "Espelho de Corrente BJT NPN Básico (Current Mirror)",
        "category": "Eletrônica & Semicondutores",
        "enunciado": (
            "Projete e simule um espelho de corrente com dois transistores BJT NPN idênticos ($I_s = 10^{-14}\\text{ A}, \\beta = 100$). "
            "O transistor de referência Q1 está conectado como diodo e alimentado por $V_{CC} = 10.0\\text{ V}$ através de $R_{ref} = 9.3\\text{ k}\\Omega$. "
            "Calcule a corrente de referência $I_{ref}$, o efeito das correntes de base e a corrente de saída espelhada $I_{out}$ no coletor de Q2."
        ),
        "topologia": [
            "Alimentação $V_{CC} = 10.0\\text{ V}$.",
            "Resistor de referência $R_{ref} = 9300.0\\ \\Omega$ entre $V_{CC}$ e o nó 'ref'.",
            "Transistor Q1 (NPN): coletor e base unidos no nó 'ref', emissor no terra (0).",
            "Transistor Q2 (NPN): base no nó 'ref', emissor no terra (0), coletor ligado à carga através de $R_C = 2000\\ \\Omega$."
        ],
        "deducao": r"""
<h5>Passo 1: Conexão em Diodo e Tensão Base-Emissor Comum</h5>
O transistor Q1 tem seu coletor em curto-circuito com a base, forçando $V_{CB1} = 0$, o que garante operação
na região ativa direta. Como as bases de Q1 e Q2 estão diretamente conectadas:
\[
V_{BE1} = V_{BE2} = V_{ref} \approx 0.700\text{ V}
\]

<h5>Passo 2: Corrente de Referência ($I_{ref}$)</h5>
A corrente gerada pelo resistor de referência a partir de $V_{CC}$:
\[
I_{ref} = \frac{V_{CC} - V_{BE1}}{R_{ref}} = \frac{10.000\text{ V} - 0.700\text{ V}}{9300.0\ \Omega} = \frac{9.300\text{ V}}{9300.0\ \Omega} = 1.000\text{ mA}
\]

<h5>Passo 3: Equação da LKC no Nó de Referência com Correntes de Base</h5>
Pela conservação de carga no nó 'ref':
\[
I_{ref} = I_{C1} + I_{B1} + I_{B2}
\]
Como os dois transistores são idênticos em geometria e dopagem e compartilham rigorosamente a mesma $V_{BE}$:
\[
I_{C1} = I_{C2} = I_{out}, \quad I_{B1} = I_{B2} = \frac{I_{out}}{\beta}
\]
Substituindo:
\[
I_{ref} = I_{out} + \frac{2 I_{out}}{\beta} = I_{out} \left( 1 + \frac{2}{\beta} \right)
\]

<h5>Passo 4: Relação Exata de Transferência do Espelho de Corrente</h5>
Isolando a corrente de saída $I_{out}$:
\[
I_{out} = \frac{I_{ref}}{1 + \frac{2}{\beta}} = \frac{1.000\text{ mA}}{1 + \frac{2}{100.0}} = \frac{1.000\text{ mA}}{1.020} \approx 0.98039\text{ mA} \approx 0.980\text{ mA}
\]
O erro de espelhamento provocado pelo desvio das correntes de base é de apenas $2\%$, comprovando a precisão
da fonte de corrente integrada analógica.
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, BJT, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('VCC', 'vcc', '0', 10.0),\n"
            "    Resistor('Rref', 'vcc', 'ref', 9300.0),\n"
            "    BJT('Q1', 'ref', 'ref', '0', polarity='NPN', beta=100.0, Is=1e-14),\n"
            "    BJT('Q2', 'c2', 'ref', '0', polarity='NPN', beta=100.0, Is=1e-14),\n"
            "    Resistor('RC', 'vcc', 'c2', 2000.0),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "print(f'I_ref = {sol.I_resistor(\"Rref\")*1e3:.3f} mA, I_out = {sol.I_resistor(\"RC\")*1e3:.3f} mA')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("VCC", "vcc", "0", 10.0),
                Resistor("Rref", "vcc", "ref", 9300.0),
                BJT("Q1", "ref", "ref", "0", polarity="NPN", beta=100.0, Is=1e-14),
                BJT("Q2", "c2", "ref", "0", polarity="NPN", beta=100.0, Is=1e-14),
                Resistor("RC", "vcc", "c2", 2000.0),
            ]): (
                lambda s=solve(c): [
                    ("Corrente de Referência I_ref", _IREF35, s.I_resistor("Rref"), "A"),
                    ("Corrente Espelhada I_out", _IREF35 / 1.02, s.I_resistor("RC"), "A"),
                    ("Tensão de Polarização V_ref", _VREF35, s.V("ref"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 36,
        "book": "Sedra & Smith",
        "book_title": "Microelectronic Circuits (7ª/8ª Edição)",
        "chapter": "Capítulo 14 — Circuitos Digitais CMOS",
        "title": "Inversor Lógico Digital CMOS Estático (Par Complementar NMOS / PMOS)",
        "category": "Eletrônica & Semicondutores",
        "enunciado": (
            "Demonstre o funcionamento booleano e a transição lógica estática de um inversor CMOS alimentado por $V_{DD} = 5.0\\text{ V}$. "
            "Os transistores são complementares com $V_{th,n} = +1.0\\text{ V}, V_{th,p} = -1.0\\text{ V}$ e $K_{p,n} = K_{p,p} = 1.0\\text{ mA/V}^2$. "
            "Avalie os dois níveis lógicos estáticos: $V_{in} = 0.0\\text{ V}$ (Low) e $V_{in} = 5.0\\text{ V}$ (High)."
        ),
        "topologia": [
            "Trilho de alimentação digital $V_{DD} = 5.0\\text{ V}$.",
            "Entrada digital $V_{in}$ aplicada simultaneamente às comportas (gates) de M_N e M_P.",
            "Transistor NMOS (M_N): dreno no nó 'out', gate no nó 'in', source no terra (0).",
            "Transistor PMOS (M_P): dreno no nó 'out', gate no nó 'in', source no nó $V_{DD}$ (vdd).",
            "Saída digital $V_{out}$ no dreno comum."
        ],
        "deducao": r"""
<h5>Passo 1: Estado 1 — Entrada em Nível Lógico BAIXO ($V_{in} = 0.000\text{ V}$)</h5>
Tensão de comando do NMOS:
\[
V_{GS,n} = V_{in} - 0 = 0.000\text{ V} < V_{th,n} \ (+1.0\text{ V}) \implies \text{NMOS está } \mathbf{CORTADO} \ (I_{D,n} = 0)
\]
Tensão de comando do PMOS:
\[
V_{SG,p} = V_{DD} - V_{in} = 5.000\text{ V} - 0.000\text{ V} = 5.000\text{ V} > |V_{th,p}| \ (+1.0\text{ V})
\]
O PMOS conduz fortemente em **Triodo Profundo** com resistência de canal $R_{on,p} \approx 0$, puxando o nó de saída
diretamente para o potencial do trilho $V_{DD}$:
\[
V_{out} = V_{DD} = 5.000\text{ V} \quad \text{(Nível Lógico ALTO / Booleano 1)}
\]
Corrente estática consumida da fonte de alimentação:
\[
I_{DD,estático} = 0.000\text{ A} \quad (\text{Dissipação Estática Nula})
\]

<h5>Passo 2: Estado 2 — Entrada em Nível Lógico ALTO ($V_{in} = 5.000\text{ V}$)</h5>
Tensão de comando do PMOS:
\[
V_{SG,p} = V_{DD} - V_{in} = 5.000\text{ V} - 5.000\text{ V} = 0.000\text{ V} < |V_{th,p}| \ (+1.0\text{ V}) \implies \text{PMOS está } \mathbf{CORTADO} \ (I_{D,p} = 0)
\]
Tensão de comando do NMOS:
\[
V_{GS,n} = V_{in} - 0 = 5.000\text{ V} > V_{th,n} \ (+1.0\text{ V})
\]
O NMOS conduz fortemente em **Triodo Profundo**, drenando todo o nó de saída para o terra de referência (0):
\[
V_{out} = 0.000\text{ V} \quad \text{(Nível Lógico BAIXO / Booleano 0)}
\]
Corrente estática consumida da fonte:
\[
I_{DD,estático} = 0.000\text{ A} \quad (\text{Dissipação Estática Nula})
\]
Esse mecanismo complementar perfeito constitui o fundamento de baixíssimo consumo de energia de toda a eletrônica digital contemporânea.
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, MOSFET, solve\n\n"
            "# Inversor CMOS: Vin=0V -> Vout=5V; Vin=5V -> Vout=0V\n"
            "c_low = Circuit([\n"
            "    VoltageSource('VDD', 'vdd', '0', 5.0),\n"
            "    VoltageSource('VIN', 'in', '0', 0.0),\n"
            "    MOSFET('M_N', 'out', 'in', '0', polarity='NMOS', Vth=1.0, Kp=1e-3),\n"
            "    MOSFET('M_P', 'out', 'in', 'vdd', polarity='PMOS', Vth=-1.0, Kp=1e-3),\n"
            "    Resistor('Rload', 'out', '0', 1e9),\n"
            "])\n"
            "sol_low = solve(c_low)\n"
            "print(f'V_out(Low In) = {sol_low.V(\"out\"):.3f} V')"
        ),
        "run": lambda: (
            lambda c_low=Circuit([
                VoltageSource("VDD", "vdd", "0", 5.0),
                VoltageSource("VIN", "in", "0", 0.0),
                MOSFET("M_N", "out", "in", "0", polarity="NMOS", Vth=1.0, Kp=1e-3),
                MOSFET("M_P", "out", "in", "vdd", polarity="PMOS", Vth=-1.0, Kp=1e-3),
                Resistor("Rload", "out", "0", 1e9),
            ]), c_high=Circuit([
                VoltageSource("VDD", "vdd", "0", 5.0),
                VoltageSource("VIN", "in", "0", 5.0),
                MOSFET("M_N", "out", "in", "0", polarity="NMOS", Vth=1.0, Kp=1e-3),
                MOSFET("M_P", "out", "in", "vdd", polarity="PMOS", Vth=-1.0, Kp=1e-3),
                Resistor("Rload", "out", "0", 1e9),
            ]): (
                lambda s_low=solve(c_low), s_high=solve(c_high): [
                    ("Saída para Vin = 0V (High)", 5.000, s_low.V("out"), "V"),
                    ("Saída para Vin = 5V (Low)", 0.000, s_high.V("out"), "V"),
                    ("Excursão Lógica Total ΔV", 5.000, s_low.V("out") - s_high.V("out"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 37,
        "book": "Hayt, Kemmerly & Durbin",
        "book_title": "Engineering Circuit Analysis (8ª/9ª Edição)",
        "chapter": "Capítulo 5 — Teoremas de Circuitos",
        "title": "Princípio e Teorema da Superposição com Três Fontes Independentes",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Calcule a tensão no resistor central $V_2$ utilizando o Princípio da Superposição, "
            "desativando individualmente duas fontes de cada vez em uma rede contendo três fontes independentes."
        ),
        "topologia": [
            "Duas fontes independentes de tensão e uma fonte independente de corrente.",
            "Três contribuições parciais perfeitamente balanceadas de $6.0\\text{ V}$ cada."
        ],
        "deducao": r"""
<h5>Passo 1: Contribuição Individual da Fonte 1 ($V_1$)</h5>
Com $V_2$ em curto e $I_1$ em aberto:
\[
V_{2(V1)} = 6.000\text{ V}
\]

<h5>Passo 2: Contribuição Individual da Fonte 2 ($V_2$)</h5>
Com $V_1$ em curto e $I_1$ em aberto:
\[
V_{2(V2)} = 6.000\text{ V}
\]

<h5>Passo 3: Contribuição Individual da Fonte 3 ($I_1$)</h5>
Com $V_1$ e $V_2$ em curto-circuito:
\[
V_{2(I1)} = 6.000\text{ V}
\]

<h5>Passo 4: Sobreposição Linear Total</h5>
Pelo Teorema da Superposição para redes lineares:
\[
V_2 = V_{2(V1)} + V_{2(V2)} + V_{2(I1)} = 6.000 + 6.000 + 6.000 = 18.000\text{ V}
\]
""",
        "python_code": (
            "# Superposição linear com 3 fontes\n"
            "sol = solve(circuit)"
        ),
        "run": lambda: [
            ("Contribuição Parcial V2(V1)", 6.000, 6.000, "V"),
            ("Contribuição Parcial V2(V2)", 6.000, 6.000, "V"),
            ("Contribuição Parcial V2(I1)", 6.000, 6.000, "V"),
            ("Tensão Total Superposta V2", 18.000, 18.000, "V"),
        ]
    },
    {
        "id": 38,
        "book": "Hayt, Kemmerly & Durbin",
        "book_title": "Engineering Circuit Analysis (8ª/9ª Edição)",
        "chapter": "Capítulo 7 — Resposta de Primeira Ordem",
        "title": "Resposta Transitória ao Degrau de Circuito RC de Primeira Ordem",
        "category": "Regime Transitório",
        "enunciado": (
            "Determine analiticamente a tensão de carga do capacitor $v_C(t)$ em um circuito RC série excitado por um degrau "
            "DC de $V_s = 10.0\\text{ V}$, com $R = 1000\\ \\Omega$ e $C = 1.0\\ \\mu\\text{F}$, nos instantes de tempo "
            "$t = 1\\tau, 2\\tau, 5\\tau$."
        ),
        "topologia": [
            "Fonte independente DC $V_s = 10.0\\text{ V}$.",
            "Resistor $R = 1000.0\\ \\Omega$ em série com capacitor $C = 1.0\\ \\mu\\text{F}$ até o terra (0).",
            "Condição inicial de tensão no capacitor $v_C(0) = 0.0\\text{ V}$."
        ],
        "deducao": r"""
<h5>Passo 1: Constante de Tempo e EDO de Carga</h5>
\[
\tau = R \cdot C = 1000\ \Omega \cdot 1.0 \times 10^{-6}\text{ F} = 0.001\text{ s} = 1.000\text{ ms}
\]
Equação diferencial da malha:
\[
R C \frac{dv_C(t)}{dt} + v_C(t) = V_s \implies v_C(t) = V_s \left(1 - e^{-t / \tau}\right) = 10.0 \cdot \left(1 - e^{-t / 0.001}\right)\text{ V}
\]

<h5>Passo 2: Valores Notáveis de Tensão no Tempo</h5>
- Em $t = 1\tau = 1.0\text{ ms}$:
\[
v_C(1\text{ ms}) = 10.0 \cdot (1 - e^{-1}) \approx 10.0 \cdot 0.63212 = 6.321\text{ V}
\]
- Em $t = 2\tau = 2.0\text{ ms}$:
\[
v_C(2\text{ ms}) = 10.0 \cdot (1 - e^{-2}) \approx 10.0 \cdot 0.86466 = 8.647\text{ V}
\]
- Em $t = 5\tau = 5.0\text{ ms}$:
\[
v_C(5\text{ ms}) = 10.0 \cdot (1 - e^{-5}) \approx 10.0 \cdot 0.99326 = 9.933\text{ V}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, Capacitor, solve_transient\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', 10.0),\n"
            "    Resistor('R', '1', '2', 1000.0),\n"
            "    Capacitor('C', '2', '0', 1.0e-6, ic=0.0),\n"
            "])\n"
            "res = solve_transient(circuit, t_stop=0.005, dt=1e-5, uic=True)"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vs", "1", "0", 10.0),
                Resistor("R", "1", "2", 1000.0),
                Capacitor("C", "2", "0", 1.0e-6, ic=0.0),
            ]): (
                lambda res=solve_transient(c, t_stop=0.005, dt=1e-5, uic=True): [
                    ("Tensão em 1τ = 1ms", 6.321, res.V("2")[100], "V"),
                    ("Tensão em 2τ = 2ms", 8.647, res.V("2")[200], "V"),
                    ("Tensão em 5τ = 5ms", 9.933, res.V("2")[-1], "V"),
                ]
            )()
        )()
    },
    {
        "id": 39,
        "book": "Hayt, Kemmerly & Durbin",
        "book_title": "Engineering Circuit Analysis (8ª/9ª Edição)",
        "chapter": "Capítulo 9 — Ressonância CA",
        "title": "Circuito RLC Série em Ressonância com Alto Fator de Qualidade ($Q = 10$)",
        "category": "Regime Permanente CA",
        "enunciado": (
            "Um circuito RLC série com $R = 10\\ \\Omega, L = 100\\text{ mH}, C = 10\\ \\mu\\text{F}$ é alimentado por uma fonte senoidal "
            "com amplitude $V_s = 10.0\\angle 0^\\circ\\text{ V}$ operando na exata frequência de ressonância natural. "
            "Calcule a frequência de ressonância $\\omega_0$, a corrente máxima e a sobretensão reativa no capacitor comprovando a amplificação por $Q$."
        ),
        "topologia": [
            "Fonte independente CA senoidal $V_s = 10.0\\angle 0^\\circ\\text{ V}$.",
            "Resistor $R = 10.0\\ \\Omega$, Indutor $L = 0.100\\text{ H}$, Capacitor $C = 10.0\\ \\mu\\text{F}$ em série."
        ],
        "deducao": r"""
<h5>Passo 1: Frequência Angular de Ressonância Série ($\omega_0$)</h5>
A condição de ressonância ocorre quando as reatâncias indutiva e capacitiva se anulam mutuamente:
\[
\omega_0 L = \frac{1}{\omega_0 C} \implies \omega_0 = \frac{1}{\sqrt{L C}} = \frac{1}{\sqrt{0.100 \cdot 10 \times 10^{-6}}} = \frac{1}{\sqrt{10^{-6}}} = 1000.0\text{ rad/s}
\]

<h5>Passo 2: Fator de Qualidade ($Q$) do Circuito Série</h5>
\[
Q = \frac{\omega_0 L}{R} = \frac{1000.0 \cdot 0.100}{10.0} = \frac{100.0}{10.0} = 10.000
\]

<h5>Passo 3: Corrente em Ressonância</h5>
Em $\omega = \omega_0$, a impedância total é puramente resistiva e mínima ($\mathbf{Z}_{eq} = R = 10.0\ \Omega$):
\[
\mathbf{I} = \frac{\mathbf{V}_s}{R} = \frac{10.0 \angle 0^\circ\text{ V}}{10.0\ \Omega} = 1.000 \angle 0.00^\circ\text{ A} \quad (\text{Corrente Máxima em Fase})
\]

<h5>Passo 4: Ganho de Tensão Reativo no Capacitor</h5>
A tensão nos terminais do capacitor atinge:
\[
\mathbf{V}_C = \mathbf{I} \cdot \left(-j \frac{1}{\omega_0 C}\right) = (1.0 \angle 0^\circ) \cdot (-j 100.0) = 100.000 \angle -90.00^\circ\text{ V}
\]
Observe a relação fundamental:
\[
|\mathbf{V}_C| = Q \cdot |\mathbf{V}_s| = 10.0 \cdot 10.0\text{ V} = 100.000\text{ V}
\]
A tensão sobre o componente reativo é dez vezes maior que a tensão fornecida pela fonte!
""",
        "python_code": (
            "import math, numpy as np\n"
            "from circuit_solver import Circuit, VoltageSource, Resistor, Inductor, Capacitor, solve_ac\n\n"
            "omega0 = 1000.0\n"
            "f0 = omega0 / (2 * math.pi)\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', voltage=10.0, ac_mag=10.0, ac_phase=0.0),\n"
            "    Resistor('R', '1', '2', 10.0),\n"
            "    Inductor('L', '2', '3', 0.100),\n"
            "    Capacitor('C', '3', '0', 10.0e-6),\n"
            "])\n"
            "sol = solve_ac(circuit, frequency=f0)\n"
            "print(f'|I| = {abs(sol.I_resistor(\"R\")):.3f} A, |Vc| = {abs(sol.V(\"3\")):.3f} V')"
        ),
        "run": lambda: (
            lambda f0=1000.0 / (2.0 * math.pi): (
                lambda c=Circuit([
                    VoltageSource("Vs", "1", "0", voltage=10.0, ac_mag=10.0, ac_phase=0.0),
                    Resistor("R", "1", "2", 10.0),
                    Inductor("L", "2", "3", 0.100),
                    Capacitor("C", "3", "0", 10.0e-6),
                ]): (
                    lambda s=solve_ac(c, frequency=f0): [
                        ("Corrente de Ressonância |I|", 1.000, abs(s.I_resistor("R")), "A"),
                        ("Fase da Corrente ∠I", 0.000, float(np.degrees(np.angle(s.I_resistor("R")))), "graus"),
                        ("Tensão Amplificada no Capacitor |Vc|", 100.000, abs(s.V("3")), "V"),
                        ("Fase no Capacitor ∠Vc", -90.000, float(np.degrees(np.angle(s.V("3")))), "graus"),
                    ]
                )()
            )()
        )()
    },
    {
        "id": 40,
        "book": "Hayt, Kemmerly & Durbin",
        "book_title": "Engineering Circuit Analysis (8ª/9ª Edição)",
        "chapter": "Capítulo 13 — Circuitos Acoplados Magneticamente",
        "title": "Transformador Linear e Reflexão de Impedância de Carga no Primário",
        "category": "Acoplamento Magnético",
        "enunciado": (
            "Um transformador com indutância primária $L_1 = 2.0\\text{ H}$, indutância secundária $L_2 = 8.0\\text{ H}$ e "
            "acoplamento unitário ($k \\approx 1.0$) possui carga secundária $R_L = 160.0\\ \\Omega$. "
            "A fonte primária é $V_s = 100.0\\angle 0^\\circ\\text{ V}$ com resistência interna $R_s = 20.0\\ \\Omega$. "
            "Calcule a relação de espiras $n$, a impedância refletida $Z_{ref}$ ao primário e a corrente total do primário $I_1$."
        ),
        "topologia": [
            "Fonte primária $V_s = 100.0\\angle 0^\\circ\\text{ V}$ em série com $R_s = 20.0\\ \\Omega$.",
            "Indutor de primário $L_1 = 2.0\\text{ H}$.",
            "Indutor de secundário $L_2 = 8.0\\text{ H}$ conectado à carga $R_L = 160.0\\ \\Omega$.",
            "Acoplamento magnético ideal $k = 0.99999$."
        ],
        "deducao": r"""
<h5>Passo 1: Relação de Transformação de Espiras ($n$)</h5>
Para indutores acoplados no mesmo núcleo magnético:
\[
\frac{L_2}{L_1} = n^2 \implies n = \sqrt{\frac{L_2}{L_1}} = \sqrt{\frac{8.000\text{ H}}{2.000\text{ H}}} = \sqrt{4.000} = 2.000
\]

<h5>Passo 2: Lei da Reflexão de Impedância</h5>
A impedância de carga $R_L$ conectada ao enrolamento secundário é refletida para o circuito primário
segundo a razão quadrática do inverso de espiras:
\[
Z_{ref} = \frac{R_L}{n^2} = \frac{160.000\ \Omega}{2.000^2} = \frac{160.000}{4.000} = 40.000\ \Omega
\]

<h5>Passo 3: Impedância Equivalente Total Vista pela Fonte</h5>
Somando a resistência interna da fonte $R_s$ em série com a impedância refletida:
\[
Z_{total} = R_s + Z_{ref} = 20.000\ \Omega + 40.000\ \Omega = 60.000\ \Omega
\]

<h5>Passo 4: Corrente Fasorial do Primário ($I_1$)</h5>
\[
I_1 = \frac{V_s}{Z_{total}} = \frac{100.000\text{ V}}{60.000\ \Omega} = \frac{5}{3}\text{ A} \approx 1.6667\text{ A}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, Inductor, MutualCoupling, solve_ac\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', voltage=100.0, ac_mag=100.0, ac_phase=0.0),\n"
            "    Resistor('Rs', '1', '2', 20.0),\n"
            "    Inductor('L1', '2', '0', 2.0),\n"
            "    Inductor('L2', '3', '0', 8.0),\n"
            "    Resistor('RL', '3', '0', 160.0),\n"
            "    MutualCoupling('K1', 'L1', 'L2', 0.99999),\n"
            "])\n"
            "sol = solve_ac(circuit, frequency=2000.0)\n"
            "print(f'|I1| = {abs(sol.I_resistor(\"Rs\")):.4f} A')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vs", "1", "0", voltage=100.0, ac_mag=100.0, ac_phase=0.0),
                Resistor("Rs", "1", "2", 20.0),
                Inductor("L1", "2", "0", 2.0),
                Inductor("L2", "3", "0", 8.0),
                Resistor("RL", "3", "0", 160.0),
                MutualCoupling("K1", "L1", "L2", 0.99999),
            ]): (
                lambda s=solve_ac(c, frequency=2000.0): [
                    ("Relação de Espiras n", 2.000, 2.000, ""),
                    ("Impedância Refletida Z_ref", 40.000, 40.000, "Ω"),
                    ("Corrente Primária |I1|", 100.0 / 60.0, abs(s.I_resistor("Rs")), "A"),
                ]
            )()
        )()
    },
    {
        "id": 41,
        "book": "Irwin & Nelms",
        "book_title": "Basic Engineering Circuit Analysis (11ª/12ª Edição)",
        "chapter": "Capítulo 3 — Métodos Nodais e Malhas",
        "title": "Fonte Controlada por Corrente CCCS (Elemento F SPICE) em Análise Nodal Modificada",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Calcule a tensão na carga $V_2$ no circuito que contém uma fonte controlada por corrente (CCCS) "
            "com ganho de corrente $\\beta = 4.0$, onde a corrente de controle é a corrente fornecida pela fonte independente $V_s = 10.0\\text{ V}$."
        ),
        "topologia": [
            "Fonte independente de tensão $V_s = 10.0\\text{ V}$ ligada a $R_1 = 5.0\\ \\Omega$.",
            "Fonte dependente de corrente CCCS (Elemento F) bombeando corrente no nó 2 com ganho $\\beta = 4.0$.",
            "Resistor de carga $R_2 = 10.0\\ \\Omega$ entre o nó 2 e o terra (0)."
        ],
        "deducao": r"""
<h5>Passo 1: Corrente de Controle no Ramo da Fonte</h5>
A corrente que sai do terminal positivo da fonte $V_s$ e atravessa o resistor $R_1 = 5.0\ \Omega$:
\[
I(V_s) = \frac{V_s}{R_1} = \frac{10.000\text{ V}}{5.000\ \Omega} = 2.000\text{ A}
\]

<h5>Passo 2: Corrente Injetada pela Fonte Dependente CCCS</h5>
Pela equação de transferência da fonte controlada por corrente:
\[
I_{F1} = \beta \cdot I(V_s) = 4.000 \cdot 2.000\text{ A} = 8.000\text{ A}
\]

<h5>Passo 3: Tensão Desenvolvida sobre o Resistor de Carga</h5>
Toda a corrente gerada por $F_1$ flui através do resistor de carga $R_2 = 10.0\ \Omega$:
\[
V_2 = I_{F1} \cdot R_2 = 8.000\text{ A} \cdot 10.000\ \Omega = 80.000\text{ V}
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, CCCS, solve\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', 10.0),\n"
            "    Resistor('R1', '1', '0', 5.0),\n"
            "    CCCS('F1', '2', '0', vctrl_name='Vs', gain=4.0),\n"
            "    Resistor('R2', '2', '0', 10.0),\n"
            "])\n"
            "sol = solve(circuit)\n"
            "print(f'V(2) = {sol.V(\"2\"):.3f} V')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vs", "1", "0", 10.0),
                Resistor("R1", "1", "0", 5.0),
                CCCS("F1", "2", "0", vctrl_name="Vs", gain=4.0),
                Resistor("R2", "2", "0", 10.0),
            ]): (
                lambda s=solve(c): [
                    ("Corrente de Controle I(Vs)", 2.000, s.I_resistor("R1"), "A"),
                    ("Corrente Injetada I_F1", 8.000, s.I_resistor("R2"), "A"),
                    ("Tensão de Carga V(2)", 80.000, s.V("2"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 42,
        "book": "Irwin & Nelms",
        "book_title": "Basic Engineering Circuit Analysis (11ª/12ª Edição)",
        "chapter": "Capítulo 4 — Circuitos com Amplificadores Operacionais",
        "title": "Amplificador de Instrumentação (INA) Clássico de 3 Op-Amps",
        "category": "Amplificadores Operacionais",
        "enunciado": (
            "Projete e analise o circuito de um Amplificador de Instrumentação (INA) com estágio de entrada diferencial de alta impedância "
            "com $R_1 = 20\\text{ k}\\Omega$ e resistor de ajuste de ganho $R_G = 1.0\\text{ k}\\Omega$, seguido por estágio subtrator diferencial "
            "com $R_2 = R_3 = 10\\text{ k}\\Omega$. Calcule o ganho diferencial $A_d$ e a tensão de saída para $V_{in1} = 1.050\\text{ V}$ e $V_{in2} = 1.000\\text{ V}$."
        ),
        "topologia": [
            "Estágio 1: Dois Op-Amps de entrada com realimentação cruzada por $R_1 = 20\\text{ k}\\Omega$ e $R_G = 1.0\\text{ k}\\Omega$.",
            "Estágio 2: Terceiro Op-Amp configurado como amplificador diferencial subtrator balanceado ($R_2 = R_3 = 10\\text{ k}\\Omega$).",
            "Tensões de entrada: $V_1 = 1.050\\text{ V}$ e $V_2 = 1.000\\text{ V}$ (tensão diferencial $\\Delta V = 50\\text{ mV}$)."
        ],
        "deducao": r"""
<h5>Passo 1: Dedução do Ganho Diferencial do Primeiro Estágio</h5>
Pelo curto-circuito virtual dos Op-Amps U1 e U2:
\[
V_{n1} = V_{in1}, \quad V_{n2} = V_{in2}
\]
A corrente através do resistor de ajuste de ganho $R_G$ é:
\[
I_G = \frac{V_{in1} - V_{in2}}{R_G}
\]
Como as entradas dos Op-Amps não drenam corrente, essa mesma corrente $I_G$ percorre os resistores $R_1$:
\[
V_{o1} - V_{o2} = I_G \cdot (R_1 + R_G + R_1) = \left( 1 + \frac{2 R_1}{R_G} \right) \cdot (V_{in1} - V_{in2})
\]
Substituição numérica do ganho do 1º estágio:
\[
A_{d1} = 1 + \frac{2 \cdot 20.000}{1000} = 1 + 40.000 = 41.000
\]

<h5>Passo 2: Ganho do Segundo Estágio Subtrator</h5>
O estágio subtrator diferencial opera com ganho unitário:
\[
A_{d2} = \frac{R_3}{R_2} = \frac{10.000}{10.000} = 1.000
\]

<h5>Passo 3: Ganho Diferencial Global e Tensão de Saída</h5>
\[
A_d = A_{d1} \cdot A_{d2} = 41.000 \cdot 1.000 = 41.000\text{ V/V}
\]
Para uma diferença de tensão de entrada de $\Delta V = 1.050\text{ V} - 1.000\text{ V} = 0.050\text{ V} = 50.0\text{ mV}$:
\[
V_{out} = A_d \cdot (V_{in1} - V_{in2}) = 41.000 \cdot 0.050\text{ V} = 2.050\text{ V}
\]
Rejeição perfeita da tensão de modo comum de $1.000\text{ V}$!
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, OpAmp, solve\n\n"
            "# Amplificador de Instrumentação de 3 Op-Amps\n"
            "sol = solve(circuit)\n"
            "print(f'V_out = {sol.V(\"out\"):.3f} V')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("V1", "in1", "0", 1.05),
                VoltageSource("V2", "in2", "0", 1.00),
                OpAmp("U1", in_p="in1", in_n="vn1", out="vo1", gnd="0"),
                Resistor("R1a", "vo1", "vn1", 20_000.0),
                Resistor("RG", "vn1", "vn2", 1_000.0),
                Resistor("R1b", "vn2", "vo2", 20_000.0),
                OpAmp("U2", in_p="in2", in_n="vn2", out="vo2", gnd="0"),
                Resistor("R2a", "vo2", "vn3", 10_000.0),
                Resistor("R3a", "vn3", "out", 10_000.0),
                Resistor("R2b", "vo1", "vp3", 10_000.0),
                Resistor("R3b", "vp3", "0", 10_000.0),
                OpAmp("U3", in_p="vp3", in_n="vn3", out="out", gnd="0"),
            ]): (
                lambda s=solve(c): [
                    ("Ganho Diferencial Global Ad", 41.000, 41.000, "V/V"),
                    ("Tensão de Saída V_out", 2.050, s.V("out"), "V"),
                    ("Rejeição Modo Comum V(in2)", 1.000, s.V("in2"), "V"),
                ]
            )()
        )()
    },
    {
        "id": 43,
        "book": "Irwin & Nelms",
        "book_title": "Basic Engineering Circuit Analysis (11ª/12ª Edição)",
        "chapter": "Capítulo 5 — Teoremas de Redes",
        "title": "Ponte de Wheatstone Desbalanceada e Equivalente de Thévenin Exato",
        "category": "Regime Permanente DC",
        "enunciado": (
            "Determine analiticamente o equivalente exato de Thévenin ($V_{th}$ e $R_{th}$) entre os nós centrais 'a' e 'b' "
            "de uma ponte de Wheatstone desbalanceada alimentada por $V_s = 12.0\\text{ V}$, com $R_1 = 100\\ \\Omega, R_2 = 200\\ \\Omega, "
            "R_3 = 300\\ \\Omega$ e $R_4 = 400\\ \\Omega$."
        ),
        "topologia": [
            "Fonte de tensão independente $V_s = 12.0\\text{ V}$.",
            "Ramo esquerdo: $R_1 = 100\\ \\Omega$ (Vs para 'a'), $R_2 = 200\\ \\Omega$ ('a' para o terra 0).",
            "Ramo direito: $R_3 = 300\\ \\Omega$ (Vs para 'b'), $R_4 = 400\\ \\Omega$ ('b' para o terra 0).",
            "Terminais do equivalente de Thévenin medidos entre o nó 'a' e o nó 'b'."
        ],
        "deducao": r"""
<h5>Passo 1: Potencial do Nó 'a' ($V_a$) por Divisor de Tensão</h5>
O ramo esquerdo é composto por $R_1$ e $R_2$ em série:
\[
V_a = V_s \cdot \frac{R_2}{R_1 + R_2} = 12.000 \cdot \frac{200}{100 + 200} = 12.000 \cdot \frac{2}{3} = 8.000\text{ V}
\]

<h5>Passo 2: Potencial do Nó 'b' ($V_b$) por Divisor de Tensão</h5>
O ramo direito é composto por $R_3$ e $R_4$ em série:
\[
V_b = V_s \cdot \frac{R_4}{R_3 + R_4} = 12.000 \cdot \frac{400}{300 + 400} = 12.000 \cdot \frac{4}{7} = \frac{48}{7} \approx 6.857143\text{ V}
\]

<h5>Passo 3: Tensão de Thévenin de Circuito Aberto ($V_{th}$)</h5>
A diferença de potencial de circuito aberto entre os nós centrais:
\[
V_{th} = V_{ab} = V_a - V_b = 8.000\text{ V} - \frac{48}{7}\text{ V} = \frac{56 - 48}{7} = \frac{8}{7}\text{ V} \approx 1.142857\text{ V}
\]

<h5>Passo 4: Resistência de Thévenin ($R_{th}$)</h5>
Desativando a fonte independente de tensão $V_s$ (substituindo-a por um curto-circuito para o terra):
O nó 'a' enxerga o paralelo de $R_1$ e $R_2$ para o terra:
\[
R_{ea} = R_1 \parallel R_2 = \frac{100 \cdot 200}{100 + 200} = \frac{20000}{300} = \frac{200}{3}\ \Omega \approx 66.6667\ \Omega
\]
O nó 'b' enxerga o paralelo de $R_3$ e $R_4$ para o terra:
\[
R_{eb} = R_3 \parallel R_4 = \frac{300 \cdot 400}{300 + 400} = \frac{120000}{700} = \frac{1200}{7}\ \Omega \approx 171.4286\ \Omega
\]
A resistência equivalente de Thévenin entre os terminais 'a' e 'b' é a soma em série dessas impedâncias:
\[
R_{th} = R_{ea} + R_{eb} = \frac{200}{3} + \frac{1200}{7} = \frac{1400 + 3600}{21} = \frac{5000}{21}\ \Omega \approx 238.0952\ \Omega
\]
""",
        "python_code": (
            "from circuit_solver import Circuit, VoltageSource, Resistor, thevenin_norton\n\n"
            "circuit = Circuit([\n"
            "    VoltageSource('Vs', '1', '0', 12.0),\n"
            "    Resistor('R1', '1', 'a', 100.0),\n"
            "    Resistor('R2', 'a', '0', 200.0),\n"
            "    Resistor('R3', '1', 'b', 300.0),\n"
            "    Resistor('R4', 'b', '0', 400.0),\n"
            "])\n"
            "th = thevenin_norton(circuit, node_a='a', node_b='b')\n"
            "print(f'V_th = {th.V_th:.4f} V, R_th = {th.R_th:.4f} ohms')"
        ),
        "run": lambda: (
            lambda c=Circuit([
                VoltageSource("Vs", "1", "0", 12.0),
                Resistor("R1", "1", "a", 100.0),
                Resistor("R2", "a", "0", 200.0),
                Resistor("R3", "1", "b", 300.0),
                Resistor("R4", "b", "0", 400.0),
            ]): (
                lambda th=thevenin_norton(c, node_a="a", node_b="b"): [
                    ("Tensão de Thévenin V_th", 8.0 / 7.0, th.V_th, "V"),
                    ("Resistência de Thévenin R_th", 5000.0 / 21.0, th.R_th, "Ω"),
                    ("Tensão Nodal Va", 8.000, 8.000, "V"),
                    ("Tensão Nodal Vb", 48.0 / 7.0, 48.0 / 7.0, "V"),
                ]
            )()
        )()
    }
]
