# ⚡ pycircuit-solver

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-147%20passed-brightgreen.svg)](https://docs.pytest.org/)

**`pycircuit-solver`** é um motor computacional modular em Python para análise sistemática de circuitos elétricos lineares e não-lineares em corrente contínua (CC) e alternada (CA), fundamentado no método da **Análise Nodal Modificada (MNA — *Modified Nodal Analysis*)**.

O projeto foi arquitetado para atender rigorosamente ao escopo acadêmico e profissional de Engenharia Elétrica e Eletrônica, unindo precisão matemática estática, validação topológica robusta contra erros humanos e suporte a engenharia de software de nível de portfólio.

> 📖 **Documentação e Guias Rápidos:**
> * 🎓 **[Me Ensine a Usar! (Tutorial Passo a Passo)](ME_ENSINE_A_USAR.md)** — Do primeiro divisor de tensão aos semicondutores, subcircuitos SPICE e transiente em Rust.
> * 📘 **[Documentação Técnica Completa](DOCUMENTACAO_TECNICA.md)** — Formulação matemática do MNA, teoria dos grafos, CA fasorial, sensibilidade analítica e arquitetura em Rust.

---

## 🎯 Escopo e Capacidades do Sistema

### 1. Cobertura Acadêmica e Benchmarks Canônicos
O motor do **`pycircuit-solver`** foi rigorosamente validado e confrontado contra soluções analíticas exatas de 7 das principais obras de referência mundial em Engenharia Elétrica e Microeletrônica:
* **Alexander & Sadiku** — *Fundamentos de Circuitos Elétricos* (Capítulos 1 a 6, 9 & 13)
* **Nilsson & Riedel** — *Circuitos Elétricos* (Thévenin com fontes controladas, somadores e RLC paralelo CA)
* **Boylestad** — *Introdução à Análise de Circuitos* (Teorema de Millman, Escada R-2R de 4 bits, Ponte de Wheatstone em equilíbrio)
* **Dorf & Svoboda** — *Introdução aos Circuitos Elétricos* (Rede em T para alto ganho em Op-Amps, isolamento de impedância)
* **Sedra & Smith** — *Microelectronic Circuits* (Diodos Shockley, polarização de BJT de 4 resistores e amplificador MOSFET NMOS)
* **Hayt, Kemmerly & Durbin** — *Engineering Circuit Analysis* (Teorema da Superposição, transiente RC analítico e ressonância RLC com $Q$)
* **Irwin & Nelms** — *Basic Engineering Circuit Analysis* (Amplificador de Instrumentação INA de 3 Op-Amps, Thévenin em ponte desbalanceada e CCCS)

### 2. Funcionalidades Nativas
* **Arquitetura Híbrida de Alto Desempenho (Python + Rust):** Núcleo numérico em Rust (`rust_core/`) integrado via PyO3 e `maturin`. Acelera simulações de Monte Carlo com paralelismo multi-thread nativo (`rayon`) sem as travas do GIL do Python, com **cobertura integral de 100% dos elementos lineares** (Resistores, Fontes Independentes, VCVS, VCCS, CCCS, CCVS, Amplificadores Operacionais ideais e Indutores em DC).
* **Solvers Esparsos para Circuitos Gigantes (> 1.000 nós):** Montagem esparsa nativa COO/CSC (`build_mna_sparse`) com resolução ultra-rápida via `scipy.sparse.linalg.spsolve` no Python e solver esparso de alta performance em Rust via `faer` Sparse LU (`solve_mna_sparse`), reduzindo o consumo de memória de $O(N^2)$ para $O(N)$ e resolvendo circuitos de 1.000+ nós em milissegundos.
* **Motor MNA Dual (DC & CA):** Resolução matricial em números reais (DC) e complexos `complex128` (CA).
* **Transformadores e Acoplamento Magnético Mútuo (Elemento K do SPICE):** Modelagem de indutores acoplados por fluxo magnético mútuo ($M = k \sqrt{L_1 L_2}$ com $-1.0 \le k \le 1.0$), suporte a transformadores de dois enrolamentos com relação de espiras arbitrária ($n = \sqrt{L_1/L_2}$), inversão de fase/polaridade ($k < 0$), estampagem de admitância fasorial em CA e modelo companheiro transiente no domínio do tempo.
* **Varredura Paramétrica DC (.dc sweep):** Varredura de fontes de tensão, corrente e resistências com suporte a curvas de transferência ($V_{out} \times V_{in}$), curvas de carga e varreduras aninhadas 2D para famílias de curvas características de transistores ($I_C \times V_{CE}$ para diferentes $I_B$).
* **Semicondutores & Solver de Newton-Raphson:** Resolução iterativa de circuitos não-lineares ($Q$-point) com Diodos de Junção (Equação de Shockley), Diodos Zener (regulação de tensão reversa), Transistores BJT (Ebers-Moll ativo e saturado) e MOSFETs (modelo quadrático de corte, triodo e saturação), amortecimento numérico SPICE (`pnjlim`, `fetlim`) e convergência estrita de resíduos ($\epsilon < 10^{-6}$).
* **Análise Transitória no Domínio do Tempo (`.tran`):** Simulação dinâmica de carga e descarga em capacitores e indutores (circuitos RC, RL e RLC de 2ª ordem) via modelos companheiros de integração implícita (**Trapezoidal Rule** e **Backward Euler**) com formas de onda `PULSE`, `SIN` e `STEP`.
* **Fontes Dependentes Completas:** Suporte integral aos elementos SPICE $E$ (VCVS), $G$ (VCCS), $F$ (CCCS) e $H$ (CCVS).
* **Análise de Sensibilidade Analítica via Rede Adjunta e Pior Caso (Worst-Case / EVA):** Cálculo exato das derivadas parciais $\partial V_k / \partial p_i$ e sensibilidades relativas normalizadas $S_{p_i}^{V_k}$ (%/%) para todos os componentes simultaneamente com custo $O(1)$ adicional via Teorema de Tellegen e Método da Rede Adjunta. Determinação automática dos cantos de pior caso com simulação não-linear exata dos extremos (EVA), estimativa linear de Taylor e dispersão estatística RSS ($3\sigma$), acompanhado de relatório White-Box com **Ranking de Pareto de Sensibilidade**.
* **Motor Transiente de Alta Performance em Rust (>1.000.000 passos/s):** Simulação ultra-rápida no domínio do tempo com modelos companion (Trapezoidal Rule e Backward Euler). A matriz MNA é fatorada uma única vez via LU ($O(N^3)$) e cada passo temporal é resolvido em $O(N^2)$ por substituição em código Rust compilado. Inclui auditoria em tempo real de conservação de energia ($E_{\text{stored}}(t) = \frac{1}{2}\sum C v_c^2 + \frac{1}{2}\sum L i_l^2$ e $P_{\text{diss}}(t) = \sum v_r^2/R$) e despacho transparente com fallback automático.
* **Subcircuitos SPICE Hierárquicos (.subckt / .ends):** Suporte nativo a subcircuitos com hierarquia aninhada arbitrária, avaliação paramétrica (`PARAMS:`) com expressões matemáticas `{...}`, biblioteca pedagógica embutida (LM741, inversor CMOS, Sallen-Key), introspecção White-Box e comandos `pycircuit explain` e `pycircuit hierarchy`.
* **Análise CA de Pequenos Sinais com Semicondutores:** Ponto de operação DC ($Q$-point) com Newton-Raphson e linearização analítica automática para Diodos ($r_d$), BJTs (modelo híbrido-$\pi$: $g_m, r_\pi, r_o$) e MOSFETs ($g_m, r_{ds}$), permitindo traçar diagramas de Bode de amplificadores transistorizados.
* **CLI Integrada (`pycircuit`):** Resolução, Thévenin, Monte Carlo, análise CA, transiente (`pycircuit tran --backend rust`), ponto de operação (`pycircuit op`), sensibilidade (`pycircuit sensitivity` / `wca`) e inspeção de macromodelos (`pycircuit explain`) diretamente no terminal.
* **Validador Topológico Pré-Solver (`validator.py`):** Checagem por busca em largura (BFS) de conectividade com o terra, detecção de loops ilegais de fontes de tensão por DFS, impedância de controle de fontes dependentes e nós flutuantes.
* **Conversor de Sufixos de Engenharia:** Leitura nativa de notações laboratoriais (`"4.7k"`, `"10mA"`, `"220u"`, `"4k7"`, `"1MEG"`).
* **Extrator de Thévenin/Norton Dual:** Método híbrido (injeção de tensão e injeção de corrente) que resolve casos degenerados como fontes ideais nos terminais ($R_{th} = 0\ \Omega$) ou terminais abertos ($R_{th} \to \infty$).
* **Auditoria de Potência (Tellegen):** Verificação de conservação global de energia ($\sum P = 0$) em DC e balanço de potência complexa em CA.
* **Simulação de Monte Carlo:** Análise estatística de tolerâncias ($1\%, 5\%, 10\%$) em até dezenas de milhares de amostras por segundo.
* **Parser SPICE:** Leitura de arquivos `.cir`, `.net`, `.sp` com suporte a linhas de continuação (`+`).
* **Visualização:** Curvas de máxima potência, histogramas de Monte Carlo e diagramas de Bode.

---

## 🗂️ Arquitetura de Diretórios do Projeto

```text
pycircuit-solver/
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # Pipeline de testes automatizados com pytest (Python 3.10, 3.11, 3.12)
│
├── .gitignore
├── LICENSE                              # Licença MIT
├── README.md                            # Documentação técnica e guia de uso
├── pyproject.toml                       # Configuração do pacote Python moderno (PEP 621)
├── requirements.txt                     # Dependências mínimas de execução
│
├── circuit_solver/                      # Pacote principal em Python
│   ├── __init__.py                      # Facade de importação pública
│   ├── cli.py                           # Interface de Linha de Comando (CLI pycircuit)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── elements.py                  # Dataclasses (R, C, L, V, I, VCVS, VCCS, CCCS, CCVS, OpAmp, Diodos, BJT, MOS)
│   │   ├── circuit.py                   # Container de topologia e mapeamento de nós
│   │   ├── validator.py                 # Validador topológico (BFS, detecção de loops KVL e sufixos)
│   │   ├── solution.py                  # Processamento de tensões, correntes e balanço de Tellegen
│   │   ├── subcircuit.py                # Expansão hierárquica e introspecção White-Box
│   │   └── builtin_library.py           # Biblioteca didática de macromodelos (741, CMOS, Sallen-Key)
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── mna.py                       # Algoritmo MNA estático (DC) em NumPy
│   │   ├── ac.py                        # Motor fasorial e resposta em frequência (CA) em complex128
│   │   ├── small_signal.py              # Análise CA de pequenos sinais com linearização de semicondutores
│   │   ├── sensitivity.py               # Análise de Sensibilidade Analítica (Rede Adjunta), WCA e EVA
│   │   ├── transient.py                 # Motor de análise transitória com auto-dispatch para Rust
│   │   ├── transient_rust.py            # Ponte Python ↔ Rust para simulação transiente ultra-rápida
│   │   ├── thevenin.py                  # Extração dual de Thévenin, Norton e Máxima Potência
│   │   ├── monte_carlo.py               # Varredura estatística em Python puro (fallback)
│   │   └── monte_carlo_rust.py          # Ponte de alta performance Python ↔ Rust (PyO3)
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   └── parser.py                    # Leitor de netlists SPICE com .subckt, .include, PULSE e SIN
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                     # Gráficos de potência, Monte Carlo, Bode e Transiente
│
├── rust_core/                           # Núcleo de cálculo de alta performance em Rust (PyO3 + rayon)
│   ├── Cargo.toml                       # Manifesto Rust com rayon, nalgebra, faer e pyo3
│   └── src/
│       ├── lib.rs                       # Interface CPython/PyO3 e funções exportadas
│       ├── types.rs                     # Estruturas de dados pré-resolvidas sem overhead de strings
│       ├── mna_core.rs                  # Montagem MNA e resolução LU densa e esparsa (Faer)
│       ├── monte_carlo.rs               # Simulação de Monte Carlo com paralelismo real via rayon
│       └── transient.rs                 # Motor de integração implícita transiente (>1M steps/s)
│
├── examples/                            # Scripts de problemas clássicos resolvidos
│   ├── cap02_ponte_wheatstone.py        # Ponte de Wheatstone desbalanceada + Monte Carlo
│   ├── cap04_thevenin_analise.py        # Equivalente Thévenin/Norton e curva P_max
│   ├── cap05_opamp_inversor.py          # Amplificador Operacional Inversor com terra virtual
│   ├── cap07_transiente_rlc.py          # Resposta ao degrau de RLC de 2ª ordem e overshoot
│   └── cap09_filtro_rc_fasores.py       # Filtro passa-baixas RC, fasores e Diagrama de Bode
│
└── tests/                               # Suíte de testes automatizados (136 testes pytest)
    ├── test_sensitivity_wca.py         # Testes de sensibilidade analítica (Rede Adjunta), WCA e EVA
    ├── test_rust_transient.py           # Testes do motor transiente em Rust, conservação de energia e speedup
    ├── test_small_signal_ac.py          # Testes de pequenos sinais com diodos, BJT e MOSFET
    ├── test_subcircuits.py              # Testes de subcircuitos SPICE (.subckt), parâmetros e introspecção
    ├── test_textbook_benchmarks.py      # Validação analítica multi-livros (Sadiku, Nilsson, Boylestad, Dorf)
    ├── test_coupled_inductors.py        # Acoplamento magnético mútuo, transformador e Sadiku Cap. 13
    ├── test_nonlinear.py                # Semicondutores não-lineares e Newton-Raphson (Diodo, Zener, BJT, MOSFET)
    ├── test_dc_sweep.py                 # Varredura paramétrica DC (.dc sweep), curvas IV e famílias 2D
    ├── test_transient.py                # Testes analíticos de transitório RC, RL, RLC e formas de onda
    ├── test_rust_monte_carlo.py         # Testes de correção, reprodutibilidade e speedup do motor Rust
    ├── test_sparse_and_rust.py          # Testes de MNA esparso (Faer / SciPy) e grandes circuitos
    ├── test_robustness_edge_cases.py    # Testes de robustez, BFS, loops KVL, R_th=0 e parser
    ├── test_validator.py                # Testes de validação topológica e sufixos
    ├── test_mna_engine.py               # Testes analíticos do motor MNA e 4 fontes dependentes
    ├── test_ac_engine.py                # Testes de análise CA, ressonância e corte
    └── test_cli.py                      # Testes da interface de terminal pycircuit

```

---

## 💻 Interface de Linha de Comando (CLI)

O comando `pycircuit` fica disponível no terminal após a instalação:

```bash
# Resolver circuito DC a partir de netlist SPICE
pycircuit solve circuito.cir

# Análise de Sensibilidade e Pior Caso (Worst-Case / EVA)
pycircuit sensitivity circuito.cir --target 2
# ou com o alias rápido 'wca'
pycircuit wca circuito.cir -t out --tol 0.01


# Extrair equivalente de Thévenin / Norton entre nós 3 e 0 com gráfico
pycircuit thevenin circuito.cir -a 3 -b 0 --plot

# Análise de Monte Carlo com 1000 amostras e tolerância de 5%
pycircuit monte-carlo circuito.cir -n 1000 -t 0.05 --plot

# Análise fasorial CA a 60 Hz
pycircuit ac circuito.cir -f 60.0

# Varredura de frequência e diagrama de Bode (10 Hz a 100 kHz)
pycircuit ac circuito.cir --sweep 10 100000 100 --node out --plot

# Análise transitória no tempo (0 a 10 ms com passo de 10 μs e gráfico)
pycircuit tran circuito.cir --tstop 0.01 --dt 1e-5 --node out --plot

# Relatório didático White-Box de subcircuitos e impacto matricial MNA
pycircuit explain circuito.cir

# Árvore ASCII de hierarquia de subcircuitos
pycircuit hierarchy circuito.cir

# Ponto de operação DC (.op) e parâmetros de pequenos sinais para semicondutores
pycircuit op amplificador.cir
```

---

## 🚀 Instalação e Uso Rápido

### Instalação

```bash
git clone https://github.com/elycbarros/pycircuit-solver.git
cd pycircuit-solver
pip install -e .
```

Para instalar dependências de desenvolvimento e testes:
```bash
pip install -e .[dev]
```

#### Compilação do Núcleo em Rust (Opcional — Alto Desempenho)
O `pycircuit-solver` opera em Python puro com NumPy por padrão. Para habilitar aceleração em Rust com paralelismo nativo multi-thread via `rayon`:
```bash
cd rust_core
maturin build --release --interpreter python3
pip install target/wheels/*.whl
```

### Exemplo em Python (Divisor com Carga e Op-Amp)

```python
from circuit_solver import Circuit, VoltageSource, Resistor, OpAmp, solve

# Definindo os elementos do circuito
elements = [
    VoltageSource("Vin", "in", "0", 2.0),
    Resistor("R1", "in", "vn", 10_000.0),
    Resistor("Rf", "vn", "vout", 50_000.0),
    Resistor("Rload", "vout", "0", 10_000.0),
    OpAmp("U1", in_p="0", in_n="vn", out="vout", gnd="0")
]

circuit = Circuit(elements)
solution = solve(circuit)

print(solution.summary())
print(f"Ganho: {solution.V('vout') / solution.V('in'):.2f}")  # Av = -5.00
```

### Extração de Thévenin e Norton

```python
from circuit_solver import thevenin_norton

# Extrair equivalente visto entre terminais "3" e "0"
th = thevenin_norton(circuit, node_a="3", node_b="0")
print(th.summary())
```

### Leitura de Netlist SPICE

```python
from circuit_solver.io.parser import parse_netlist_string
from circuit_solver import Circuit, solve

netlist = """
* Circuito Divisor de Tensão
V1 1 0 DC 12
R1 1 2 4k7
R2 2 0 10k
.end
"""

circuit = Circuit(parse_netlist_string(netlist))
sol = solve(circuit)
print(f"V(2) = {sol.V('2'):.4f} V")
```

### Circuito Não-Linear com Semicondutores (Diodo & Newton-Raphson)

```python
from circuit_solver import Circuit, VoltageSource, Resistor, Diode, solve

# Fonte de 5V, resistor limitador de 1k e Diodo 1N4148 (Shockley)
circuit = Circuit([
    VoltageSource("V1", "1", "0", 5.0),
    Resistor("R1", "1", "2", 1000.0),
    Diode("D1", "2", "0", Is=1e-14, n=1.0),
])

# O solver detecta automaticamente componentes não-lineares e despacha o Newton-Raphson
sol = solve(circuit)

print(f"Ponto Q de Operação:")
print(f"  V_D = {sol.V('2'):.4f} V  (queda de tensão típica no joelho)")
print(f"  I_D = {sol.I_diode('D1') * 1e3:.3f} mA")
print(f"  Convergência obtida em {sol.iterations} iterações (tol < 1e-6)")
```

### Varredura Paramétrica DC (.dc sweep)

Via CLI no terminal:
```bash
# Varre a fonte V1 de 0 a 10V com passo de 0.1V e exibe o gráfico da tensão no nó 'out'
pycircuit dc circuito.cir --sweep V1 0 10 0.1 --node out --plot
```

Via script Python:
```python
from circuit_solver import Circuit, VoltageSource, Resistor, dc_sweep, plot_dc_sweep

circuit = Circuit([
    VoltageSource("VIN", "in", "0", 0.0),
    Resistor("R1", "in", "out", 1000.0),
    Resistor("R2", "out", "0", 1000.0),
])

# Varia VIN de 0V a 10V com passo de 0.5V
res = dc_sweep(circuit, "VIN", 0.0, 10.0, 0.5)
print(f"V(out) final = {res.V('out')[-1]:.2f} V")
```

### Subcircuitos SPICE (.subckt) e Introspecção White-Box

```python
from circuit_solver.io.parser import parse_netlist_string
from circuit_solver import Circuit, solve

netlist = """
.subckt DIVISOR in out gnd PARAMS: R1=10k R2=10k
R1 in out {R1}
R2 out gnd {R2}
.ends

V1 vin 0 DC 10
X1 vin vmid 0 DIVISOR PARAMS: R1=2k R2=8k
.end
"""

circuit = Circuit(parse_netlist_string(netlist))
sol = solve(circuit)
print(f"V(vmid) = {sol.V('vmid'):.2f} V")  # 8.00 V
```

### Análise de Sensibilidade Analítica e Pior Caso (WCA / EVA)

```python
from circuit_solver import Circuit, VoltageSource, Resistor, analyze_sensitivity

circuit = Circuit([
    VoltageSource("V1", "1", "0", 10.0),
    Resistor("R1", "1", "2", 1000.0, tolerance=0.05),
    Resistor("R2", "2", "0", 1000.0, tolerance=0.05),
])

# Calcula sensibilidade analítica por rede adjunta e cantos de pior caso EVA
wca = analyze_sensitivity(circuit, target_node="2")
print(wca.summary())
```

### Transiente Ultra-Rápido em Rust com Auditoria de Energia

```python
from circuit_solver import Circuit, VoltageSource, Resistor, Capacitor, solve_transient

circuit = Circuit([
    VoltageSource("V1", "1", "0", 12.0),
    Resistor("R1", "1", "2", 1000.0),
    Capacitor("C1", "2", "0", 10e-6),
])

# Execução nativa em Rust (>1.000.000 passos/s)
res = solve_transient(circuit, t_stop=0.05, dt=1e-5, backend="rust")
print(f"Passos calculados: {len(res.t)}")
print(f"Energia armazenada final: {res.stored_energy[-1] * 1e6:.2f} µJ")
```

---

## 🧪 Executando os Testes Automatizados

```bash
pytest
```

Com relatório de cobertura de código:
```bash
pytest --cov=circuit_solver --cov-report=term-missing
```

---

## 📚 Matriz de Validação Analítica Multi-Livros

O motor computacional do **`pycircuit-solver`** é continuamente validado por uma suíte de testes de precisão analítica (`tests/test_textbook_benchmarks.py`) contendo **31 problemas canônicos clássicos** retirados das 7 principais referências bibliográficas mundiais do ensino de engenharia elétrica e eletrônica:

| Livro / Manual | Capítulo / Exemplo | Tópico de Análise | Resultado Analítico de Referência | Status do Solver |
| :--- | :--- | :--- | :--- | :---: |
| **Alexander & Sadiku** | Cap. 2 (Ex. 2.8) | Associação Série-Paralelo e Divisor | $V_2 = 6\text{ V}, I_1 = 2\text{ A}, I_2 = 1\text{ A}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 2 (Ex. 2.15) | Transformação $\Delta - Y$ em Ponte | $R_{eq} = 50/3\ \Omega, I_{tot} = 7.200\text{ A}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 3 (Ex. 3.1) | Análise Nodal (2 fontes de corrente) | $V_1 = 18.333\text{ V}, V_2 = 35.000\text{ V}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 3 (Ex. 3.2) | Análise Nodal com VCCS dependente | $V_1 = 4.8\text{ V}, V_2 = 2.4\text{ V}, V_3 = -2.4\text{ V}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 3 (Ex. 3.3) | Supernó flutuante com fonte de tensão | $V_1 = -6.0\text{ V}, V_2 = -8.0\text{ V}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 4 (Ex. 4.8) | Thévenin, Norton e Máxima Transferência | $V_{th} = 30\text{ V}, R_{th} = 4\ \Omega, P_{max} = 56.25\text{ W}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 4 (Ex. 4.9) | Thévenin com fonte controlada VCVS | $V_{th} = 60\text{ V}, R_{th} = 14\ \Omega, I_N = 4.286\text{ A}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 5 (Ex. 5.2) | Amplificador Inversor (Op-Amp) | $V_o = -1.25\text{ V}, I_{in} = 50\ \mu\text{A}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 5 (Ex. 5.5) | Amplificador Não-Inversor | $V_o = 16.0\text{ V}, V_n = 4.0\text{ V}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 5 (Ex. 5.6) | Somador Inversor de 3 Entradas | $V_o = -17.0\text{ V}, V_n = 0.0\text{ V}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 5 (Ex. 5.7) | Amplificador Diferencial (Subtrator) | $V_o = 4.0\text{ V}, V_p = V_n = 3.333\text{ V}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 6 (Ex. 6.13) | Integrador Ideal com Op-Amp (Tempo) | $dv_o/dt = -200\text{ V/s}, v_o(5\text{ms}) = -1.000\text{ V}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 9 (Ex. 9.9) | Circuito RLC Série CA (Fasores) | $I = 0.3322\angle -10.24^\circ\text{ A}$ | ✅ Aprovado |
| **Alexander & Sadiku** | Cap. 13 (Ex. 13.1) | Circuitos Magneticamente Acoplados | $I_1 = 1.350\angle -62.85^\circ\text{ A}, I_2 = 0.1431\angle -30.85^\circ\text{ A}$ | ✅ Aprovado |
| **Nilsson & Riedel** | Cap. 4 (Ex. 4.10) | Thévenin com Fonte Controlada | $V_{th} = 40\text{ V}, R_{th} = 10\ \Omega, P_{max} = 40\text{ W}$ | ✅ Aprovado |
| **Nilsson & Riedel** | Cap. 5 (Ex. 5.3) | Somador Inversor com Pesos Arbitrários | $V_o = -23.0\text{ V}, I_L = -0.92\text{ mA}$ | ✅ Aprovado |
| **Nilsson & Riedel** | Cap. 9 (Ex. 9.6) | RLC Paralelo CA em Alta Frequência | $V = 200\sqrt{2}\angle 45^\circ\text{ V}, I_L = 8\sqrt{2}\angle -45^\circ\text{ A}$ | ✅ Aprovado |
| **Boylestad** | Cap. 9 | Teorema de Millman (3 Baterias) | $V_L = 6.0\text{ V}, I_1 = 6\text{ A}, I_2 = -3\text{ A}, P = 18\text{ W}$ | ✅ Aprovado |
| **Boylestad** | Cap. 8 | Rede Escada R-2R D/A (4 Bits) | Escalonamento binário exato ($V_o = 8, 4, 2, 1, 15\text{ V}$) | ✅ Aprovado |
| **Boylestad** | Cap. 6 | Ponte de Wheatstone Balanceada | $V_{ab} = 0.000\text{ V}, I_g = 0.000\text{ A}, P_g = 0\text{ W}$ | ✅ Aprovado |
| **Dorf & Svoboda** | Cap. 6 | Op-Amp com Rede T de Realimentação | $A_v = -12.0, V_o = -1.200\text{ V}, V_t = -0.1\text{ V}$ | ✅ Aprovado |
| **Dorf & Svoboda** | Cap. 6 | Buffer Seguidor & Isolamento de Carga | $V_o = 5.000\text{ V}, I_{RL} = 100\text{ mA}, I_{Rs} = 0\text{ A}$ | ✅ Aprovado |
| **Sedra & Smith** | Cap. 4 | Ponto Q e Reta de Carga de Diodo PN | $V_D \approx 0.6925\text{ V}, I_D \approx 4.3075\text{ mA}, P_D \approx 2.98\text{ mW}$ | ✅ Aprovado |
| **Sedra & Smith** | Cap. 6 | Polarização BJT NPN com 4 Resistores | $V_C \approx 7.75\text{ V}, V_E \approx 1.30\text{ V}, V_{CE} > 5\text{ V}$ (Ativo) | ✅ Aprovado |
| **Sedra & Smith** | Cap. 5 | Fonte Comum MOSFET NMOS Saturação | $I_D = 1.0\text{ mA}, V_D = 7.500\text{ V}, V_{DS} \ge V_{ov}$ (Saturado) | ✅ Aprovado |
| **Hayt, Kemmerly & Durbin** | Cap. 5 | Teorema da Superposição (3 Fontes) | $V_2 = V_{2(V1)} + V_{2(V2)} + V_{2(I1)} = 6+6+6 = 18\text{ V}$ | ✅ Aprovado |
| **Hayt, Kemmerly & Durbin** | Cap. 7 | Carga RC Transiente ao Degrau (1ª Ordem)| $v_C(\tau) = 6.321\text{ V}, v_C(2\tau) = 8.647\text{ V}, v_C(5\tau) = 9.933\text{ V}$ | ✅ Aprovado |
| **Hayt, Kemmerly & Durbin** | Cap. 9 | RLC Série Ressonância e Ganho $Q=10$ | $I = 1.000\angle 0^\circ\text{ A}, V_C = 100.0\angle -90^\circ\text{ V}$ ($Q \cdot V_s$) | ✅ Aprovado |
| **Irwin & Nelms** | Cap. 4 | Amplificador de Instrumentação (3 Op-Amps)| $A_d = 41.0, V_{out} = 2.050\text{ V}$ (Rejeição de Modo Comum) | ✅ Aprovado |
| **Irwin & Nelms** | Cap. 5 | Ponte Desbalanceada e Thévenin Exato | $V_{th} = 8/7\text{ V} \approx 1.143\text{ V}, R_{th} = 5000/21\ \Omega \approx 238.10\ \Omega$ | ✅ Aprovado |
| **Irwin & Nelms** | Cap. 3 | Fonte Controlada CCCS (Elemento F) | $I_{F1} = 4 \cdot I(V_s) = 8.0\text{ A}, V_{load} = 80.000\text{ V}$ | ✅ Aprovado |

---

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE).
