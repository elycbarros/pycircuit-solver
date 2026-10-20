# ⚡ pycircuit-solver

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)

**`pycircuit-solver`** é um motor computacional modular em Python para análise sistemática de circuitos elétricos lineares em corrente contínua (CC), fundamentado no método da **Análise Nodal Modificada (MNA — *Modified Nodal Analysis*)**.

O projeto foi arquitetado para atender rigorosamente ao escopo acadêmico dos **Capítulos 1 a 5** de *Fundamentos de Circuitos Elétricos* (Alexander & Sadiku), unindo precisão matemática estática, validação topológica robusta contra erros humanos e suporte a engenharia de software de nível de portfólio.

---

## 🎯 Escopo e Capacidades do Sistema

### 1. Cobertura Acadêmica (Referência: Alexander & Sadiku)
* **Capítulos 1 & 2:** Leis básicas, Lei de Ohm, LKC, LKT, associações série/paralelo, transformações $\Delta - Y$ e balanço de potência.
* **Capítulo 3:** Análise Nodal e de Malhas (com fontes independentes e dependentes).
* **Capítulo 4:** Teoremas de Circuitos (Superposição, Thévenin, Norton e Máxima Transferência de Potência).
* **Capítulo 5:** Amplificadores Operacionais Ideais (circuitos inversores, não-inversores, somadores e diferenciais).

### 2. Funcionalidades Nativas
* **Motor MNA em Blocos:** Resolução de sistemas lineares $A\mathbf{x} = \mathbf{z}$ utilizando a formulação particionada da Análise Nodal Modificada.
* **Validador Topológico Pré-Solver (`validator.py`):** Checagem automática de nós flutuantes (*dangling nodes*), verificação obrigatória do nó de referência Terra (`"0"`), detecção de fontes em conflito e restrições físicas de valores ($R > 0$).
* **Conversor de Sufixos de Engenharia (`validator.py`):** Leitura nativa de notações corriqueiras de laboratório (ex: `"4.7k"`, `"10mA"`, `"220u"`, `"4k7"`, `"1MEG"`).
* **Extrator de Thévenin/Norton (`thevenin.py`):** Cálculo automatizado de $V_{\text{th}}$, $I_{\text{N}}$, $R_{\text{th}}$ e varredura de máxima potência entre quaisquer dois terminais $(A, B)$.
* **Auditoria de Potência (Teorema de Tellegen):** Verificação de conservação global de energia ($\sum P = 0$) em pós-processamento com precisão de máquina.
* **Simulação de Monte Carlo (`monte_carlo.py`):** Análise estatística de pior caso (*worst-case*) considerando tolerâncias comerciais de resistores ($1\%, 5\%, 10\%$).
* **Parser SPICE (`parser.py`):** Leitura direta de arquivos netlist SPICE padrão (`.cir`, `.net`, `.sp`).
* **Visualização Elegante (`plots.py`):** Gráficos com tema escuro moderno para curva de potência e histogramas de Monte Carlo via Matplotlib.

---

## 📐 Fundamentação Matemática (MNA)

A Análise Nodal Modificada expande a análise nodal clássica acoplando tensões nodais e correntes auxiliares de restrição:

$$\begin{bmatrix} \mathbf{G} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix} \begin{bmatrix} \mathbf{v} \\ \mathbf{j} \end{bmatrix} = \begin{bmatrix} \mathbf{i} \\ \mathbf{e} \end{bmatrix}$$

* **$\mathbf{G}$:** Matriz de condutâncias nodais e transcondutâncias de fontes controladas ($VCCS$).
* **$\mathbf{B}$ e $\mathbf{C}$:** Matrizes de incidência e restrição de tensão (fontes independentes, VCVS e saídas de Op-Amps).
* **$\mathbf{D}$:** Matriz de acoplamento entre variáveis dependentes (nula para fontes ideais independentes).
* **$\mathbf{v}$ e $\mathbf{j}$:** Vetor de potenciais nodais desconhecidos e vetor de correntes auxiliares nas fontes e Op-Amps.
* **$\mathbf{i}$ e $\mathbf{e}$:** Injeções de corrente nodais e valores nominais de restrição de tensão.

---

## 🗂️ Arquitetura de Diretórios do Projeto

```text
pycircuit-solver/
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # Pipeline de testes automatizados com pytest
│
├── .gitignore
├── LICENSE                              # Licença MIT
├── README.md                            # Documentação técnica e guia de uso
├── pyproject.toml                       # Configuração do pacote Python moderno (PEP 621)
├── requirements.txt                     # Dependências mínimas de execução
│
├── circuit_solver/                      # Pacote principal
│   ├── __init__.py                      # Facade de importação pública
│   ├── core/
│   │   ├── __init__.py
│   │   ├── elements.py                  # Dataclasses (Resistor, VoltageSource, CurrentSource, VCVS, VCCS, OpAmp)
│   │   ├── circuit.py                   # Container de topologia e mapeamento de nós
│   │   ├── validator.py                 # Validador topológico e conversor de sufixos de engenharia
│   │   └── solution.py                  # Processamento de tensões, correntes e balanço de Tellegen
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── mna.py                       # Algoritmo de estampagem (stamping) e resolução linear
│   │   ├── thevenin.py                  # Extração de Thévenin, Norton e Máxima Potência
│   │   └── monte_carlo.py               # Varredura estatística de tolerâncias
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   └── parser.py                    # Leitor de arquivos Netlist padrão SPICE (.cir)
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                     # Gráficos de curvas de potência e distribuição
│
├── examples/                            # Scripts de problemas clássicos resolvidos
│   ├── cap02_ponte_wheatstone.py        # Ponte de Wheatstone desbalanceada + Monte Carlo
│   ├── cap04_thevenin_analise.py        # Equivalente Thévenin/Norton e curva P_max
│   └── cap05_opamp_inversor.py          # Amplificador Operacional Inversor com terra virtual
│
└── tests/                               # Suíte de validação analítica (pytest)
    ├── test_validator.py                # Testes de validação topológica e sufixos
    ├── test_mna_engine.py               # Testes analíticos do motor MNA e SPICE parser
    └── test_thevenin.py                 # Testes de Thévenin/Norton e máxima transferência
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

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE).
