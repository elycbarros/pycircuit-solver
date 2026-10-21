# 📘 Documentação Técnica Completa: `pycircuit-solver`
*Arquitetura de Software, Modelagem Matemática e Engenharia de Circuitos Elétricos*

---

## 📑 Sumário Executivo

O **`pycircuit-solver`** é uma plataforma computacional híbrida (Python + Rust) voltada para a análise, simulação e validação estática de circuitos elétricos lineares. O projeto foi projetado com duplo objetivo:
1. **Rigor Eletrotécnico:** Implementar os fundamentos clássicos da Teoria de Circuitos baseando-se no método da **Análise Nodal Modificada (MNA — *Modified Nodal Analysis*)**, com suporte integral a elementos passivos, ativos, fontes dependentes, regime permanente senoidal (CA fasorial), Teorema de Tellegen e teoremas de Thévenin/Norton.
2. **Engenharia de Software de Alto Nível:** Apresentar uma arquitetura desacoplada, tipagem estática rigorosa, validação topológica pré-solver baseada em Teoria dos Grafos, suíte automatizada de 68 testes e um **núcleo de alta performance em Rust (`rust_core`)** que acelera simulações estatísticas (Monte Carlo) via multi-threading nativo sem bloqueio de GIL.

---

## 1. 📐 Fundamentação Teórica e Formulação Matemática

### 1.1 O Método da Análise Nodal Modificada (MNA)
A Análise Nodal convencional falha diretamente na presença de **fontes ideais de tensão**, pois a corrente que as atravessa é determinada pelo circuito externo e não por uma relação de condutância simples ($i = G \cdot v$). O método MNA (desenvolvido originalmente por Ho, Ruehli e Brennan em 1975) resolve essa limitação introduzindo a corrente das fontes de tensão como **variáveis auxiliares de estado**.

O sistema global de equações é expresso na forma matricial particionada em 4 blocos:

$$\begin{bmatrix} \mathbf{G} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix} \begin{bmatrix} \mathbf{v} \\ \mathbf{j} \end{bmatrix} = \begin{bmatrix} \mathbf{i} \\ \mathbf{e} \end{bmatrix}$$

Onde:
* $\mathbf{G} \in \mathbb{R}^{N \times N}$ (ou $\mathbb{C}^{N \times N}$ em CA): Submatriz de condutâncias nodais entre os $N$ nós não-terra do circuito.
* $\mathbf{B} \in \mathbb{R}^{N \times M}$: Matriz de incidência nó-ramo para os $M$ elementos que impõem restrição de tensão (fontes de tensão, VCVS, saídas de Op-Amps).
* $\mathbf{C} \in \mathbb{R}^{M \times N}$: Matriz de restrições de tensão impostas pelas fontes ou curto-circuito virtual.
* $\mathbf{D} \in \mathbb{R}^{M \times M}$: Matriz de acoplamento entre variáveis dependentes (nula para fontes ideais independentes; não-nula para CCVS ou modelos reais com resistência interna).
* $\mathbf{v} \in \mathbb{R}^{N}$: Vetor das tensões nodais incógnitas (medidas em relação ao terra $\text{node } 0 = 0\text{ V}$).
* $\mathbf{j} \in \mathbb{R}^{M}$: Vetor das correntes auxiliares incógnitas através dos ramos que impõem tensão.
* $\mathbf{i} \in \mathbb{R}^{N}$: Vetor de injeção nodal de corrente proveniente de fontes de corrente independentes.
* $\mathbf{e} \in \mathbb{R}^{M}$: Vetor de valores de restrição de tensão das fontes independentes.

---

### 1.2 Regras de Estampa Nodal (*Element Stamping*)

Cada elemento adicionado ao circuito injeta coeficientes aditivos diretamente nas posições correspondentes de $\mathbf{A} = \begin{bmatrix} \mathbf{G} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix}$ e $\mathbf{z} = \begin{bmatrix} \mathbf{i} \\ \mathbf{e} \end{bmatrix}$:

#### A. Resistor linear ($R$, condutância $g = 1/R$) entre nós $p$ e $n$:
A Lei de Ohm $i = g(v_p - v_n)$ resulta em contribuições na Lei de Kirchhoff das Correntes (KCL):
* Nó $p$: $\mathbf{G}[p, p] \mathrel{+}= g$, $\mathbf{G}[p, n] \mathrel{-}= g$
* Nó $n$: $\mathbf{G}[n, p] \mathrel{-}= g$, $\mathbf{G}[n, n] \mathrel{+}= g$
*(Se $p$ ou $n$ for o nó terra `'0'`, a respectiva linha/coluna é omitida pois $v_0 \equiv 0$).*

#### B. Fonte Independente de Tensão ($V$) entre nós $p$ e $n$, índice auxiliar $k$:
Impõe a equação $v_p - v_n = V$:
* Coluna $\mathbf{B}$: $\mathbf{A}[p, k] \mathrel{+}= 1$, $\mathbf{A}[n, k] \mathrel{-}= 1$ (injeção da corrente $j_k$ nos nós $p$ e $n$)
* Linha $\mathbf{C}$: $\mathbf{A}[k, p] \mathrel{+}= 1$, $\mathbf{A}[k, n] \mathrel{-}= 1$
* Vetor $\mathbf{e}$: $\mathbf{z}[k] = V$

#### C. Fonte Independente de Corrente ($I$) saindo de $n$ e entrando em $p$:
A corrente fixa entra no nó $p$ e sai de $n$:
* $\mathbf{z}[p] \mathrel{+}= I$
* $\mathbf{z}[n] \mathrel{-}= I$

#### D. Fontes Dependentes / Controladas (Elementos SPICE E, G, F, H):
1. **VCVS ($E$, ganho $\mu$):** Tensão de saída controlada por tensão $v_p - v_n = \mu (v_{cp} - v_{cn})$.
   * Coluna $B$: estampa usual de fonte de tensão no índice auxiliar $k$.
   * Linha de restrição $k$: $\mathbf{A}[k, p] \mathrel{+}= 1$, $\mathbf{A}[k, n] \mathrel{-}= 1$, $\mathbf{A}[k, cp] \mathrel{-}= \mu$, $\mathbf{A}[k, cn] \mathrel{+}= \mu$.
2. **VCCS ($G$, transcondutância $g_m$):** Corrente de saída controlada por tensão $i = g_m (v_{cp} - v_{cn})$.
   * Não requer variável auxiliar. Estampa direta no bloco $\mathbf{G}$:
   * $\mathbf{G}[p, cp] \mathrel{-}= g_m$, $\mathbf{G}[p, cn] \mathrel{+}= g_m$
   * $\mathbf{G}[n, cp] \mathrel{+}= g_m$, $\mathbf{G}[n, cn] \mathrel{-}= g_m$
3. **CCCS ($F$, ganho $\beta$):** Corrente controlada pela corrente que atravessa a fonte de controle $V_{ctrl}$.
   * Como a corrente $I(V_{ctrl}) = -j_{k_{ctrl}}$ (convenção gerador), a injeção na KCL do nó $p$ e $n$ é estampada na coluna auxiliar $k_{ctrl}$:
   * $\mathbf{A}[p, k_{ctrl}] \mathrel{+}= \beta$, $\mathbf{A}[n, k_{ctrl}] \mathrel{-}= \beta$.
4. **CCVS ($H$, transresistência $r_m$):** Tensão de saída controlada por corrente $v_p - v_n = r_m I(V_{ctrl})$.
   * Linha auxiliar de restrição $k$: $\mathbf{A}[k, p] \mathrel{+}= 1$, $\mathbf{A}[k, n] \mathrel{-}= 1$, $\mathbf{A}[k, k_{ctrl}] \mathrel{+}= r_m$.

#### E. Amplificador Operacional Ideal (Op-Amp):
Modelado em malha fechada ideal ($A_{OL} \to \infty, R_{in} \to \infty, R_{out} = 0$):
* **Curto-Circuito Virtual:** $v(in_p) - v(in_n) = 0 \implies \mathbf{A}[k, in_p] \mathrel{+}= 1, \mathbf{A}[k, in_n] \mathrel{-}= 1$.
* **Corrente de Saída:** A corrente necessária para manter a saída é fornecida pela variável auxiliar $k$: $\mathbf{A}[out, k] \mathrel{+}= 1$.

#### F. Indutores e Capacitores em Regime Permanente CC:
* **Capacitor em CC:** Comporta-se como um circuito aberto ideal ($i_C = C \frac{dv}{dt} = 0$). Não adiciona condutância nem ramo ao MNA.
* **Indutor em CC:** Comporta-se como um curto-circuito ideal ($v_L = L \frac{di}{dt} = 0$). É modelado via condutância de alta condutividade ($g = 10^6\ \text{S}$, equivalente a resistência parasita de $1\ \mu\Omega$).

---

### 1.3 Regime Permanente Senoidal (Análise CA Fasorial)
Para sinais puramente senoidais em frequência angular $\omega = 2\pi f$, o circuito é transformado no domínio da frequência utilizando fasores complexos:
* Indutor: Impedância $Z_L = j\omega L \implies Y_L = \frac{1}{j\omega L} = -j\frac{1}{\omega L}$.
* Capacitor: Impedância $Z_C = \frac{1}{j\omega C} \implies Y_C = j\omega C$.
* Resistor: Impedância $Z_R = R \implies Y_R = \frac{1}{R}$.

O sistema MNA torna-se complexo ($\mathbf{A}_{AC} \in \mathbb{C}^{M \times M}, \mathbf{z}_{AC} \in \mathbb{C}^{M}$), resolvido por decomposição LU em dupla precisão complexa (`complex128`).
As grandezas de saída são decompostas em:
$$V(t) = |V| \cos(\omega t + \theta) \iff \mathbf{V} = |V| e^{j\theta} = V_{real} + j V_{imag}$$

---

### 1.4 Teorema de Tellegen e Auditoria Energética
Em qualquer rede concentrada e linear, a conservação global de potência estipula que a soma algébrica das potências absorvidas e fornecidas em todos os ramos $B$ é rigorosamente nula:

$$\sum_{k=1}^{B} v_k \cdot i_k = 0$$

Em regime CA, a potência aparente complexa $S = P + jQ$ obedece ao mesmo princípio conservativo:

$$\sum_{k=1}^{B} \mathbf{V}_k \cdot \mathbf{I}_k^* = 0 \implies \sum P_k = 0 \quad \text{e} \quad \sum Q_k = 0$$

O `pycircuit-solver` audita automaticamente a conservação de energia através do método `Solution.power_audit()` e `ACSolution.power_audit()`, reportando desvios numéricos com tolerância inferior a $10^{-9}\text{ W}$.

---

### 1.5 Extração Dual de Thévenin e Norton
O cálculo analítico e computacional de circuitos equivalentes entre terminais arbitrários $A$ e $B$ opera através de duas etapas:
1. **Tensão de Circuito Aberto ($V_{th}$):** Resolve o circuito nominal sem carga entre $A$ e $B$: $V_{th} = V(A) - V(B)$.
2. **Resistência Equivalente ($R_{th}$):** Utiliza um **algoritmo de teste dual**:
   * **Injeção de Tensão:** Desativa fontes independentes e conecta uma fonte de teste $V_{test} = 1.0\text{ V}$ entre $A$ e $B$. A corrente $I_{test}$ é obtida da solução MNA, resultando em $R_{th} = 1.0 / I_{test}$.
   * **Injeção de Corrente (Fallback para $R_{th} = 0$):** Se houver uma fonte de tensão ou curto direto entre $A$ e $B$, a matriz MNA com fonte de teste de tensão torna-se singular. O motor detecta essa condição e comuta automaticamente para a **injeção de corrente** ($I_{test} = 1.0\text{ A}$), medindo a tensão resultante $\Delta V$:
     $$R_{th} = \frac{\Delta V}{I_{test}} = 0.0\ \Omega$$
3. **Corrente de Norton ($I_N$):** Calculada por $I_N = V_{th} / R_{th}$ quando $R_{th} > 0$.
4. **Máxima Transferência de Potência:** Ocorre quando $R_L = R_{th}$, fornecendo $P_{max} = \frac{V_{th}^2}{4 R_{th}}$.

---

### 1.6 Análise Transitória no Domínio do Tempo (`engine/transient.py`)
Para simular a dinâmica contínua $v(t)$ e $i(t)$ no tempo, as equações constitutivas diferenciais dos elementos reativos são discretizadas numericamente utilizando **modelos companheiros Norton (*companion models*)**:

#### A. Discretização do Capacitor ($i = C \frac{dv}{dt}$):
* **Regra dos Trapézios (*Trapezoidal Rule*, 2ª ordem, $A$-estável):**
  $$i_{k} = G_{eq} \cdot v_{k} - I_{eq, k-1}$$
  Onde:
  $$G_{eq} = \frac{2C}{\Delta t}, \quad I_{eq, k-1} = G_{eq} \cdot v_{k-1} + i_{k-1}$$
* **Euler Implícito (*Backward Euler*, 1ª ordem, $L$-estável):**
  $$G_{eq} = \frac{C}{\Delta t}, \quad I_{eq, k-1} = G_{eq} \cdot v_{k-1}$$

#### B. Discretização do Indutor ($v = L \frac{di}{dt}$):
* **Regra dos Trapézios:**
  $$i_k = G_{eq} \cdot v_k + I_{eq, k-1}$$
  Onde:
  $$G_{eq} = \frac{\Delta t}{2L}, \quad I_{eq, k-1} = G_{eq} \cdot v_{k-1} + i_{k-1}$$
* **Euler Implícito:**
  $$G_{eq} = \frac{\Delta t}{L}, \quad I_{eq, k-1} = i_{k-1}$$

#### C. Otimização por Fatoração LU Pré-Computada:
Para um passo de tempo fixo $\Delta t$, as condutâncias equivalentes $G_{eq}$ são estáticas ao longo de toda a simulação. Portanto, a matriz de coeficientes $\mathbf{A} \in \mathbb{R}^{M \times M}$ é invariante no tempo.
O solver realiza a **decomposição LU de $\mathbf{A}$ apenas uma vez antes do laço temporal** (`scipy.linalg.lu_factor`). A cada instante $t_k$, apenas o vetor $\mathbf{z}(t_k)$ é atualizado com as correntes históricas e fontes variáveis, resolvendo cada passo de tempo via substituição direta e reversa (`lu_solve`) em complexidade $O(M^2)$, permitindo $10.000$ passos temporais em frações de segundo.

---

### 1.7 Semicondutores & Solver de Newton-Raphson (`engine/nonlinear.py`)

Circuitos reais contêm dispositivos semicondutores com características corrente-tensão estritamente não-lineares. O `pycircuit-solver` implementa um motor de **Newton-Raphson** para linearizar iterativamente cada componente não-linear em torno do seu ponto de operação ($Q$-point) a cada iteração:

$$\mathbf{f}(\mathbf{x}) = \mathbf{A}_{lin} \mathbf{x} + \mathbf{i}_{nl}(\mathbf{x}) - \mathbf{z}_{lin} = \mathbf{0}$$

A linearização de primeira ordem via série de Taylor produz os modelos companheiros Norton:

$$\mathbf{i}_{nl}(\mathbf{x}) \approx \mathbf{i}_{nl}(\mathbf{x}^{(k)}) + \mathbf{J}^{(k)} (\mathbf{x} - \mathbf{x}^{(k)}) = \mathbf{J}^{(k)} \mathbf{x} - \mathbf{I}_{eq}^{(k)}$$

Onde a matriz Jacobiana $\mathbf{J}^{(k)} = \left.\frac{\partial \mathbf{i}_{nl}}{\partial \mathbf{x}}\right|_{\mathbf{x}^{(k)}}$ contém as condutâncias dinâmicas e transcondutâncias, e $\mathbf{I}_{eq}^{(k)} = \mathbf{J}^{(k)} \mathbf{x}^{(k)} - \mathbf{i}_{nl}(\mathbf{x}^{(k)})$ é o vetor de correntes equivalentes independentes estampadas no vetor $\mathbf{z}$.

A cada iteração $k$, o sistema linear resultante é resolvido:

$$(\mathbf{A}_{lin} + \mathbf{J}^{(k)}) \mathbf{x}^{(k+1)} = \mathbf{z}_{lin} + \mathbf{I}_{eq}^{(k)}$$

O algoritmo itera até que a norma do resíduo satisfaça o critério estrito de parada:

$$\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|_\infty < \epsilon \quad (\text{com } \epsilon = 10^{-6}\text{ V})$$

#### Modelos de Dispositivos Implementados:

1. **Diodo de Junção PN (Equação de Shockley):**
   $$I_D = I_S \left(e^{V_D / (n V_T)} - 1\right), \quad g_d = \frac{\partial I_D}{\partial V_D} = \frac{I_S}{n V_T} e^{V_D / (n V_T)}$$
   * Amortecimento exponencial SPICE (`pnjlim`): previne *overflow* numérico em ponto flutuante ($e^{80} \approx 5.5 \times 10^{34}$) limitando o avanço de tensão por passo em junções fortemente polarizadas.

2. **Diodo Zener:**
   * Polarização direta: modelo padrão de diodo de junção PN.
   * Regulação reversa (efeito Zener / avalanche): conduz bruscamente quando $V_D \le -V_Z$ com condutância dinâmica $g_Z = 1 / R_Z$:
     $$I_D = -I_S - \frac{(-V_Z) - V_D}{R_Z}$$

3. **Transistor Bipolar de Junção (BJT — NPN e PNP):**
   * Modelo acoplado de Ebers-Moll com diodos de base-emissor ($I_F$) e base-coletor ($I_R$):
     $$I_B = \frac{I_F}{\beta} + I_R, \quad I_C = I_F - 2 I_R, \quad I_E = -(I_B + I_C)$$
   * Modela naturalmente a região ativa ($V_{CE} > 0.2\text{ V}, I_C = \beta I_B$) e a saturação profunda ($V_{CE} \approx V_{CE,sat} \approx 0.05\text{ a }0.1\text{ V}$).

4. **Transistor de Efeito de Campo (MOSFET — NMOS e PMOS):**
   * Modelo quadrático clássico para as 3 regiões de operação:
     * **Corte ($V_{GS} \le V_{th}$):** $I_D = 0, g_m = 0, g_{ds} = g_{min} = 10^{-12}\ \Omega^{-1}$.
     * **Triodo / Linear ($V_{DS} < V_{GS} - V_{th}$):** Canal resistivo bidirecional:
       $$I_D = K_p \left[ (V_{GS} - V_{th}) V_{DS} - \frac{1}{2} V_{DS}^2 \right] (1 + \lambda V_{DS})$$
     * **Saturação ($V_{DS} \ge V_{GS} - V_{th}$):**
       $$I_D = \frac{1}{2} K_p (V_{GS} - V_{th})^2 (1 + \lambda V_{DS})$$
   * Amortecimento de canal FET (`fetlim`): impede oscilações estáticas de tensão entre corte e saturação.

---

### 1.8 Varredura Paramétrica DC (.dc sweep — `engine/dc_sweep.py`)

A análise de ponto de operação estático (.op) calcula as grandezas elétricas para um único conjunto de valores constantes. A **Varredura Paramétrica DC (.dc sweep)** estende essa capacidade, permitindo analisar o comportamento do circuito sob variação contínua de um parâmetro (tensão, corrente ou resistência):

1. **Curvas de Transferência ($V_{out} \times V_{in}$):**
   Mapeia o ganho de amplificadores, limiares de comutação de comparadores e inversores lógicos.
2. **Curvas de Carga ($V_L \times I_L$ ou $P_L \times R_L$):**
   Avalia a regulação de fontes e máxima transferência de potência sobre resistores de carga variáveis.
3. **Varredura Aninhada 2D (Família de Curvas Características):**
   Permite variar um parâmetro primário (ex: $V_{CE}$ de 0 a 5V) para múltiplos patamares discretos de um parâmetro secundário (ex: $I_B \in [10, 20, 30, 40, 50]\ \mu\text{A}$), gerando matrizes de resultados $\mathbf{V} \in \mathbb{R}^{N_2 \times N_1}$ e $\mathbf{I} \in \mathbb{R}^{N_2 \times N_1}$ e traçando diretamente as curvas características de dispositivos semicondutores ($I_C \times V_{CE}$, $I_D \times V_{DS}$).
4. **Continuação de Solução Homotópica:**
   Em circuitos não-lineares, a solução $\mathbf{x}^{(k)}$ do passo anterior é utilizada como aproximação inicial (*initial guess*) para o passo subsequente $\mathbf{x}^{(k+1)}$, garantindo convergência ultrarrápida do Newton-Raphson (tipicamente 2 a 4 iterações por ponto).

---

### 1.9 Transformadores e Acoplamento Magnético Mútuo (Elemento K do SPICE — Alexander & Sadiku Cap. 13)

Circuito com indutores acoplados magneticamente por fluxo compartilhado (transformadores de dois enrolamentos e elemento de acoplamento mútuo SPICE `K`):

1. **Equações Constitutivas e Indutância Mútua:**
   A indutância mútua $M$ entre duas bobinas $L_1$ e $L_2$ com coeficiente de acoplamento magnético $k$ ($-1.0 \le k \le 1.0$) é regida por:
   $$M = k \sqrt{L_1 L_2}$$
   No domínio do tempo:
   $$\begin{bmatrix} v_1(t) \\ v_2(t) \end{bmatrix} = \begin{bmatrix} L_1 & M \\ M & L_2 \end{bmatrix} \frac{d}{dt}\begin{bmatrix} i_1(t) \\ i_2(t) \end{bmatrix} = \mathbf{L} \frac{d\mathbf{i}}{dt}$$
   A matriz de indutância inversa $\mathbf{\Gamma} = \mathbf{L}^{-1}$ (unidade $\text{H}^{-1}$) é dada por:
   $$\det(\mathbf{L}) = L_1 L_2 - M^2 = L_1 L_2 (1 - k^2)$$
   $$\mathbf{\Gamma} = \frac{1}{\det(\mathbf{L})} \begin{bmatrix} L_2 & -M \\ -M & L_1 \end{bmatrix}$$
   Para prevenir singularidade numérica quando $|k| = 1.0$, o solver aplica clamp $|k| \le 0.9999999$.

2. **Estampagem de Admitância AC (Regime Permanente Senoidal):**
   No domínio fasorial:
   $$\mathbf{I} = \frac{1}{j\omega} \mathbf{\Gamma} \mathbf{V} = \mathbf{Y}_{coupled} \mathbf{V}$$
   onde $\mathbf{Y}_{coupled} = -j \frac{\mathbf{\Gamma}}{\omega}$. A matriz de duas portas é estampada na matriz nodal MNA considerando os nós do primário $(p_1, n_1)$ e do secundário $(p_2, n_2)$:
   * Auto-admitância primário: $+Y_{11}$ em $(p_1, p_1)$ e $(n_1, n_1)$; $-Y_{11}$ em $(p_1, n_1)$ e $(n_1, p_1)$.
   * Auto-admitância secundário: $+Y_{22}$ em $(p_2, p_2)$ e $(n_2, n_2)$; $-Y_{22}$ em $(p_2, n_2)$ e $(n_2, p_2)$.
   * Acoplamento cruzado mútuo:
     $$\mathbf{A}[p_1, p_2] \mathrel{+}= Y_{12}, \quad \mathbf{A}[p_1, n_2] \mathrel{-}= Y_{12}, \quad \mathbf{A}[n_1, p_2] \mathrel{-}= Y_{12}, \quad \mathbf{A}[n_1, n_2] \mathrel{+}= Y_{12}$$
     $$\mathbf{A}[p_2, p_1] \mathrel{+}= Y_{21}, \quad \mathbf{A}[p_2, n_1] \mathrel{-}= Y_{21}, \quad \mathbf{A}[n_2, p_1] \mathrel{-}= Y_{21}, \quad \mathbf{A}[n_2, n_1] \mathrel{+}= Y_{21}$$

3. **Modelo Companheiro Transiente (Domínio do Tempo):**
   Integrando numericamente $\frac{d\mathbf{i}}{dt} = \mathbf{\Gamma} \mathbf{v}(t)$ com passo $\Delta t$:
   $$\mathbf{G}_{eq} = \begin{cases} \frac{\Delta t}{2} \mathbf{\Gamma} & \text{(Regra Trapezoidal)} \\ \Delta t \mathbf{\Gamma} & \text{(Backward Euler)} \end{cases}$$
   $$\mathbf{I}_{eq, k} = \begin{cases} \mathbf{i}_{k-1} + \mathbf{G}_{eq} \mathbf{v}_{k-1} & \text{(Regra Trapezoidal)} \\ \mathbf{i}_{k-1} & \text{(Backward Euler)} \end{cases}$$
   A matriz $\mathbf{G}_{eq}$ é estampada invariantemente no tempo na matriz $\mathbf{A}$, e o vetor de correntes históricas $\mathbf{I}_{eq}$ é injetado no vetor $\mathbf{z}$ a cada passo de tempo ($-I_{eq, 1}$ em $p_1$, $+I_{eq, 1}$ em $n_1$; $-I_{eq, 2}$ em $p_2$, $+I_{eq, 2}$ em $n_2$).

4. **Regime Permanente DC:**
   Em corrente contínua estática ($\omega = 0$), os enrolamentos de indutores ideais comportam-se como curto-circuitos perfeitos (condutância parasita $g = 10^6\ \Omega^{-1}$).

5. **Sintaxe SPICE Suportada:**
   * Acoplamento mútuo entre dois indutores: `K<name> <L1> <L2> <k>` (ex: `K1 L1 L2 0.999`)
   * Transformador integrado de 2 enrolamentos: `T<name> <p1> <n1> <p2> <n2> <L1> <L2> [k]` (ex: `T1 1 0 2 0 10m 2.5m 0.999`)

---

## 2. 🛡️ Validação Topológica e Teoria dos Grafos (`validator.py`)

A maior causa de falha em simuladores de circuitos (inclusive no SPICE original) são circuitos fisicamente incoerentes que resultam em matrizes singulares (`LinAlgError`). O `pycircuit-solver` intercepta esses problemas **antes** de montar as matrizes através de algoritmos de Teoria dos Grafos:

```
[Netlist / Elementos] 
         ↓
 ┌─────────────────────────────────────────────────────────┐
 │                  CircuitValidator                       │
 │  1. Validação Léxica: valores > 0, nomes válidos        │
 │  2. Verificação de Terra: existência obrigatória de '0' │
 │  3. Conectividade BFS: detecção de sub-circuitos ilhados│
 │  4. Detecção DFS de Loops de Fontes de Tensão (KVL loop)│
 │  5. Verificação de Fontes de Controle de CCCS / CCVS    │
 └─────────────────────────────────────────────────────────┘
         ↓ (Circuito Topologicamente Seguro)
   [MNA Matrix Assembly]
```

### 2.1 Algoritmo BFS para Conectividade com o Terra
O circuito é interpretado como um grafo não-direcionado onde os nós do circuito são vértices e os elementos bifilares são arestas.
* Executa uma Busca em Largura (BFS) partindo do nó de referência `'0'`.
* Se algum nó não for visitado durante a travessia, esse nó pertence a uma "ilha flutuante" desconectada do terra. O validador rejeita a topologia com uma exceção explícita indicando exatamente o nó flutuante.

### 2.2 Algoritmo DFS para Detecção de Conflitos KVL (Loops de Tensão)
Duas fontes de tensão ideais em paralelo com valores diferentes, ou um anel fechado formado exclusivamente por fontes de tensão, violam a Lei de Kirchhoff das Tensões ($\sum V \neq 0$).
* O validador filtra o subgrafo composto exclusivamente por ramos de fontes ideais de tensão.
* Executa Busca em Profundidade (DFS) rastreando ciclos. Ao encontrar um ciclo de fontes de tensão, rejeita a simulação com erro topológico descritivo.

### 2.3 Tratamento de Sufixos de Engenharia
O módulo `parse_value` converte strings de notação científica e laboratorial em floats de SI:
* `1k` / `1K` $\to 1 \times 10^3$
* `4k7` $\to 4.7 \times 10^3$
* `10u` / `10uF` $\to 10 \times 10^{-6}$
* `1MEG` / `1meg` $\to 1 \times 10^6$
* `2.2m` / `2.2mA` $\to 2.2 \times 10^{-3}$

---

## 3. 🦀 Acelerador Híbrido em Rust (`rust_core`)

### 3.1 Por que Rust para o PyCircuit?
Em análises determinísticas de um único circuito (10 a 50 nós), o NumPy resolve o sistema linear em menos de $1\text{ ms}$, tornando o overhead imperceptível. No entanto, em **análises estatísticas de Monte Carlo**, o circuito precisa ser perturbado e resolvido **milhares de vezes** ($10.000$ a $100.000$ iterações).

No Python puro:
* O interpretador CPython possui o **GIL (*Global Interpreter Lock*)**, impedindo que threads Python executem computação numérica em paralelo real em processadores multi-core.
* A alocação contínua de objetos Python e dicionários de mapeamento de nós dentro do loop de Monte Carlo gera enorme pressão no Garbage Collector.

### 3.2 Arquitetura de Integração (PyO3 + Rayon + Nalgebra)
A camada Rust elimina esses gargalos operando como uma extensão nativa:

```
[Python User Space]
    Circuit, monte_carlo(circuit, n_samples=50_000)
         │
         ▼
[monte_carlo_rust.py] (Bridge Layer)
    - Converte rótulos de nós (strings) em inteiros 0..N
    - Empacota estruturas PyResistor, PyVSource, PyISource
         │  (Chamada CPython via PyO3 C-ABI)
         ▼
[rust_core (C-Extension compilada com LTO e Opt-Level 3)]
    ┌────────────────────────────────────────────────────────┐
    │  lib.rs (PyO3 interface)                               │
    ├────────────────────────────────────────────────────────┤
    │  rayon::par_iter                                       │
    │  ├── Thread 1: ChaCha8Rng (seed+1) → MNA LU Solve      │
    │  ├── Thread 2: ChaCha8Rng (seed+2) → MNA LU Solve      │
    │  ├── Thread 3: ChaCha8Rng (seed+3) → MNA LU Solve      │
    │  └── Thread N: ChaCha8Rng (seed+N) → MNA LU Solve      │
    ├────────────────────────────────────────────────────────┤
    │  nalgebra::full_piv_lu() (Pivotamento Total Estável)   │
    │  Estatísticas Nativas: Média, Desvio Padrão, P5, P95   │
    └────────────────────────────────────────────────────────┘
         │
         ▼ (Zero-Copy via numpy::IntoPyArray)
Retorno instantâneo para Python como NumPy ndarrays
```

### 3.3 Ganhos de Desempenho Medidos
Em testes automatizados no hardware de desenvolvimento (Apple Silicon M-series):
* **5.000 amostras (Ponte de Wheatstone):**
  * Python puro: $\approx 180\text{ ms}$
  * Rust multi-thread (`rayon`): $\approx 12\text{ ms}$
  * **Aceleração (*Speedup*): $\mathbf{15\times}$ mais rápido.**
* **50.000 amostras:**
  * Concluído em menos de $0.15\text{ s}$ em Rust.

### 3.4 Padrão de Resiliência: *Graceful Fallback*
O módulo `circuit_solver.engine.monte_carlo_rust` verifica a disponibilidade do binário compilado `pycircuit_core`. Caso o usuário esteja em um ambiente sem compilador Rust, o solver automaticamente utiliza o motor Python puro sem lançar exceções e sem alterar a interface pública da função `monte_carlo()`.

---

## 4. 🗂️ Estrutura e Responsabilidades dos Módulos

```
pycircuit-solver/
├── circuit_solver/
│   ├── core/
│   │   ├── elements.py     # Dataclasses de componentes (Resistor, Fontes, OpAmp, etc.)
│   │   ├── circuit.py      # Gestor de topologia e dicionários de mapeamento nodal
│   │   ├── validator.py    # Grafos (BFS/DFS), coerência de nós e conversão de unidades
│   │   └── solution.py     # Encapsulador do vetor solução com Tellegen e métodos V(node)
│   ├── engine/
│   │   ├── mna.py          # Montagem matricial MNA e solução linear DC
│   │   ├── ac.py           # Formulação fasorial em complex128 e resposta em frequência
│   │   ├── thevenin.py     # Algoritmo de injeção dual de Thévenin/Norton e P_max
│   │   ├── monte_carlo.py  # Implementação de referência em Python
│   │   └── monte_carlo_rust.py # Ponte de alta performance para o crate Rust
│   ├── io/
│   │   └── parser.py       # Parser de netlists padrão SPICE com suporte a '+'
│   ├── visualization/
│   │   └── plots.py        # Gráficos de Monte Carlo, Thévenin e Diagramas de Bode
│   └── cli.py              # Interface CLI (pycircuit solve/thevenin/ac/monte-carlo)
│
├── rust_core/              # Crate Rust nativo
│   ├── Cargo.toml          # Dependências: pyo3, rayon, nalgebra, rand_chacha, numpy
│   └── src/
│       ├── lib.rs          # Definição do módulo PyO3
│       ├── types.rs        # Structs sem overhead de strings
│       ├── mna_core.rs     # Montagem e decomposição LU pura em Rust
│       └── monte_carlo.rs  # Rayon data-parallel worker
│
└── tests/                  # 68 testes unitários e de integração
```

---

## 5. 📊 Matriz de Validação Multi-Livros (68 Testes)

A biblioteca é auditada contra exercícios canônicos clássicos com respostas analíticas publicadas:

1. **Alexander & Sadiku:**
   * Divisores e reduções série-paralelo (Cap. 2).
   * Transformação estrela-triângulo ($\Delta - Y$) em ponte (Cap. 2).
   * Análise nodal com supernós e fontes de corrente (Cap. 3).
   * Análise com fonte controlada VCCS (Cap. 3).
   * Teorema de Thévenin com VCVS dependente (Cap. 4).
   * Circuitos com Op-Amp: Inversor, Não-Inversor, Somador e Subtrator (Cap. 5).
   * Regime permanente CA fasorial RLC série (Cap. 9).
2. **Nilsson & Riedel:**
   * Equivalente de Thévenin com fonte dependente mista (Cap. 4).
   * Somador inversor com ponderação de ganhos arbitrários (Cap. 5).
   * Circuito RLC paralelo em alta frequência (Cap. 9).
3. **Boylestad:**
   * Teorema de Millman para baterias em paralelo (Cap. 9).
   * Escada R-2R de 4 bits para conversão Digital/Analógica (Cap. 8).
   * Ponte de Wheatstone em condição de equilíbrio de nulo (Cap. 6).
4. **Dorf & Svoboda:**
   * Op-Amp com rede de realimentação em T (Cap. 6).
   * Buffer seguidor de tensão e isolamento de impedância (Cap. 6).
5. **Casos Degenerados e Robustez:**
   * Redes com $R_{th} = 0\ \Omega$ (fonte ideal nos terminais).
   * Detecção de loops fechados de fontes de tensão.
   * Isolamento de subcircuitos sem terra via BFS.
   * Parser com linhas longas quebradas com o operador SPICE `+`.

---

## 6. 🚀 Como Utilizar o Sistema

### 6.1 Via Código Python
```python
from circuit_solver import Circuit, VoltageSource, Resistor, solve
from circuit_solver.engine.thevenin import thevenin_norton
from circuit_solver.engine.monte_carlo_rust import monte_carlo

# 1. Definir circuito
circuit = Circuit([
    VoltageSource("V1", "1", "0", 24.0),
    Resistor("R1", "1", "2", 10.0, tolerance=0.05),
    Resistor("R2", "2", "0", 20.0, tolerance=0.05),
    Resistor("Rload", "2", "0", 30.0)
])

# 2. Resolução Nodal
sol = solve(circuit)
print(f"Tensão no nó 2: {sol.V('2'):.4f} V")
print(f"Auditoria de Potência: {sol.power_audit()['balance']:.2e} W")

# 3. Equivalente de Thévenin entre nó 2 e terra
th = thevenin_norton(circuit, node_a="2", node_b="0")
print(f"Vth = {th.v_th:.2f} V, Rth = {th.r_th:.2f} Ohms, Pmax = {th.p_max:.2f} W")

# 4. Simulação Estatística Acelerada em Rust
mc = monte_carlo(circuit, n_samples=10000, seed=42)
print(mc.summary())
```

### 6.2 Via Linha de Comando (CLI)
```bash
# Resolver netlist SPICE
pycircuit solve circuito.cir

# Extrair equivalente de Thévenin
pycircuit thevenin circuito.cir -a 2 -b 0 --plot

# Análise de Monte Carlo com histograma
pycircuit monte-carlo circuito.cir -n 10000 -t 0.05 --plot

# Resposta em frequência CA (Diagrama de Bode)
pycircuit ac filtro.cir --sweep 10 100000 100 --node out --plot
```

---

## 7. 🦀 Núcleo Rust Expandido e Solvers Esparsos para Circuitos Gigantes

### 7.1 Expansão do Núcleo Rust para 100% dos Elementos Lineares
O crate `pycircuit_core` (em Rust) foi expandido para cobrir **100% dos elementos lineares** suportados pelo framework:
* **Resistores** com tolerâncias estatísticas gaussianas.
* **Fontes Independentes:** Tensão (`V`) e Corrente (`I`).
* **Fontes Dependentes / Controladas:**
  * VCVS ($E$): Fonte de tensão controlada por tensão ($\mu$).
  * VCCS ($G$): Fonte de corrente controlada por tensão ($g_m$).
  * CCCS ($F$): Fonte de corrente controlada por corrente ($\beta$).
  * CCVS ($H$): Fonte de tensão controlada por corrente ($r_m$).
* **Amplificadores Operacionais Ideais:** Elemento $U$/$A$ com curto-circuito virtual ($V_{in+} - V_{in-} = 0$) e ramo de saída auxiliar.
* **Indutores em DC:** Estampados como curto-circuito ideal com condutância parasita controlada ($10^6\ \Omega^{-1}$).

Com isso, circuitos ativos complexos (filtros ativos, amplificadores de instrumentação com Op-Amps, circuitos a transistores em pequenos sinais com fontes controladas) rodam diretamente no kernel multi-thread Rayon em Rust, executando 10.000 amostras de Monte Carlo em milissegundos sem recorrer ao fallback Python.

### 7.2 Solvers Esparsos para Circuitos Gigantes (> 1.000 nós)
Para redes de grande porte (malhas de distribuição, chips VLSI, escadas R-2R de alta resolução com centenas ou milhares de nós):
1. **Montagem Esparsa Triplet/COO:**
   * `build_mna_sparse(circuit)` coleta as estampas em formato de coordenadas (COO) e as converte diretamente para matriz de colunas comprimidas (`scipy.sparse.csc_matrix`), somando condutâncias paralelas automaticamente em complexidade $O(N)$.
2. **Resolução com SuperLU no Python:**
   * Utiliza `scipy.sparse.linalg.spsolve`, reduzindo o consumo de memória de $O(N^2)$ para $O(N)$ e o tempo de execução de segundos para frações de milissegundo.
3. **Solver Esparso Nativo em Rust (`faer`):**
   * Implementado em `mna_core::solve_mna_sparse` e exposto via `pycircuit_core.solve_mna_sparse`.
   * Realiza fatoração simbólica e numérica $LU$ esparsa com suporte a instruções SIMD nativas e paralelismo de thread pool, resolvendo circuitos com mais de 1.000 nós em menos de 50 ms.
4. **Despacho Automático (`solver="auto"`):**
   * O solver `solve(circuit, solver="auto", sparse_threshold=500)` automaticamente ativa o solver esparso quando o número de incógnitas atinge o limiar configurado (padrão 500 nós).

---

## 8. 🧩 Suporte a Subcircuitos SPICE Hierárquicos e Mecanismo White-Box Pedagógico

### 8.1 Compatibilidade Industrial Padrão SPICE (.subckt / .ends e Elementos X)
O `pycircuit` implementa o mecanismo de modularização hierárquica compatível com as ferramentas comerciais de EDA (SPICE, LTspice, HSPICE, PSpice):
1. **Definição de Macromodelos:**
   Blocos reutilizáveis definidos por `.SUBCKT <nome> <pinos...> [PARAMS: p1=v1 ...]`, encerrados por `.ENDS [<nome>]`.
2. **Instanciação Modular:**
   Instâncias `X<nome> <nós_conectados...> <modelo> [PARAMS: p1=v1 ...]`.
3. **Avaliação Paramétrica e Expressões Matemáticas:**
   Suporte a parâmetros padrão, sobrescrita de instância e fórmulas matemáticas SPICE entre chaves `{...}` (ex: `{1 / (2 * pi * FC * C)}`, `{R1 * 2}`), com suporte a sufixos de engenharia (`k`, `u`, `n`, `p`, `MEG`, etc.) e funções elementares (`sin`, `cos`, `sqrt`, `exp`, etc.).
4. **Subcircuitos Aninhados Recursivos (*Nested Subcircuits*):**
   Suporte a múltiplos níveis de hierarquia (subcircuitos que instanciam outros subcircuitos), com qualificação correta de escopo (`X_BLOCO.X_ESTAGIO.R1`).
5. **Diretiva de Inclusão de Arquivos Externos:**
   Suporte a `.INCLUDE "<arquivo.lib>"` e `.INC "<arquivo.cir>"`, permitindo carregar modelos de fabricantes comerciais e bibliotecas acadêmicas.

### 8.2 Arquitetura Pedagógica de Introspecção (*White-Box Hierarchy Engine*)
Projetado especificamente para pesquisa acadêmica, dissertações de mestrado e ensino de Engenharia Elétrica, eliminando o comportamento "caixa-preta" (*black box*) dos simuladores proprietários:
1. **Árvore Hierárquica Inspecionável (`HierarchyTree`):**
   Acesso em tempo de execução via Python (`circuit.hierarchy_tree()`) e via linha de comando (`pycircuit hierarchy circuito.cir`).
2. **Relatório Didático Passo a Passo (`circuit.explain_hierarchy()` / `pycircuit explain`):**
   * Tabela visual "Pino Formal $\rightarrow$ Nó Global".
   * Resolução detalhada de cada parâmetro e fórmula.
   * Lista de nós internos privados criados (`X1.int1`).
   * **Análise de Impacto Dimensional MNA:** Quantifica exatamente quantos nós nodais e variáveis de corrente auxiliar cada subcircuito adicionou e o crescimento da ordem da matriz $\mathbf{A}$.
3. **Sondagem de Grandezas Internas (*Scoped Probing*):**
   Permite medir grandezas internas de qualquer bloco com notação de ponto: `sol.V("X1.diff_pair")` ou gerar raio-X completo do bloco com `sol.subcircuit_summary("X1")`.

### 8.3 Biblioteca Pedagógica Embutida (*Textbook Library*)
O módulo `circuit_solver.core.builtin_library` disponibiliza macromodelos de livros-texto consagrados (Alexander & Sadiku, Sedra & Smith):
* `OPAMP_741`: Macromodel clássico do LM741 com resistência de entrada diferencial ($R_{in} = 2\text{ M}\Omega$), ganho em malha aberta finito ($A_{OL} = 200.000$) e impedância de saída ($R_{out} = 75\ \Omega$).
* `OPAMP_IDEAL`: Curto-circuito virtual ideal com ganho infinito.
* `RC_LOWPASS`: Filtro passa-baixa paramétrico de 1ª ordem.
* `SALLEN_KEY_LP`: Filtro ativo passa-baixas de 2ª ordem com topologia Sallen-Key.
* `CMOS_INVERTER`: Célula lógica digital inversora CMOS com par complementar NMOS/PMOS.
* `BJT_DIFF_PAIR`: Par diferencial básico a transistor bipolar com resistores de cauda e carga.

---

## 9. 🔬 Análise CA de Pequenos Sinais com Semicondutores e Ponto de Operação (.op)

A análise fasorial em regime permanente senoidal clássica aplica-se a redes estritamente lineares ($R, L, C$). Para amplificadores de áudio, rádio-frequência (RF) e circuitos com transistores, o `pycircuit` implementa a metodologia rigorosa em duas etapas utilizada nos simuladores industriais e nos livros de Microeletrônica (Sedra & Smith, Neamen, Boylestad):

### 9.1 Algoritmo em Duas Etapas (Ponto Q DC + Linearização Incremental)

1. **Passo 1: Ponto de Operação DC Estático ($Q$-Point — Newton-Raphson):**
   * Fontes de sinal CA são desativadas temporariamente ($V_{ac} = 0, I_{ac} = 0$).
   * Capacitores comportam-se como circuitos abertos ($g = 0$) e indutores como curto-circuitos ($g = 10^6\ \Omega^{-1}$).
   * O solver de Newton-Raphson com amortecimento `pnjlim` e `fetlim` converge para as tensões dos nós $\mathbf{V}_{DC}$ e as correntes estáticas de cada semicondutor ($I_D, I_C, I_B, I_D$).

2. **Passo 2: Modelagem Incremental de Pequenos Sinais (Jacobiano Analítico):**
   No ponto $Q$, cada semicondutor é substituído pelo seu circuito equivalente linear:
   * **Diodo / Zener:**
     $$r_d = \frac{n V_T}{I_{D0} + I_S}, \quad g_d = \frac{1}{r_d}$$
   * **Transistor Bipolar (BJT — Modelo Híbrido-$\pi$ de Giacoleto):**
     * Transcondutância: $g_m = \frac{|I_{C0}|}{V_T}$
     * Resistência de entrada base-emissor: $r_\pi = \frac{\beta}{g_m} = \frac{V_T}{|I_{B0}|}$
     * Resistência de saída de coletor: $r_o = \frac{V_A + |V_{CE0}|}{|I_{C0}|}$
     * Estampa CA: Resistor $r_\pi$ entre base e emissor, resistor $r_o$ entre coletor e emissor, e fonte de corrente controlada por tensão (VCCS) $g_m \cdot v_{be}$ fluindo do coletor para o emissor.
   * **Transistor de Efeito de Campo (MOSFET):**
     * Na saturação ($V_{DS} \ge V_{GS} - V_{th}$):
       $$g_m = K_p (V_{GS} - V_{th}) (1 + \lambda V_{DS}) = \sqrt{2 K_p I_D (1 + \lambda V_{DS})}, \quad r_{ds} = \frac{1}{\lambda I_D}$$
     * No triodo ($V_{DS} < V_{GS} - V_{th}$):
       $$g_m = K_p V_{DS} (1 + \lambda V_{DS}), \quad r_{ds} = \frac{1}{g_{ds}}$$
     * Estampa CA: Fonte VCCS $g_m \cdot v_{gs}$ do dreno para a fonte e resistor de saída $r_{ds}$.

3. **Passo 3: Superposição de Fontes e Resolução Fasorial AC:**
   * Fontes CC puras de polarização ($V_{CC}, V_{DD}$) tornam-se terra CA ($V_{ac} = 0$).
   * Fontes CA independentes fornecem a excitação senoidal complexa $V_{ac} e^{j\theta}$.
   * O sistema complexo $\mathbf{A}_{AC}(j\omega) \mathbf{X}(j\omega) = \mathbf{Z}_{AC}$ é montado e resolvido diretamente para cada frequência da varredura (`ac_sweep`), traçando diagramas de Bode de amplificadores Emissor Comum, Coletor Comum e Fonte Comum.

### 9.2 Relatório Pedagógico do Ponto de Operação (`pycircuit op`)
Permite ao estudante ou pesquisador inspecionar instantaneamente:
* Tensão nos terminais e polarização de junção ($V_D, V_{BE}, V_{CE}, V_{GS}, V_{DS}$).
* Identificação automática da região de operação: **Corte**, **Saturação** ou **Ativa Direta / Triodo**.
* Parâmetros calculados do modelo equivalente ($g_m, r_\pi, r_o, r_d, r_{ds}$).

