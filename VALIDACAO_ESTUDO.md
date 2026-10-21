# Validação das correções para estudo pessoal

Executada em 21/10/2026, macOS arm64, Python 3.14.8. A suíte Python foi executada com a extensão abi3 recém-compilada, extraída da wheel em um diretório temporário. A wheel também foi instalada na `.venv` do projeto.

- `maturin build --release --features extension-module --interpreter python3`: compilação release concluída.
- `python3 -m pytest -q -rs`: 468 aprovados, 10 ignorados por circuitos aleatórios singulares; inclui 31 novos casos de regressão e execução dos exemplos Python do tutorial.
- `cargo test --release --offline`, com `PYO3_PYTHON` apontando para Python 3.12: concluído; o crate atualmente contém zero testes nativos. Os testes numéricos Rust são chamados pela suíte Python, incluindo comparação entre backends e ordem de convergência.
- Ruff nos módulos alterados e `git diff --check`: sem erros.
- Memória regenerada: 48 exercícios, 164 comparações aprovadas segundo os critérios publicados.

## Escopo matemático

A varredura conserva o passo, aceita sentido decrescente e preserva o circuito original, inclusive em falhas. Os testes verificam também varredura aninhada e parâmetros inválidos.

O transitório permanece linear: semicondutores geram erro explícito. Indutores acoplados e transformadores no backend Python são verificados contra a solução independente por exponencial matricial, para Euler e trapézio, incluindo polaridade e energia mútua.

A potência trifásica usa fasores RMS e a identidade de três fios S=Vac·conj(Ia)+Vbc·conj(Ib). Os testes cobrem equilíbrio, desequilíbrio e ambas as sequências. Sistemas com corrente de neutro requerem as três potências fase–neutro.

A memória distingue aproximações de VBE constante das referências exponenciais, calculadas por equações escalares independentes. O cruzamento por zero do transitório usa tolerância absoluta explícita de 0,0002 V com dt=2 µs. A aprovação é calculada a partir das comparações; valores não finitos e divergências recebem indicação própria.

RSS combina contribuições de primeira ordem; não é automaticamente 3σ. Os dois cantos selecionados pela sensibilidade nominal não garantem extremos globais. Falhas nas resoluções desses cantos não são substituídas silenciosamente por estimativas de Taylor.


## Complemento: estados acoplados e fontes controladas

- Adicionados 32 testes Python: condições iniciais não nulas, solução por exponencial matricial, balanço de energia, ambas as polaridades de acoplamento, início pelo ponto DC, comparação completa entre backends, falha DC com UIC explícito e derivadas/cantos de CCCS e CCVS conferidos por diferenças finitas.
- Suíte final: 500 aprovados, 10 ignorados por circuitos aleatórios singulares.
- Adicionados e executados 3 testes nativos Rust, cobrindo UIC/energia, corrente inicial DC dos enrolamentos e falha DC sem substituição por zero. A indicação anterior de zero testes nativos descreve a validação anterior.
- Extensão Rust reconstruída em release e instalada novamente na `.venv`.

Limite explícito: o ponto DC aproxima os enrolamentos por 1 µΩ em ambos os backends. Em falhas, o usuário precisa escolher UIC conscientemente; o solver não toma essa decisão silenciosamente.


## Complemento: descartes numéricos auditáveis

A suíte agora só captura `numpy.linalg.LinAlgError` no solver de referência dos testes aleatórios e exige também cond(A)>1e12 para tratar essa exceção como descarte esperado. Uma LinAlgError em matriz bem condicionada e qualquer outro erro propagam como falha. A mesma regra protege a contagem de circuitos resolvíveis.

Foram adicionadas 11 regressões com erros injetados. A suíte completa passou com 511 aprovados e 10 ignorados. Os dez casos continuam fora da comparação de soluções, sem serem contabilizados como aprovação; o relatório agora exibe semente, presença de transformador, condicionamento e exceção original.

Sementes dos descartes nesta execução:

- Sem transformadores: 16 e 57.
- Com transformadores: 8, 20, 27, 31, 32, 43, 47 e 57.

Esses descartes indicam condicionamento acima do limite de comparação, não uma demonstração de singularidade matemática exata. Os testes de igualdade das estampas densa/esparsa continuam sendo executados para essas sementes.

## Complemento: ambiente local reparado e instalação reproduzida

A validação anterior usava Python 3.14 externo à `.venv`. A pendência foi encerrada com a instalação e execução completas em CPython 3.12.14, NumPy 2.2.6 e SciPy 1.14.1, respeitando as dependências declaradas do projeto.

- Instaladas as dependências transitivas ausentes Pillow 13.0.0 e pyparsing 3.3.3. Reinstalado maturin 1.15.0 para recuperar seu executável na `.venv`.
- Removidos apenas arquivos auxiliares AppleDouble com assinatura verificada dentro da `.venv`. Esses arquivos estavam sendo confundidos com metadados de distribuições e estilos do Matplotlib. A rotina está em `scripts/clean_venv_appledouble.py`; ela exige `pyvenv.cfg`, não segue arquivos simbólicos e preserva arquivos que não tenham a assinatura AppleDouble.
- `pyproject.toml` continua como fonte das dependências. `requirements.txt` agora instala esse mesmo projeto; maturin faz parte do extra de desenvolvimento.
- `constraints-study-py312.txt` registra as versões de dependências diretas e transitivas validadas para Python 3.12/macOS arm64. A extensão Rust é construída separadamente com `rust_core/Cargo.lock`.
- Extensão Rust recompilada em release usando o interpretador da `.venv`, instalada e importada nela.
- `pip check` na `.venv`: nenhuma dependência quebrada. Suíte completa: **511 aprovados e 10 descartes numéricos diagnosticados**.
- Instalação reproduzida em outro ambiente Python 3.12 criado do zero, com as constraints e a wheel Rust recém-compilada. `pip check` também passou; suíte completa: **511 aprovados e os mesmos 10 descartes**.

As execuções usaram `MPLBACKEND=Agg` para gráficos sem interface e diretórios temporários separados de configuração do Matplotlib. O README registra como reproduzir a instalação e executar os testes usando a `.venv` do projeto.


## Complemento: cinco roteiros de estudo testados

Executadas 19 configurações dos cinco roteiros: divisor nos solvers denso, esparso Python e esparso Rust; RC/RL em Euler e trapézio nos dois backends; transformador com acoplamento positivo/negativo nos dois backends; trifásico desequilibrado nas sequências ABC/ACB; tolerâncias com Monte Carlo Python/Rust. Acrescentadas quatro verificações de RMS/pico para fontes Y/DELTA em ambas as sequências: **23 novas regressões**.

As referências são independentes: Ohm/Kirchhoff, exponenciais de primeira ordem, equações dos enrolamentos/exponencial matricial, deslocamento do neutro por admitâncias e quadratura de Gauss-Legendre da distribuição exata do divisor. O Monte Carlo usa 8.000 amostras por backend e semente 42. Os resultados e critérios constam em `RELATORIO_ROTEIROS_ESTUDO.md`.

A validação integrada revelou e corrigiu duas inconsistências: `P_resistor` retorna dissipação positiva, embora a auditoria use absorção negativa; e `build_three_phase_source` deve converter a entrada RMS para pico, seguindo a convenção de `solve_ac`. Os consumidores trifásicos agora convertem os fasores resolvidos para RMS antes dos wattímetros; o exercício 47 também mostra corretamente a diferença entre correntes de ramo e linha em delta.

Suíte completa na `.venv`: **534 aprovados e os mesmos 10 descartes numéricos diagnosticados**. Memória regenerada: **164/164 comparações aprovadas**. Ruff e `git diff --check` passaram.


## Complemento: curto ideal DC e auditoria dos antigos descartes

A aproximação por 1 µΩ foi substituída pela estampa exata Vp−Vn=0, com uma corrente auxiliar por enrolamento. O índice específico DC (`dc_size`) preserva as dimensões AC/transitórias; as correntes resolvidas alimentam a inicialização transitória. A matriz linear do Newton-Raphson agora reutiliza as mesmas estampas. Python e Rust têm o mesmo modelo, sem conversão de tensão em corrente por uma condutância artificial.

Adicionadas 53 regressões DC: orientação de corrente, resistores entre 1 mΩ e 1 GΩ, enrolamentos flutuantes, duas correntes independentes no transformador, correntes DC estacionárias em Euler/trapézio, condições iniciais em malhas de indutores, AC sem alteração de dimensão/impedância, ponto não linear, varredura, Monte Carlo e chamadas diretas à extensão Rust. Adicionadas 13 testemunhas topológicas das sementes incompatíveis/indeterminadas. Os cinco testes nativos Rust passaram; a extensão foi reconstruída em release para Python 3.12 e instalada na `.venv`.

Os dez descartes anteriores foram reclassificados como sistemas sem ponto DC: a fonte de 1 V é fechada por um caminho de enrolamentos com tensão obrigatoriamente nula. O modelo ideal identifica mais três casos com correntes circulantes não únicas, que antes eram selecionadas pela resistência artificial. Os testes dessas 13 sementes agora exigem a rejeição correta nos três solvers. Os outros 107 circuitos aleatórios têm soluções comparadas normalmente. Não se contabiliza uma rejeição correta como solução física.

Auditoria detalhada: `AUDITORIA_DC_IDEAL.md`. Suíte completa na `.venv`: **610 aprovados, nenhum ignorado**. Memória: **164/164 comparações aprovadas**. Cinco roteiros: **19 configurações aprovadas**. Ruff e `git diff --check` passaram. As contagens e a limitação de 1 µΩ descritas nas seções anteriores registram estados históricos, anteriores a esta alteração.


O ensaio de Monte Carlo em uma malha DC de corrente indeterminada também passou a exigir erro quando todas as amostras falham. O adaptador Rust não retorna mais médias e desvios zero para uma coleção vazia de soluções; seu comportamento agora coincide com o Python.


## Complemento: enumeração de cantos no pior caso

A análise de sensibilidade agora enumera automaticamente os 2ⁿ cantos até dez parâmetros com intervalo relativo não nulo (1.024 resoluções adicionais). Acima desse limite usa os dois cenários indicados pelas derivadas nominais, identificados como `selected`. A API e a CLI permitem forçar o método e configurar o orçamento; exigir enumeração com orçamento insuficiente gera erro. O resultado registra método efetivamente usado, número de resoluções, quantidade de parâmetros variáveis e combinações dos extremos avaliados. Um domínio sem variação reutiliza o nominal.

Foram acrescentadas dez regressões: interação com derivadas nominais nulas, reprodução dos extremos pelos parâmetros registrados e preservação do circuito, heurística e fallback por orçamento, rejeição de orçamento insuficiente, polo interior com cantos finitos, rejeição de canto singular com contexto dos parâmetros, domínio degenerado, quatro opções inválidas e integração CLI. O exemplo V=(g₁−1)(g₂−1), com tolerâncias de 20% e 30%, mostra ±0,06 V nos cantos mesmo com Taylor e RSS nulos. Já V=g/(1−g) mostra por que extremos finitos nos cantos não certificam o interior: pode existir um polo. Os relatórios e a documentação deixam essa limitação explícita, sem aprovação global implícita.

Validação na `.venv` Python 3.12: **620 testes aprovados, nenhum ignorado**; **5 testes nativos Rust aprovados**; **19 configurações aprovadas nos cinco roteiros**. Esta etapa altera apenas Python, CLI e documentação; a extensão Rust instalada permanece a reconstruída na etapa anterior. Ruff e `git diff --check` passaram.
