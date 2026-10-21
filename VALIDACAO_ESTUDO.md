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
