# Plano do artigo do PFC

Este plano define o argumento, a ordem, as evidências e a frase-tópico de cada parágrafo.
O texto segue o plano. Qualquer desvio (evidência que não existe, parágrafo que não cabe,
número sem label) é reportado, não improvisado.

## 0. Tese e título

**Tese (uma frase).** Verificar se um gerador caótico filtrado continua caótico exige uma
cadeia longa de cálculo; este trabalho especifica os requisitos dessa cadeia, constrói uma
ferramenta que os atende, avalia a ferramenta contra eles e a usa para projetar um enlace
de comunicação em canal de banda limitada.

**Título sugerido.** Verificação confiável do regime caótico em geradores de Hénon
filtrados: requisitos, arquitetura e aplicação em comunicação.

**Leitor-alvo.** Professor de computação que não é da área de sinais e sistemas dinâmicos.
Cada conceito de dinâmica chega com uma frase de intuição antes da definição.

## 1. Fio condutor

Cada seção responde a uma pergunta e deixa a seguinte. As considerações finais respondem
explicitamente às perguntas 2 e 11.

| # | Pergunta do leitor | Onde é respondida |
|---|---|---|
| 1 | Por que filtrar um gerador caótico? | 1 Introdução |
| 2 | O filtro mantém o caos? | 1 Introdução (não se sabe a priori) |
| 3 | O que verificar o caos exige? | 2 Fundamentação |
| 4 | O que uma ferramenta de verificação precisa garantir? | 3.1 Requisitos |
| 5 | Como ela foi construída? | 3.2 a 3.5 |
| 6 | Por que ela é assim? | 3.2 Quadro de decisões de projeto |
| 7 | Ela calcula certo? | 4.1 |
| 8 | É rápida e determinística? | 4.2 |
| 9 | O que ela revela? | 4.3 |
| 10 | Serve para projetar um enlace? | 4.4 Estudo de caso |
| 11 | Os requisitos foram atendidos? O que falta? | 4.5 e 4.6 |

## 2. Elementos gráficos (lista final)

| Id | Tipo | Seção | Conteúdo | Origem |
|---|---|---|---|---|
| Fig. 1 | dados | 2.1 | Sinais do Hénon: caótico (α = 1,4) e periódico (α = 0,9) | fig_henon_signals.pdf (existe) |
| Fig. 2 | TikZ | 2.4 | Gerador filtrado e enlace mestre-escravo, mesmo filtro nas duas pontas, canal ideal ou FIR | Fig. 2 atual, sem nomes de função |
| Fig. 3 | TikZ | 3.2 | Arquitetura em componentes (configuração, dinâmica, comunicação, análise, manuscrito, interface de linha de comando) com o fluxo de dados entre eles; só nomes de componentes, nenhum nome de função | redesenhar |
| Fig. 4 | TikZ | 3.3 | Fluxo do estimador e da classificação | Fig. 3 atual, sem nomes de função |
| Fig. 5 | dados | 4.3 | Mapas de classificação: passa-baixas Kaiser βK = 17 e passa-altas Kaiser βK = 15,5 | existem |
| Fig. 6 | dados | 4.4 | Mapa do passa-baixas Hamming com o ponto do estudo de caso marcado (Nz = 8, ωc/π = 0,52) | GERAR (ver seção 7) |
| Fig. 7 | dados | 4.4 | Enlace pelo canal ideal | fig_comm_ideal.pdf (existe) |
| Fig. 8 | dados | 4.4 | Enlace pelo canal FIR, sem e com filtro na malha | fig_comm_bandlimited.pdf (existe) |
| Tab. 1 | tabela | 3.1 | Requisitos e critérios de aceitação | nova |
| Quadro 1 | quadro | 3.2 | Decisões de projeto: decisão, alternativa, motivo, evidência | GATE L, L1 |
| Quadro 2 | quadro | 3.5 | Verificações da CI: verificação, o que barra, obrigatória | Quadro 1 atual, condensado |
| Tab. 2 | tabela | 4.2 | Desempenho e determinismo | Tabela 1 atual |
| Tab. 3 | tabela | 4.3 | Melhor configuração por tipo de filtro | Tabela 2 atual |
| Tab. 4 | tabela | 4.5 | Verificação dos requisitos: requisito, evidência, resultado, situação | nova |

Saem do artigo: o quadro de correspondência entre diagramas e código, a figura de densidade
agregada, a figura da Kaiser e a figura de espectro.

## 3. Requisitos (conteúdo da Tabela 1)

Os critérios são verificáveis e não inventam metas numéricas que não existiam.

| Id | Requisito | Critério de aceitação |
|---|---|---|
| RF1 | Classificar cada ponto de projeto (tipo de filtro, janela, ordem, frequência de corte) como caótico, não caótico ou divergente | Classe atribuída a todos os pontos da grade, pelo critério da Seção 2.3 |
| RF2 | Produzir mapas e estatísticas de cobertura caótica | Mapas e tabelas gerados a partir dos dados gravados |
| RF3 | Simular o enlace mestre-escravo com canal ideal e canal FIR e medir BER e MSE | Métricas calculadas para os dois canais |
| RNF1 | Correção do estimador | Reproduz o valor de referência da literatura e a identidade analítica do mapa sem filtro |
| RNF2 | Tempo de execução compatível com um computador pessoal | Grade completa executada e tempo medido e registrado |
| RNF3 | Determinismo | Mesmos resultados para a mesma semente, independentemente do número de threads |
| RNF4 | Rastreabilidade dos números | Todo número do texto calculado dos dados gravados e conferido automaticamente |
| RNF5 | Falha segura | Dado corrompido ou verificação quebrada interrompe o fluxo do manuscrito |
| RNF6 | Integridade dos dados | Nenhum arquivo de dados executa código ao ser aberto |
| RNF7 | Validação de entradas | Parâmetro inválido gera exceção de domínio explícita |
| RNF8 | Segurança da cadeia de suprimentos | Mudança só entra com as verificações obrigatórias aprovadas |

## 4. Matriz de evidências

Toda afirmação de resultado usa um label do numbers.py ou uma entrada do new_numbers.json.

| Afirmação | Evidência |
|---|---|
| Estimador reproduz referência e identidade | VAL.ns1.full_spectrum, VAL.ns1.single_vector, VAL.ns1.full_vs_ref, VAL.ns1.single_vs_full, VAL.ns1.single_vs_ref, VAL.ns1 (resíduo máximo) |
| Ganho JIT, paralelismo, tempo total | benchmarks/baseline/pfc_bench_i7_8550u.json; sweeps_184_i7_8550u.json |
| Determinismo entre 1 e 8 threads | class_sha256_numba1 = class_sha256_numba (pfc_bench); TestThreadDeterminism |
| Grade: 184 configurações, 552.000 pontos | GRID.* (configuração, não resultado; pode ir na Seção 3) |
| Ganho DC | GAIN.n_absG_lt_0.1, GAIN.chaotic_count_absG_lt_0.1, GAIN.lowpass.*_pct, TOT.bandstop.pct_divergent, GAIN.highpass.chaotic_absG_lt_0.1, PSD.bandpass.point_gain |
| Melhor configuração por tipo | T2.*.1.Nc, T2.*.1.rho_abs |
| Robustez da classificação | LH.* (horizonte longo), SEED.* (dez sementes) |
| Estudo de caso | λ1 do projeto 0,0786 e ponto vizinho da grade (ωc/π = 0,5152, λ1 = 0,0788, caótico); BER e MSE dos canais (fig_comm_*) |
| Falha segura | TestAuditFailClosed; test_load_all_rejects_corrupt_checkpoint |
| Integridade | test_malicious_pickled_npz_is_refused_without_executing_payload; test_all_184_versioned_checkpoints_load |
| Validação de entradas | tests/test_validation.py; tests/test_contract.py; job contract |
| Cadeia de suprimentos | ruleset (obrigatórias), jobs gitleaks, zizmor, Semgrep, CodeQL |

## 5. Plano por seção

Formato: **Pn** frase-tópico (o que o parágrafo afirma) / evidência ou referência / tamanho.
Nenhum nome de função, arquivo ou job no corpo do texto.

### Resumo e abstract
Escritos por último. Ordem: problema (filtro pode destruir o caos), objetivo (ferramenta
confiável de verificação), solução (requisitos, arquitetura, decisões), avaliação (os três
números de desempenho, determinismo, falha segura), resultado científico (ganho DC),
estudo de caso (BER), limitações. 200 a 240 palavras.

### 1 Introdução (escrita pelos autores, cerca de 1,5 página)
- **P1** Sinais caóticos são candidatos a portadoras por terem espectro largo, aparência de
  ruído e sensibilidade às condições iniciais. / Baptista 2021; Kaddoum 2016; Pecora 1990.
- **P2** Canais reais têm banda limitada, e inserir um filtro na realimentação do gerador
  confina o espectro transmitido. / Eisencraft 2009; Fontes 2016.
- **P3** O filtro preserva a sincronização, mas pode destruir o caos do gerador, e isso
  muda com cada escolha de tipo, janela, ordem e corte. / Borges 2026; Candido 2015;
  Borges e Eisencraft 2022a, 2022b; Borges, Silva e Eisencraft 2023, 2024; Silva et al. 2026a.
- **P4** Verificar o caos de cada projeto exige uma cadeia longa de cálculo, e um erro
  silencioso em qualquer etapa produz uma conclusão errada com aparência correta; daí a
  pergunta do trabalho: como construir, validar e usar uma ferramenta confiável para essa
  verificação. / sem números de resultado.
- **P5** Contribuições: (i) requisitos e arquitetura de uma ferramenta de verificação, com
  decisões de projeto justificadas; (ii) avaliação da ferramenta contra os requisitos;
  (iii) estudo de caso em que um filtro é escolhido no mapa, verificado e usado num enlace
  de banda limitada.
- **P6** Organização do texto, em três linhas.

### 2 Fundamentação (cerca de 3 páginas)

**2.1 Caos e o mapa de Hénon**
- **P1** Intuição de sensibilidade às condições iniciais e o que distingue caos de
  comportamento periódico. / Alligood 1996.
- **P2** O mapa de Hénon, equação (1), e os parâmetros usados; o atrator estranho em
  (1,4; 0,3) e a estrutura do espaço de parâmetros. / Hénon 1976; Gallas 1993.
- **P3** A Fig. 1 mostra a diferença entre as duas classes, que é a distinção que a
  ferramenta precisa automatizar. / Fig. 1.

**2.2 Filtro FIR na malha e ganho DC**
- **P1** O filtro FIR, a saída filtrada que substitui x1 no termo não linear, e o aumento
  da dimensão do sistema com a ordem. / equações do filtro e do mapa filtrado.
- **P2** O ponto fixo do mapa filtrado depende dos coeficientes só pelo ganho DC, o que
  faz de G uma variável de projeto. / Borges, Silva e Eisencraft 2024. Sem números.
- **P3** Os coeficientes vêm do método da janela, cuja forma troca seletividade por
  atenuação; nas ordens estudadas, o filtro realizado não reproduz a máscara nominal.
  / Oppenheim 2009; Harris 1978; Kaiser 1980.

**2.3 Expoente de Lyapunov e classificação**
- **P1** Intuição (taxa média de afastamento de órbitas próximas) e definição do maior
  expoente. / equação; Benettin 1980a,b; Wolf 1985.
- **P2** Na prática usa-se uma estimativa de tempo finito e um limiar θ = 5 × 10⁻³, com
  três classes; o limiar é uma decisão operacional e a classificação é local no espaço de
  estados. / Sprott 2003; Silva et al. 2026a.

**2.4 Enlace mestre-escravo**
- **P1** Transmissor e receptor com o mesmo gerador; a mensagem modula a portadora e o
  receptor a recupera depois da sincronização; canal ideal ou FIR; métricas BER e MSE.
  / Pecora 1990; Fig. 2.
- **P2** Com o mesmo filtro nas duas pontas, a sincronização se mantém para qualquer filtro
  BIBO estável, mas o caos do gerador não é garantido; por isso o gerador é verificado
  antes do enlace. / Borges 2026; Candido 2015; Fontes 2017; Fontes e Eisencraft 2024.

**2.5 Confiabilidade de software científico**
- **P1** Cadeias numéricas longas falham de modos silenciosos: NaN que se propaga,
  resultado que muda com o número de threads, dado corrompido, desserialização que executa
  código, dependência comprometida. / sem autorreferência.
- **P2** Técnicas que mitigam esses modos: compilação JIT para tornar viável a grade,
  testes baseados em propriedades, comparação com identidades analíticas, verificação de
  integridade. / Lam 2015; Harris 2020; Virtanen 2020; MacIver 2019.

### 3 Requisitos e solução (cerca de 4,5 páginas; nenhum número de resultado)

**3.1 Requisitos**
- **P1** Da pergunta 4: o que a ferramenta precisa garantir para que a conclusão sobre um
  projeto seja confiável; Tabela 1 com requisitos funcionais e não funcionais e critérios
  de aceitação. / Seção 3 deste plano.

**3.2 Arquitetura e decisões de projeto**
- **P1** A ferramenta é organizada em componentes com responsabilidades separadas, e o
  fluxo de dados vai da configuração ao texto sem estado global. / Fig. 3.
- **P2** Cada componente atende a requisitos específicos (por exemplo, a persistência
  atende RNF6, a análise e o manuscrito atendem RNF4). / Tabela 1.
- **P3** As decisões de projeto e as alternativas consideradas estão no Quadro 1. Usar
  apenas motivos documentados; onde o motivo de rejeitar uma alternativa não estiver
  documentado, a linha descreve o motivo da escolha, não o da rejeição. / GATE L, L1
  (decisões 1 a 7).

**3.3 Núcleo de cálculo**
- **P1** A grade de projetos: quatro tipos, seis janelas fixas e a Kaiser com 40 valores
  de βK, ordens e cortes; 184 configurações e 552.000 pontos. / GRID.*.
- **P2** O estimador: condições iniciais perto do ponto fixo, descarte do transitório,
  vetor tangente único renormalizado, parada antecipada e recomputação com espectro
  completo perto do limiar; a Fig. 4 resume o fluxo. / Fig. 4.
- **P3** Execução paralela determinística: as perturbações das condições iniciais são
  geradas antes do núcleo paralelo e a ordem das tarefas é intercalada; isso atende RNF3 e
  RNF2 por construção. / decisões 1 e 7.

**3.4 Persistência e integridade**
- **P1** Cada varredura é gravada com os resultados e os metadados que a reproduzem.
- **P2** Os arquivos não executam código ao serem abertos, e arquivos antigos só são lidos
  depois de verificação de integridade contra referências compiladas; isso atende RNF6.
  / decisão 3.

**3.5 Verificação e cadeia de suprimentos**
- **P1** Tipos de teste e o que cada um pega: contratos (RNF7), propriedades, identidades
  analíticas (RNF1), regressão bit a bit (RNF3).
- **P2** O caminho do manuscrito falha de forma segura e os números do texto são
  calculados e conferidos; isso atende RNF4 e RNF5. / decisões 4 e 5.
- **P3** A integração contínua separa verificações obrigatórias, que bloqueiam o merge, de
  verificações que reportam; Quadro 2. / decisão 6 (ver seção 7 sobre o pip-audit).

### 4 Avaliação (cerca de 6 páginas; cada número de resultado aparece só aqui)

**4.1 Correção do estimador (RNF1)**
- **P1** No mapa sem filtro, o estimador reproduz a referência da literatura e a identidade
  analítica, o que o valida por dois caminhos independentes. / VAL.*.

**4.2 Desempenho e determinismo (RNF2, RNF3)**
- **P1** Tabela 2: ganho da compilação, ganho do paralelismo e tempo da grade completa,
  com máquina e protocolo. / pfc_bench, sweeps_184.
- **P2** O ganho vem do núcleo compilado; as demais operações não passam por ele e por
  isso ficam próximas de 1. / pfc_bench.
- **P3** As classes dos pontos são idênticas com uma e com oito threads. / hashes.

**4.3 O que a ferramenta revela (RF1, RF2)**
- **P1** Os mapas da Fig. 5 mostram onde o gerador permanece caótico no passa-baixas e no
  passa-altas.
- **P2** O ganho DC explica a diferença: com G próximo de zero o ponto fixo é estável e
  nenhum ponto com |G| < 0,1 é caótico; com G = 1 a classe depende de ordem, corte e janela.
  / GAIN.*, TOT.bandstop.pct_divergent, PSD.bandpass.point_gain. Única aparição completa.
- **P3** Tabela 3: a melhor configuração de cada tipo. / T2.*.
- **P4** A classificação resiste a sementes e limiares, mas parte dos pontos caóticos do
  rejeita-faixa e do passa-altas é transiente. / SEED.*, LH.*.

**4.4 Estudo de caso: projeto de um enlace em canal de banda limitada (RF3)**
- **P1** O requisito do enlace: um canal FIR que remove as frequências mais altas.
- **P2** A escolha do filtro no mapa (Fig. 6): o projeto Hamming com Nz = 8 e
  ωc/π = 0,52 fica na região caótica; o ponto vizinho da grade é caótico e o expoente do
  projeto, calculado pela mesma ferramenta, confirma.
- **P3** Controle pelo canal ideal (Fig. 7): mensagem recuperada sem erro.
- **P4** Canal FIR (Fig. 8): sem filtro na malha a mensagem não é recuperada; com o filtro
  verificado ela é recuperada sem erro.
- **P5** Alcance do resultado: canal linear e conhecido, sem ruído; demonstração de
  princípio.

**4.5 Verificação dos requisitos**
- **P1** Tabela 4: cada requisito com a evidência, o resultado e a situação.
- **P2** O que está atendido parcialmente e por quê (por exemplo, desempenho medido numa
  única máquina; verificação de vulnerabilidades que reporta sem bloquear).

**4.6 Limitações**
- **P1** Limitações científicas: classificação local, limiar sobre tempo finito,
  transientes, máscara nominal não realizada, grade limitada.
- **P2** Limitações de engenharia e de escopo: uma máquina, canal sem ruído, nenhuma
  garantia criptográfica do esquema.

### 5 Considerações finais (escritas pelos autores, cerca de 1 página)
- **P1** Resposta à pergunta 2: o filtro pode destruir o caos, e o ganho DC decide se ele
  é possível na grade estudada.
- **P2** Resposta à pergunta 11: a ferramenta atende aos requisitos, com as ressalvas da
  Tabela 4.
- **P3** As três contribuições, retomadas.
- **P4** Trabalhos futuros: canais com ruído, filtros IIR, outras larguras de banda e
  parâmetros, implementação em hardware.

## 6. Orçamento

Cerca de 18 páginas com referências. Se passar de 20, cortar primeiro 2.5 P2 e 4.3 P4.

## 7. Pendências antes da Fase R (na main)

1. Documentar em docs/security.rst por que o pip-audit é opcional (uma vulnerabilidade nova
   numa dependência pode bloquear uma mudança sem relação com ela).
2. Conciliar a complexidade do espectro completo: o JCIS diz O(K³) por iteração e a
   documentação diz O(Ns²). Corrigir a documentação ou o texto conforme o código.
3. Gerar a Fig. 6: mapa do passa-baixas Hamming com o ponto (Nz = 8, ωc/π = 0,52) marcado,
   no mesmo estilo dos mapas do JCIS, em PDF para o PFC.
