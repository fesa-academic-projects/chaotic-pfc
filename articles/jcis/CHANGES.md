# Rodada 2, registro de alterações (JCIS)

Documento de leitura do orientador. Contém apenas o registro das alterações.

## A. Correções factuais

| Subseção | Dizia | Passa a dizer | Evidência |
|---|---|---|---|
| Resumo, Introdução e Algoritmo 1 | caminho de produção descrito como "Modified Gram--Schmidt orthonormalization" | propagação de um vetor tangente com renormalização periódica; o Gram--Schmidt modificado passa a ser mencionado apenas no kernel de espectro completo e na validação | `_kernel.py`: o caminho de produção propaga um vetor (só normalização); não há ortogonalização nesse caminho. |
| II-A (sistema expandido) | "the total gain G is the most decisive scalar parameter" | "the total gain G is a relevant scalar parameter" | arXiv:2401.10281 afirma a dependência dos pontos fixos em G, não o superlativo. |
| III-A (arquitetura) | "as observed in package microbenchmarks during development" | "as measured by the package benchmarks", com citação ao pacote | benchmarks/standalone.py indica cerca de 80 a 100 vezes; docs/architecture.rst indica 20 a 50 vezes. |
| III-F (métricas) | regra de classificação por célula ausente | célula divergente quando ao menos uma das 25 condições iniciais produz estado não finito durante as 500 iterações de burn-in; a IV-C passa a remeter a esta regra | Kernel da varredura (`_kernel.py`) e tools/report_sweep_stats.py. |
| II-D e Algoritmo 1 | divergência por $\|\mathbf{x}\|>X_{\max}$, com $X_{\max}=10^{6}$ | divergência por estado não finito (NaN ou infinito) | `_kernel.py`: a checagem é `np.isfinite`; não há limiar. |
| III-H (reprodutibilidade) | "versioned in data/sweeps/ via Git LFS" | "versioned in data/sweeps/" | Não há `.gitattributes` no repositório e `git lfs` não está instalado. |
| III-H (reprodutibilidade) | `.npz` com "classification vector" e "execution metadata (seed, code hash, timestamp)" | tensor $h$, desvio `h_desvio`, ordens/banco de coeficientes, grade de corte, rótulos, `n_iters_used` e bloco `metadata` (parâmetros da simulação); a classificação é derivada de $h$ | Chaves reais dos `.npz`: `coef`, `filter_type`, `h`, `h_desvio`, `metadata`, `n_iters_used`, `wcorte`, `window`. |
| III-D (estimador) | período de renormalização sem fonte | período de dez iterações atribuído à prática estabelecida desde Benettin et al. | DOI confirmados: 10.1007/BF02128236 (Parte 1) e 10.1007/BF02128237 (Parte 2), Meccanica 15, 1980. |
| III-A (arquitetura) | custo descrito apenas com base no kernel de base completa | custo por iteração $\mathcal{O}(K)$ no caminho de vetor único; o $\mathcal{O}(K^3)$ e a estimativa de $5\times 10^9$ declarados como do kernel de espectro completo (células marginais e validação) | `_kernel.py`. |
| IV-A (validação) | identidade satisfeita "pelo estimador" sem distinguir qual | validação feita com o estimador de espectro completo; as varreduras usaram o caminho de produção | Execução de validação e `_kernel.py`. |
| III-E (condições iniciais) | cubo de aresta $\epsilon=0.1$ "around the positive equilibrium point" (centrado) | cubo semiaberto, não centrado: cada componente é deslocada de $\epsilon\,U$ com $U\sim\mathcal{U}[0,1)$, de modo que o estado está em $[p_k^+,p_k^++0.1)$ por componente | `_kernel.py`: `xx[0,k] = 0.1*noise[k] + p_k`, com `noise ~ U[0,1)`. |
| III-H (reprodutibilidade) | testes verificam a exatidão contra "the reference value of Alligood1996" | testes verificam contra os valores publicados de Wolf et al. e Sprott | `tests/test_scientific_validation.py` compara λ1/λ2 com Wolf1985 e Sprott2003; entrada `Sprott2003` acrescentada ao `.bib` e à whitelist. |
| IV-G (reprodutibilidade) | script de figuras em `src/chaotic_pfc/sweep_plotting.py` | `src/chaotic_pfc/analysis/sweep_plotting.py` | O arquivo existe em `src/chaotic_pfc/analysis/`. |
| IV-G (custo) | $2.8\times10^{10}$ iterações e $3.5\times10^{11}$ operações Jacobiana-base | soma dos horizontes efetivos por célula, cerca de $2\times10^{8}$ (da ordem de $5\times10^{9}$ iterações do mapa contando as 25 condições iniciais por célula); contagem de operações Jacobiana-base removida | Soma de `n_iters_used` nos 124 `.npz` = 2,03×10^8; `n_iters_used` é a média por célula sobre as condições iniciais não divergentes (kernel, `iters_total/count`). |
| IV-G (hardware/tempo) | i7 de 8ª geração, 6 núcleos/12 threads, 16 GB, cerca de 6 h | Intel Core i7-8550U, 4 núcleos/8 threads, 32 GB; conjunto das 124 varreduras em 489 s (cerca de 8 min), medido de ponta a ponta | `lscpu` (4C/8T), `benchmarks/baseline/README.md` (32 GB), medição direta. |
| III-B (filtros) | não mencionava coeficientes de filtro não finitos | acrescenta que, nas janelas Hann e Bartlett com filtros passa-baixas e passa-faixa, o projetista devolve coeficientes não finitos em parte da grade, e essas células são classificadas como divergentes por falha de projeto do filtro, não por dinâmica do mapa | `firwin` emite um RuntimeWarning de divisão; 400 células de 372.000 (0,11%) têm coeficientes não finitos. |
| Resumo | "an ensemble of initial conditions centered at the equilibrium point" | "anchored at the equilibrium point", sem afirmar centragem | A III-E já descreve um cubo semiaberto não centrado. |
| Resumo | "the protocol is validated against the standard map" | "validated against the standard H\'enon map" | "standard map" é o mapa padrão de Chirikov; a validação é contra o Hénon canônico (IV-A). |
| III-D | a parada antecipada reduz o custo "without compromising accuracy" | "while preserving the adopted classification criterion" | Não há medição de exatidão no artigo que sustente "without compromising accuracy". |
| III-F (métricas) | regra de divergência espalhada em três formas ao longo do texto | as três etapas reunidas na III-F: (i) valor não finito na acumulação por condição inicial; (ii) célula divergente se ao menos uma das 25 condições diverge no burn-in; (iii) condições que divergem após o burn-in são descartadas da média | `_kernel.py`: checagem `np.isfinite` na acumulação; OR sobre o conjunto no burn-in; descarte por `np.isnan` na média. |

## B. Alterações que afetam conclusão

| Subseção | Dizia | Passa a dizer | Evidência | O que se perde |
|---|---|---|---|---|
| Equação (4) | entrada (3,2) da Jacobiana como $\beta c_0$ | $c_0\beta + c_2$ | Diferenças finitas contra o mapa do pacote (erro ~2e-13); Apêndice A de arXiv:2401.10281, eq. (A.3), (A.7), (A.9) | Nada; a matriz passa a coincidir com o código e com a fonte. |
| IV-C (Subsec. 4.3) | explicação mecanicista única pelo ganho DC, com G próximo de 1 preservando as órbitas limitadas e G próximo de 0 levando à divergência dominante | composição medida por tipo de filtro (contagens absolutas e fração divergente); no passa-faixa, o conjunto caótico é limitado predominantemente por órbitas periódicas e não por divergência; mantém-se a dependência analítica de p+ em G | Agregação dos 124 checkpoints por tools/report_sweep_stats.py | Perde-se a narrativa causal unificadora entre os quatro tipos de filtro; permanece a descrição quantitativa. |
| IV-B (Subsec. 4.2) | "consistent with the DC-gain argument developed in Subsection IV-C", com a explicação causal | remoção da causa e da remissão; mantida a observação factual de que o rejeita-faixa rejeita banda estreita afastada de $\omega=0$ e tem G próximo da unidade | Agregação dos 124 checkpoints | Perde-se o elo causal entre G e a cobertura observada. |
| V (Síntese) | "first-order screening rule" pelo ganho DC | enunciado descritivo das frações divergentes medidas | Agregação dos 124 checkpoints | Perde-se a prescrição de projeto de primeira ordem. |
| Introdução e V | literatura anterior descrita como restrita a filtros passa-baixas e, no máximo, duas janelas de projeto | a literatura anterior varia geometrias de posicionamento de zeros com janela fixa; este trabalho adota seletividade em frequência e janela de projeto como eixos primários | arXiv:2401.10281, Experimentos I a V | Perde-se a alegação de que o trabalho anterior era restrito a passa-baixas; a diferença passa a ser de eixos. |
| III-D e Algoritmo 1 | reortonormalização de Gram--Schmidt modificado a cada iteração sobre a base completa | estimador de vetor único, com reortonormalização em blocos de 10 iterações e recálculo de espectro completo nas células marginais; parada adaptativa | `_kernel.py` e `_types.py` (bloco 10, margem 5e-3) | Perde-se a descrição do método como estava; o texto passa a descrever o procedimento que gerou os resultados publicados. |
| V (Limitações) | a rodada anterior acrescentou que condições iniciais divergentes são descartadas da média e que a vizinhança seria um cubo de aresta 0,1 "em contraste com a esfera de raio 0,01" de Borges et al. | mantém-se o descarte de condições iniciais divergentes; a comparação com a esfera de raio 0,01 foi retirada ao se verificar que a implementação de referência usa a mesma amostragem | Implementação de referência e `_kernel.py` | Perde-se a comparação com o texto de referência; a limitação sobre o descarte permanece. |
| Resumo e Introdução | escopo não delimitado | o trabalho cobre ordem, frequência de corte, tipo de filtro e janela de projeto; demais dimensões do espaço de parâmetros ficam como trabalho futuro habilitado pela infraestrutura | Redação do próprio manuscrito | Perde-se a promessa de cobertura ampla do espaço de parâmetros. |
| Introdução | "a concrete operating recommendation for bandlimited CBCS" | removida | Não sustentada pelos resultados | Perde-se a promessa de recomendação operacional concreta. |
| IV-C (Subsec. 4.3) | "values of $G$ close to zero shift $\mathbf{p}^+$ toward regions in which the term $\alpha - x_3^2$ tends to diverge, which in our maps appears as the gray (divergent) regions that dominate high orders" | removida; o parágrafo mantém a dependência analítica de p+ em G, as cascatas de duplicação de período e os shrimps | Contradiz as frações medidas na própria IV-C (G próximo de 0: 6,77% e 10,32%; G próximo de 1: 68,87% e 92,53%) | Perde-se a leitura causal da região cinza; a contagem "two observations" permanece válida. |
| II-B (justificação em frequência) | argumento apenas em prosa | acrescenta a figura do canal ideal (`fig:comm_ideal`, em inglês) e dois parágrafos com os três cenários, com BER de 0, 0,4324 e 0 e métricas calculadas após compensar o atraso de grupo de 100 amostras do canal de 201 taps | `figures/comm_ideal.svg` e as CLIs de comunicação; `articles/jcis/Figures/fig_comm_ideal.svg` | Nada; a afirmação de precondição passa a ter base quantitativa numa configuração específica. |

v6: reescrita após a revisão do orientador.

## Notas de leitura (histórico)

Os caminhos citados nas tabelas acima referem-se ao estado do repositório na
época de cada rodada (por exemplo, `tools/report_sweep_stats.py` foi depois
substituído por `chaotic-pfc sweeps stats`, e `src/chaotic_pfc/sweep_plotting.py`
por `src/chaotic_pfc/analysis/sweep_plotting/`); permanecem como registro
histórico.

O parágrafo da IV-C sobre o ganho DC (composição medida por tipo de filtro, com
a dependência analítica de $p^+$ em $G$) foi acrescentado nesta rodada e é
mantido como descrito, sem a narrativa causal única por $G$.
