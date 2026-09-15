# Rodada 2, registro de alterações (JCIS)

Documento de leitura do orientador. Contém apenas o registro das alterações.

## A. Correções factuais

| Subseção | Dizia | Passa a dizer | Evidência |
|---|---|---|---|
| Resumo e Introdução | "Gram--Schmidt orthonormalization" | "Modified Gram--Schmidt orthonormalization" | A Subseção III-D (3.4) especifica o algoritmo de Gram--Schmidt modificado; a descrição do estimador foi alinhada a ele. |
| II-A (sistema expandido) | "the total gain G is the most decisive scalar parameter" | "the total gain G is a relevant scalar parameter" | arXiv:2401.10281 afirma a dependência dos pontos fixos em G, não o superlativo. |
| III-A (arquitetura) | "as observed in package microbenchmarks during development" | "as measured by the package benchmarks", com citação ao pacote | benchmarks/standalone.py indica cerca de 80 a 100 vezes; docs/architecture.rst indica 20 a 50 vezes; ambos dentro de uma a duas ordens de grandeza. |
| IV-C (mapas de classificação) | regra de classificação por célula ausente | uma célula é marcada divergente quando ao menos uma das 25 condições iniciais produz estado não finito durante as 500 iterações de burn-in | tools/report_sweep_stats.py e o kernel da varredura (OR sobre o conjunto; NaN). |

## B. Alterações que afetam conclusão

| Subseção | Dizia | Passa a dizer | Evidência | O que se perde |
|---|---|---|---|---|
| IV-C (Subsec. 4.3) | explicação mecanicista única pelo ganho DC, com G próximo de 1 preservando as órbitas limitadas e G próximo de 0 levando à divergência dominante | composição medida por tipo de filtro (contagens absolutas e fração divergente); no passa-faixa, o conjunto caótico é limitado predominantemente por órbitas periódicas e não por divergência; mantém-se a dependência analítica de p+ em G | Agregação dos 124 checkpoints por tools/report_sweep_stats.py | Perde-se a narrativa causal unificadora entre os quatro tipos de filtro; permanece a descrição quantitativa. |
| V (Síntese) | "first-order screening rule" pelo ganho DC (G próximo de 1 conserva região caótica utilizável; G próximo de 0 leva à divergência) | enunciado descritivo das frações divergentes medidas | Agregação dos 124 checkpoints | Perde-se a prescrição de projeto de primeira ordem. |
| Introdução e V | literatura anterior descrita como restrita a filtros passa-baixas e, no máximo, duas janelas de projeto | a literatura anterior varia geometrias de posicionamento de zeros com janela fixa; este trabalho adota seletividade em frequência e janela de projeto como eixos primários | arXiv:2401.10281, Experimentos I a V | Perde-se a alegação de que o trabalho anterior era restrito a passa-baixas; a diferença passa a ser de eixos. |
| V (Limitações) | não mencionava o descarte de condições iniciais divergentes nem a vizinhança de amostragem | acrescenta que condições iniciais que divergem após o burn-in são descartadas da média e que a vizinhança é um cubo de aresta 0,1, contra a esfera de raio 0,01 de Borges et al. | arXiv:2401.10281 (esfera de raio 0,01); kernel da varredura | Perde-se a omissão; explicita-se o procedimento efetivo. |
| Resumo e Introdução | escopo não delimitado | o trabalho cobre ordem, frequência de corte, tipo de filtro e janela de projeto; demais dimensões do espaço de parâmetros ficam como trabalho futuro habilitado pela infraestrutura | Redação do próprio manuscrito | Perde-se a promessa de cobertura ampla do espaço de parâmetros. |
| Introdução | "a concrete operating recommendation for bandlimited CBCS" | removida | Não sustentada pelos resultados | Perde-se a promessa de recomendação operacional concreta. |
