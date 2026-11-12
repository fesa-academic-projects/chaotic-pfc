# Resposta v4 → v5

Base do diff: tag `pfc-v4-orientador`, commit `569ad35` (25 páginas). O diff
está em `diff-v4-v5.pdf`. O limite passou de 25 para 30 páginas.

## Correções do orientador (24 anotações)

| Item | Como ficou | Observação |
|---|---|---|
| 1 | Resumo e abstract com “formando uma grade”, “cada combinação (ponto da grade)” e “para cada filtro projetado”. | Como sugerido. |
| 2 | ω definido como frequência angular normalizada em radianos por amostra, com π na metade da frequência de amostragem e as figuras divididas por π. | Como sugerido. |
| 3 | A frase do ganho DC passou para o fim do parágrafo dos quatro tipos, com a explicação de por que o passa-baixas e o rejeita-faixa têm G = 1. | Como sugerido. |
| 4 | xj(n) guarda a amostra de x1 de j − 2 instantes atrás. | Como sugerido. |
| 5 | A equação do ponto fixo agora mostra o passo p = α − G²p² + βp. | Como sugerido. |
| 6 | J_H(n) para a jacobiana do mapa (1) e J(n) para a do mapa filtrado, inclusive em det J_H(n) = −β. | Ajuste de símbolo. |
| 7 | A repetição de “a cada iteração” saiu; a reortogonalização evita estouros numéricos e os passos vêm em seguida. | Como sugerido. |
| 8 | s(n) e μ são definidos antes das equações do transmissor e do receptor. | Como sugerido. |
| 9 | Acrescentada a remissão “(equação (16), Seção 3.6)” para a matriz do erro com filtro. | Como sugerido. |
| 10 | Semente explicada na primeira menção e usada sem nova definição depois. | Como sugerido. |
| 11 | (14) usa λ̄1, com “λ̄1 é a estimativa de λ1, média sobre as condições iniciais”; λ̄1 também em 4.1, 4.4, 4.5 e Figura 6. | Ajuste de símbolo. |
| 12 | “Delineamento” no lugar de “desenho” e SS explicado como sum of squares. | Como sugerido. |
| 13 | A explicação da paridade das ordens veio antes da contagem de pontos. | Como sugerido. |
| 14 | O corpo do texto cita a obra, não o DOI; “DOI de conceito” saiu. | Ajuste pela mensagem. |
| 15 | “Os pontos com 0 < λ̄1 < 10⁻², isto é, a menos de θ do limiar…”. | Como sugerido. |
| 16 | Gram-Schmidt clássico e modificado com referência a Trefethen e Bau (1997) e explicação da variante modificada. | Ajuste pela mensagem. |
| 17 | Testes de contrato explicados. | Como sugerido. |
| 18 | Execução sem compilador e commits assinados explicados entre parênteses. | Como sugerido. |
| 19 | “A matriz da recursão (16) reproduz a jacobiana…”. | Como sugerido. |
| 20 | O cenário de controle é apresentado como forma de separar atraso e banda. | Como sugerido. |
| 21 | λ̄1 = 0,4185 está 0,17% abaixo de 0,41922; λ̄1 = 0,4194 está 0,04% acima da referência e 0,21% do espectro completo. | Conta refeita com os valores não arredondados. |
| 22 | η²p explícito para 0,588, 0,759, 0,495 e 0,387, com a leitura 58,8% e 75,9%. | Como sugerido. |
| 23 | Teste de horizonte longo com 3 × 10⁵ iterações e transitórios caóticos definidos. | Como sugerido. |
| 24 | O título do SBrT foi mantido como “exponente”. | Ver evidência abaixo. |

### Item 21 — conta refeita

- (0,41922 − 0,4185)/0,41922 = 0,001717 → **0,17%**.
- (0,4194 − 0,41922)/0,41922 = 0,000429 → **0,04%**.
- (0,4194 − 0,4185)/0,4185 = 0,002150 → **0,21%**.

### Item 24 — título do SBrT

O PDF do próprio artigo, na primeira página, imprime “Um estudo numérico do
**exponente** de Lyapunov do Mapa Hénon com filtro FIR”. O catálogo do SBrT e o
DOI também trazem “exponente”. A referência foi mantida exatamente como
publicada.

### Gram-Schmidt

TREFETHEN, L. N.; BAU, D. **Numerical linear algebra**. Philadelphia: SIAM,
1997. ISBN 978-0-89871-361-9. A obra trata do Gram-Schmidt clássico e do
modificado (Lecture 8).

### 3.3 sem o DOI

> O código e os dados estão disponíveis no repositório da ferramenta
> (Pereira et al., 2026a).

O DOI permanece apenas na lista de referências.

## Restaurado com o novo limite de páginas

- Figuras 8 e 13 de volta ao aspecto 0,68, com painéis confortáveis.
- Figura 2 de volta a 3,05 in de altura.
- Figura 11 de volta ao aspecto 0,60.
- Figura 9 com painéis próximos de quadrados (aspecto 0,67).
- Figuras 4 e 7 com a cópia do mapa em y = −1,75 e μ = 1,35.
- Bibliografia com todos os autores (sem `maxbibnames=3`); citações seguem com
  “et al.”.
- 3.5 com a lista completa das 9 verificações restantes do CI.
- Figuras 8 e 13: as margens superior e inferior do layout foram reduzidas
  (top 0,80 → 0,86; bottom 0,14 → 0,09), o que passou a altura para os painéis
  sem mudar a altura total da figura; a fonte ficou logo abaixo do desenho.

## Tabela do item 13 — cortes por espaço

| Trecho | Commit | Motivo do corte | Volta? |
|---|---|---|---|
| Figuras 8 e 13, aspecto 0,68 → 0,48 | 266b289 | caber em 25 páginas | sim, 0,68 |
| Figura 2, altura 3,05 → 2,55 in | 266b289 | caber em 25 páginas | sim, 3,05 in |
| Figura 11, aspecto 0,60 → 0,50 | 266b289 | caber em 25 páginas | sim, 0,60 |
| Figura 9, aspecto 0,67 → 0,36 | 2aef4e1 | caber em 25 páginas | sim, 0,67 |
| Figura 8, altura reduzida | f162740 | caber em 25 páginas | sim, 0,68 |
| Figuras 4 e 7, y = −1,75 → −1,55 e μ = 1,35 → 1,2 | 9fcd6db | caber em 25 páginas | sim, −1,75 e 1,35 |
| Lista das 9 verificações do CI | c46d491 | caber em 25 páginas | sim, lista completa |
| `maxbibnames=3` | 32d758d | formato do template | sim, todos os autores |
| Frases enxugadas na fluidez | ee89cf9 | redundância, não espaço | não |
| Largura da Figura 12 (0,60 → 0,70) | rodada anterior | legibilidade e espaço | mantido (não é corte de conteúdo) |

## Páginas

- **27 páginas** (limite 30)
- Última página: **13 linhas, 27 livres**
- Nenhuma figura foi reduzida para paginar nesta rodada.

## Ajustes adicionais no padrão das correções

- **Numba (3.3):** “compilado com Numba (Lam; Pitrou; Seibert, 2015), isto é,
  traduzido para código de máquina na primeira execução, o que torna os laços em
  Python muito mais rápidos”. A 1.4 mantém “compilada”, sem glosa.
- **Hipercubo (3.3):** “25 condições iniciais, cada coordenada entre a coordenada
  correspondente do ponto fixo positivo e esse valor mais 0,1”. Conferido no
  código: `0.1 * U[0,1) + p`.
- **Invariantes (3.4):** “geram entradas aleatórias e verificam invariantes,
  propriedades que devem valer para qualquer entrada válida”.
- **Threads (3.3):** glosa na primeira menção, “fluxos de execução que rodam ao
  mesmo tempo em núcleos diferentes do processador”; depois, sem glosa.
- **Estouros numéricos (2.5):** “valores grandes ou pequenos demais para a
  representação em ponto flutuante”.
- **Mediana (4.4 e 4.6):** “a mediana, sobre os pontos da grade, do número de
  iterações acumuladas cai de 3.000 para 708”.
- **Fronteira (3.5):** “na entrada de cada função pública” e “na entrada do
  módulo”; os demais usos de fronteira são científico (borda da região caótica).
- **Contagens:** a suíte tem **1.527 testes** (`pytest --collect-only`); o texto e
  a auditoria passaram de 1.517 para 1.527. O CI continua com **23 verificações**,
  sendo **14** no Quadro 1 e **9** nas demais.
- **4.6:** a frase da mediana virou uma remissão curta para a Seção 4.4, sem
  repetir 3.992/4.000.

## Pente fino final (16 itens)

1. **3.4:** “cobertura de código (a fração do código executada pelos testes) de
   no mínimo 75%”.
2. **4.2:** “abaixo de εs” e “acima de εs”; os demais “limiar” passaram a θ ou εs.
3. **Figura 6:** o losango passou a “0 < λ̄1 < 10⁻²”.
4. **4.5:** “o λ̄1 do filtro escolhido, com ωc/π = 0,52, é 0,0786”.
5. **2.2:** “as órbitas se separam”.
6. **4.3:** “Cada filtro projetado deve ser verificado individualmente”.
7. **DPS:** a glosa passou para a legenda da Figura 8; a Figura 14 ficou sem glosa.
8. **3.5:** exceções de domínio com “ordem negativa ou corte fora da faixa”.
9. **3.1:** “(o suporte comum aos quatro tipos, Seção 3.2)”.
10. **3.1:** “SS (do inglês sum of squares) indica soma de quadrados: SSfator é a
    soma entre níveis do fator…”.
11. **3.4:** “uma verificação que lança exceção”.
12. **3.6:** “Usam-se µ = 0,01 e 10⁶ amostras; a mensagem é um texto codificado em
    bits, com m(n) = ±1 e 20 amostras por bit.”
13. **4.3:** “88,9% dos pontos do passa-baixas com Nz > 20”.
14. **Figura 12:** saiu a frase sobre passa-faixa e rejeita-faixa.
15. **1.2:** Fontes e Eisencraft (2016) mostram que a sincronização se mantém com
    o filtro na malha; Borges (2026) mostra que, com filtros FIR, ela é
    preservada para quaisquer coeficientes.
16. **2.6:** saiu a frase que repetia as equações (10) a (13); a estimativa de
    m̂(n) continua.

### Item 15 — o que cada fonte diz

- **Fontes e Eisencraft (2016):** sistema de comunicação caótico de banda
  limitada com filtro na malha de realimentação; a sincronização se mantém.
- **Borges (2026):** com filtros FIR, o bloco de x1 e x2 da matriz do erro mantém
  os autovalores ±√β e os demais são nulos, de modo que a sincronização é
  preservada para quaisquer coeficientes.

## Revisão de fluidez

Foram aplicadas as nove reescritas pedidas e, no restante do texto, corrigidos
apenas os padrões de ponto e vírgula, cadeias de “e”, “isto é” e frases longas.
Nenhum número, termo do glossário, citação ou trecho do orientador foi alterado.

- **1.1:** primeira frase dividida em duas, com “desses sinais” e a propriedade
  nomeada no fim.
- **1.2:** Eisencraft, Fanganiello e Baccalá (2009) em uma frase; Fontes e
  Eisencraft (2016) e Borges (2026) em outra.
- **1.2:** a frase dos trabalhos anteriores virou duas, separando os filtros de
  dois coeficientes dos passa-baixas e dos filtros com mais coeficientes.
- **2.5:** a definição de órbita limitada e a de sinal caótico viraram duas
  frases.
- **3.2:** a definição de ωc para os quatro tipos virou duas frases, com o
  símbolo ωc/π mantido nas figuras.
- **3.3, Numba:** duas frases, com “o que o torna muito mais rápido” e “Esse laço
  é executado em paralelo…”.
- **3.3, Gram-Schmidt:** três frases, com “podem mudar de classe…”, “Por isso,
  são recalculados…” e “Foram 23.000…”.
- **3.5, pilares:** três frases para I, D + C e A + N, mantendo a citação e a
  frase final sobre o enlace.
- **4.4, horizonte longo:** uma frase para as porcentagens e outra para os
  transitórios, com a definição entre parênteses.

### Padrões 10 a 13

**Ponto e vírgula (13 ocorrências trocadas por ponto):** 1.3 “Hénon; a escolha”;
2.2 “separam; outros valores”; 2.3 “do filtro; com βK = 0”; 2.4 “mapa original;
para Nz = 2”; 2.5 “nessa direção; um expoente”; 2.5 “jacobiana; os vetores”;
2.5 “classificação; perto da fronteira”; 2.6 “círculo unitário; o erro”;
3.1 “sem a Kaiser; na Kaiser”; 3.3 “órbita; se alguma”; 3.6 “0,99π; o seu
atraso”; 4.3 “Nz = 40; as órbitas”; 4.4 “caóticos; o maior desvio”. As listas
de contribuições, testes, TRL, trabalhos futuros e citações mantiveram o ponto e
vírgula.

**Cadeias de “e” (3 orações):** 2.6 “e os demais são nulos, de modo que a
condição…” → “e os demais são nulos. Assim, a condição…”; 2.7 “…chaveamento
diferencial… (Kolumbán et al., 1996), que dispensa…” → “…(Kolumbán et al.,
1996). Esse último dispensa…”; 4.3 “menos que o tipo, e o efeito se concentra…”
→ “menos que o tipo. O efeito se concentra…”.

**“Isto é”:** saiu de 3.3 (“isto é, a menos de θ do limiar” → “a menos de θ do
limiar”). As seis ocorrências restantes definem termos (DSCI, semente, Numba,
falha segura, leitura percentual do η²p, margem do filtro) e foram mantidas.

**Frases longas divididas:** 2.3 janela (“…fora da banda. Janelas com…”); 2.6
filtro (“…são nulos. Assim,…”); 3.1 η² (“…sem teste de significância. Ele dá…”);
3.1 Hann/Bartlett (“…são nulas. Esses pontos…”); 3.5 Quadro (“…entre parênteses.
As outras 9…”); 4.3 janela (“…menos que o tipo. O efeito se concentra…”); 4.3
rejeita-faixa (“…formar o entalhe. Com Nz = 40…”); 2.7 esquemas (“…(Kolumbán
et al., 1996). Esse último…”); conclusão (“…sete janelas propostas. Isso
amplia…”).

### Métricas

- **2.7:** “Esse último dispensa…” passou a “O chaveamento diferencial dispensa…”.
- **3.1:** o segundo dois-pontos da frase do η² virou ponto: “… indica soma de
  quadrados. SSfator é a soma entre níveis do fator…”.
- Média de palavras por frase: **29,4 → 26,0**.
- Pontos e vírgulas no corpo: **44 → 18**.
- Total de palavras: **7.584 → 7.574** (sem conteúdo novo).

## Revisão final das referências (NBR 6023:2025)

- **Regra única de autores:** `maxbibnames=3` e `minbibnames=1` no estilo; todas
  as obras com quatro ou mais autores saem com “et al.”, inclusive Chorti,
  Alvarez et al. (2003), Benettin et al., Mukherjee et al., Wolf et al. e os
  artigos do NumPy e do SciPy. Nenhuma lista foi editada à mão.
- **Partículas:** Silva passou a `given-i={K. de S.}` nas duas entradas do grupo;
  Santos ficou `G. R.` para não gerar ponto duplo; Borges permanece “V. da S.”.
- **Pontuação:** não há mais “dos.;” na lista.
- **Alligood:** editora corrigida para “Springer-Verlag”.
- **Mankins:** “Washington: NASA, 1995. Advanced Concepts Office.”
