# Resposta v3 → v4

Base do diff: tag `pfc-v3-orientador`, commit `c388b17` (25 páginas). O diff
está em `diff-v3-v4.pdf`; o `diff-v2-v3.pdf` permanece congelado.

## Mudanças da v3 para a v4

| Bloco | O que mudou | Motivo |
|---|---|---|
| Rodapé | A data passou ao formato de intervalo do template, “1º a 4 de dezembro de 2026”, com `\diainicial{1º}`, `\diafinal{4}` e `\mes{dezembro}` no preâmbulo. Como no PDF canônico, o “ - ” fecha a primeira linha (“… SP - Brasil -”) e a data vem na linha seguinte. | A banca ocorre na semana de 1º a 4 de dezembro. |
| Metodologia | A abertura da Seção 3 começa por “Segundo a classificação de Prodanov e Freitas (2013)” e classifica a pesquisa como aplicada, quantitativa, descritiva e experimental por simulação, com as variáveis manipuladas e observadas e as condições controladas, sem a citação repetida no fim. A frase do canal passou a “Como todo canal, porém, limita a banda, o filtro FIR entra…”. | Explicitar o tipo de pesquisa com uma fonte verificada e tirar os parênteses seguidos. |
| 1.3, hipótese | A hipótese passou a ser geral: o efeito do filtro depende do tipo e da janela e existem filtros que preservam o caos e permitem essa sincronização. O passa-baixas deixou de ser premissa e aparece como resultado na 4.3 e na conclusão. | Evitar que a hipótese pareça escrita depois dos resultados. |
| Figuras 8 e 13 | A Figura 8 voltou à altura de 510,22 × 244,906 pt, igual à Figura 13. Os títulos dos painéis de tempo ficaram acima dos eixos e os do espectro dentro do gráfico, no topo; a Figura 8 é um flutuante no topo da página 16, com legenda, desenho e fonte juntos, e o texto da 4.2/4.3 preenche a página 15. A margem inferior da Figura 13 foi ajustada para a fonte ficar logo abaixo do desenho. | Recuperar a legibilidade dos painéis e tirar o vão antes da fonte. |
| Diagramação | As Figuras 8, 9 e 10 são flutuantes `[t]`/`[!t]`, com legenda, desenho e fonte juntos; as demais figuras, tabelas e quadros foram conferidos um a um e ficam inteiros na mesma página. A Figura 9 voltou ao aspecto 0,67, com os painéis próximos de quadrados e o eixo y de 0 a 1 com marcas a cada 0,2, e flutua para o topo da página 17. | Impedir que legenda, desenho e fonte se separem entre páginas, sem encolher figuras. |
| Abstract | “each design is classified” e “verified for each design” passaram a “each grid point is classified” e “verified for each grid point”, para concordar com “cada ponto” do resumo. | Alinhar o abstract ao resumo. |
| 3.1, η² parcial | A frase passou a dizer que o η² parcial é uma medida descritiva do tamanho do efeito, sem teste de significância. | Deixar explícito o alcance da medida. |
| Conclusão | Reescrita em quatro parágrafos: abertura pedida, “O objetivo geral foi atingido, e os resultados confirmam a hipótese”, objetivos específicos na ordem da Seção 1.3 com “Quanto ao primeiro/segundo/terceiro/quarto”, limitações e trabalhos futuros ligados a cada limitação. O primeiro objetivo foi dividido em duas frases, sem ponto e vírgula, e a comparação de escopo foi refeita conforme o que cada trabalho anterior cobre. O último parágrafo começa por “Como trabalhos futuros, recomenda-se, em resposta a essas limitações”. | Responder às três perguntas do modelo da faculdade. |
| Fluidez | Ponto e vírgula, aposições empilhadas, frases “X é Y” encadeadas e conectivos repetidos foram revistos em todo o texto, sem mudar conteúdo, números, termos, símbolos ou citações. | Melhorar a legibilidade. |
| 4.3, η² | Três frases: tipo (0,588 fixas; 0,759 Kaiser), βK (0,495) e janela (0,387) em ordem, e 32,7% como variação não explicada pelo modelo aditivo, igual à interação tipo × janela no desenho fixo. | Dar o referente de cada número e explicar o resíduo/interação. |
| SIMAC | “Foi apresentada” no lugar de “foi aceita”; a frase foi juntada ao fim do parágrafo das contribuições; o `note` do `.bib` passou a “apresentado em 7 nov. 2026”. | Trabalho apresentado em 7/11. |
| Classe e template | Classe original restaurada; opções e macro de data no preâmbulo. Bloco de autores com nota de rodapé, orientador “Dr.”, fontes “Autoria própria”, tabelas na largura do texto, Tabela 1 centralizada, nota da Tabela 3 em linha própria. | Conformidade com o modelo da faculdade. |
| Introdução | `\needspace` antes da seção, sem quebra forçada; o bloco vai junto para a página 2 quando não cabem duas linhas. | Evitar órfão. |
| Quadro 1 | Nova coluna “Pilar”, com I nas três primeiras linhas e em avisos numéricos e auditoria do manuscrito, D em execução sem compilador, C, I em análise estática e D, I em dependências; legenda com a abreviatura dos pilares. | Dizer a que pilar cada categoria de verificação serve. |
| 3.5 | Parágrafo que associa os mecanismos já descritos aos pilares C, I, D, A e N e registra que C, A e N não foram tratados no enlace (Seção 4.7). A lista de categorias secundárias do CI passou a “as outras 9 cuidam da distribuição e da documentação”, e o texto explicita que o Quadro 1 cobre 14 das 23 verificações. | Explicitar os pilares sem nova análise e deixar claro o alcance do quadro. |
| 3.5, citação | Acrescentada a referência à ISO/IEC 27000:2018, que define confidencialidade, integridade, disponibilidade, autenticidade e não repúdio: “(ISO, 2018)” ao fim da primeira frase do parágrafo dos pilares. | Dar origem normativa aos cinco pilares. |
| Trabalhos futuros | Acrescentado o item “a análise de segurança do esquema de comunicação (confidencialidade e autenticidade do enlace)”. | Registrar a lacuna de segurança do enlace. |
| 4.7 | Conferido: o texto continua dizendo que o esquema de comunicação não oferece garantia criptográfica. | Sustentar a remissão da 3.5. |
| Figuras 4 e 7 | Novo desenho dos dois enlaces, com estilos próprios de prefixo `enl` no preâmbulo: blocos, grupos tracejados de transmissor e receptor, realimentação de `s(n)`, `r(n)` acionando a cópia e sinais `+` e `−` no somador do receptor. Legendas, fontes e `\label` mantidos; o desenho ganhou 2,5 pt de respiro sob a legenda, e os rótulos `x̂1(n)` e `x̂3(n)` foram afastados 3 pt da seta. | Deixar a geometria igual nos dois enlaces e mostrar a filtragem no transmissor e no receptor. |

## Sugestão para o orientador (resumo e abstract)

O modelo da faculdade pede a descrição básica da metodologia. Foram alteradas
apenas duas expressões, marcadas para aprovação ou reversão:

- Resumo: “por meio de simulações numéricas” passou a “por meio de uma pesquisa
  quantitativa e experimental com simulações numéricas”.
- Abstract: “through numerical simulations” passou a “through a quantitative
  experimental study based on numerical simulations”.

O restante do resumo e do abstract não foi tocado. Se o orientador preferir o
texto original, basta reverter essas duas expressões.

## Revisão final de português e definições

- **4.3:** “Cada painel corresponde a um ponto da grade…” passou a “Em cada
  painel, cada ponto corresponde a um filtro com tipo, janela, ordem e corte
  definidos…”.
- **1.1:** a citação passou a “(Kaddoum, 2016; ver Seções 2.1 e 2.7)”.
- **1.1:** “a baixa probabilidade de interceptação” passou a “propriedade chamada
  de baixa probabilidade de interceptação”.
- **2.3:** a frase passou a “Com Nz + 1 coeficientes cj, a saída x3(n) obtida a
  partir de x1(n) e a função de transferência H(z) são dadas por”.
- **Equações:** “o filtro FIR da (2)” passou a “de (2)” e “na mesma ordem da
  (14)” a “na mesma ordem de (14)”.
- **Seção 3:** “a grade de pontos de filtro” passou a “a grade de filtros”.
- **3.1:** as duas frases do η² foram juntadas em uma, com a medida descritiva do
  tamanho do efeito e a variação de 0 a 1 antes da fórmula.
- **4.3:** “cada coordenada de ponto” passou a “cada par (Nz, ωc)”.
- **4.3:** saiu “a relação se inverte” de “No rejeita-faixa, o efeito da janela
  muda de sinal…”.
- **4.5:** passou a “A conclusão não depende do limiar de sincronização: com εs
  entre 10⁻³ e 10⁻¹, os três cenários se classificam da mesma forma.”
- **Conclusão:** “critério de corte sobre uma estimativa” passou a “limiar
  aplicado a uma estimativa”.
- **Definições:** OSI ganhou “(Open Systems Interconnection, modelo de referência
  em camadas para redes)” e o ganho DC, “(do inglês direct current, o ganho na
  frequência zero)”.
- **Tabela 3:** o título passou a “Otimizações do desempenho”; a nota mantém o
  processador.

## Observações

- A tag `pfc-v2-orientador` continua em `f4886d6`.
- O PDF do roteiro de segurança não foi localizado na pasta; a proposta usa os
  cinco pilares nomeados no pedido.
- Não foi localizada referência verificada para a lista dos cinco pilares, então
  a 3.5 não traz citação nova.
- A citação dos pilares passou a ser a ISO/IEC 27000:2018, verificada na página
  oficial da ISO: título, 5. ed., 2018, Geneva, ISO. A edição de 2018 está
  retirada desde 3 jul. 2026 e substituída pela ISO/IEC 27000:2026; a entrada
  mantém a edição de 2018, que é a que o pedido indica.
- O estilo do template é o biblatex-abnt 4.0 (NBR 6023:2018); a entrada foi
  montada com os campos da NBR 6023:2025 e não houve conflito de renderização.
- A tabela de mecanismos para os slides, com os nomes do artigo, está em
  `articles/pfc/revisao/seguranca/mecanismos-por-pilar.md`.
- `tools/check_pfc_diagrams.py` e `diagram_paths.json` só conferem rótulos e
  símbolos do código, não coordenadas; nenhum ajuste foi necessário.
- As Figuras 4 e 7 foram conferidas seta a seta contra as equações do texto; o
  modelo não mudou, apenas o desenho.
- O `ci.yml` tem 23 jobs; as oito categorias do Quadro 1 somam 14 verificações
  (2+3+1+1+1+3+2+1), e as outras 9 são distribuição e documentação.
- A ISO/IEC 27000:2026 foi conferida na página da ISO (título, 6. ed., 2026-07,
  Geneva); como a própria página diz que a edição deixou de ser um documento de
  terminologia e não foi possível confirmar nela as cinco definições, a citação
  dos pilares permaneceu na ISO/IEC 27000:2018, como pedido.
- A referência de metodologia é PRODANOV, C. C.; FREITAS, E. C. *Metodologia do
  trabalho científico: métodos e técnicas da pesquisa e do trabalho acadêmico*.
  2. ed. Novo Hamburgo: Feevale, 2013. ISBN 978-85-7717-158-3. Os metadados foram
  conferidos na edição digital gratuita da Feevale; as quatro classificações
  usadas estão nas seções 3.4.1, 3.4.2, 3.4.3 e 3.4.4 da obra.
  é reduzida em altura ou aspecto para resolver paginação; usam-se flutuantes e
  o fluxo do texto, e, se passar de 25 páginas, o trabalho para e reporta as
  linhas excedentes.
- As Figuras 4 e 7 tiveram o espaço vertical reduzido (cópia do mapa de −1,75
  para −1,55 e μ de 1,35 para 1,2) para o artigo voltar a 25 páginas.
- Escopo dos trabalhos anteriores da linha, conferido nas fontes:

  | Trabalho | Tipos | Janelas | Fonte |
  |---|---|---|---|
  | Borges e Eisencraft (2022a) | FIR de dois coeficientes | nenhuma | artigo, seção 2 |
  | Borges e Eisencraft (2022b) | passa-baixas | Hamming e Blackman | artigo, seção III |
  | Borges, Silva e Eisencraft (2023) | passa-baixas | Hamming | artigo, p. 1 |
  | Borges, Silva e Eisencraft (2024) | cinco famílias: zeros igualmente espaçados, par complexo (rejeita-faixa), par + zero em −1, zeros em −1 e passa-baixas | Hamming (no passa-baixas) | artigo, seção 4 |
  | Borges (2026) | as mesmas cinco famílias, com cenários de comunicação | Hamming | tese, cap. 4 e Experimento V |
