# Resposta ao orientador — PFC v2 → v3

Base do diff: tag local `pfc-v2-orientador` (`f4886d6`). O PDF `diff-v2-v3.pdf`
está nesta pasta. Os itens 1 a 30 são as anotações do PDF; M1 a M4 são os pedidos
por mensagem.

| Item | Status | Observação |
|---|---|---|
| 1 | Aplicado com ajuste | Resumo e abstract reescritos com o texto proposto; “cada projeto” virou “cada ponto” para cumprir o item 14. |
| 2 | Aplicado | Rodapé com “1º de dezembro de 2026”, por `\datadefesa`, sem intervalo. |
| 3 | Aplicado | Segundo “porém” trocado por “Além disso”. |
| 4 | Aplicado | Vetor de estado definido na 2.2 e estendido na 2.4. |
| 5 | Aplicado | Órbita definida em frase própria. |
| 6 | Aplicado | Filtro escrito uma vez, com índice j, $H(z)=X_3/X_1$; a 2.4 cita a equação. |
| 7 | Aplicado | Ganho DC $G=H(e^{j0})=\sum c_j$ definido na 2.3. |
| 8 | Aplicado | Variáveis auxiliares com o atraso explícito. |
| 9 | Aplicado | “Dimensão (número de variáveis de estado)”. |
| 10 | Aplicado | Ponto fixo e equação $G^2p^2+(1-\beta)p-\alpha=0$ na 2.4; 3.1 e 4.3 citam. |
| 11 | Aplicado | Jacobiana do mapa (1) escrita. |
| 12 | Aplicado | Recursão do erro sem filtro e autovalores $\pm\sqrt{\beta}\approx\pm0{,}55$. |
| 13 | Aplicado | Equações do transmissor/receptor sem filtro e com filtro; derivação conferida contra o código. |
| 14 | Aplicado | “Configuração” e “ponto” definidos; $N_c$ definido; termos antigos substituídos. |
| 15 | Aplicado com ajuste | Desenho balanceado: 24 fixas e 160 da Kaiser; η² tipo/janela 0,588/0,387; βK 0,495. Com uma configuração por célula, os 32,7% são o resíduo do modelo aditivo, igual à interação tipo × janela; o η² do tipo vem de dois desenhos e a comparação com βK e janela é de ordem, não de valor. |
| 16 | Aplicado | $\omega_c$ mantido; texto e legendas avisam que é frequência central no passa-faixa e no rejeita-faixa. |
| 17 | Aplicado | Paridade justificada pelo ganho nulo em $\omega=\pi$. |
| 18 | Aplicado com ajuste | O código trunca as bordas em $10^{-5}$ e $1-10^{-5}$: 3 valores por borda, 6 por ordem; 240 no passa-faixa e 120 no rejeita-faixa, 360 por par de configurações da mesma janela. Na grade inteira, os 46 pares somam 16.560 dos 552.000 pontos. |
| 19 | Aplicado | “Transitório” em todo o texto, com “da órbita” e “de sincronização”; Fig. 6 corrigida. |
| 20 | Aplicado | $m(n)=\pm1$ explicitado. |
| 21 | Aplicado | Notação $\hat x$ e alinhamento pelo atraso de 100 amostras explicitados. |
| 22 | Aplicado | $\varepsilon_s=10^{-3}$; frase do θ removida. |
| 23 | Aplicado | Fronteira do passa-baixas descrita com os valores de $\omega_c/\pi$. |
| 24 | Aplicado | “A maior parte” no lugar de generalização; $G=0$ e equação do ponto fixo citados. |
| 25 | Aplicado | Janelas nomeadas: 3,5% (retangular) e 14,3% (Blackman-Harris), 16,4% no melhor βK. |
| 26 | Aplicado | Recomendação remete à janela adotada e às Figs. 9a e 12. |
| 27 | Aplicado | “Elevar θ para $10^{-2}$”. |
| 28 | Aplicado | “$\lambda_1>0{,}05$, dez vezes o limiar θ”. |
| 29 | Aplicado | Resultado em função de $\varepsilon_s$. |
| 30 | Aplicado com ajuste | Título “exponente” mantido como publicado; ordem cronológica corrigida com `sorting=nyt` e `sortyear`. |
| M1 | Aplicado com ajuste | A seta do fluxograma (Fig. 6) tinha uma ponta intermediária no ponto `j`; virou uma única seta até a decisão. Recorte: `fig6-seta-antes-depois.png`. Demais diagramas conferidos a 200 dpi. |
| M2 | Aplicado | Borges, Silva e Eisencraft (2024) citados no ganho DC e no ponto fixo. |
| M3 | Aplicado | Título novo na capa e nos metadados. |
| M4 | Aplicado com ajuste | Rodapé com “1º de dezembro de 2026”; a classe ganhou `\datadefesa` para data única, mantendo o intervalo como fallback. |

## Pendências

- O PDF ficou com **25 páginas**, com 5 linhas livres na última. A folga veio da
  redução das alturas das Figs. 8, 13, 11 e 2; o Quadro 1 e as repetições não
  foram alterados.
- `pfc-banca-2026-11` não existe local nem remotamente; a base do diff é
  `pfc-v2-orientador`.
