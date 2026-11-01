%% ===========================================================================
%%  pfccefsa.cls — classe LaTeX do artigo científico do PFC/TCC
%%  Faculdade Engenheiro Salvador Arena (CEFSA / FESA) — modelo oficial V-05
%% ---------------------------------------------------------------------------
%%  Esta classe reúne, em um só lugar, todo o layout do modelo oficial em Word
%%  ("Template Canonico para PFC CEFSA.docx" / "Template Canonico PFC CEFSA.pdf",
%%  versão V-05) e as normas ABNT que se aplicam a um artigo científico:
%%
%%    ABNT NBR 6022:2018  artigo em publicação periódica científica
%%    ABNT NBR 6023:2018  elaboração de referências (via biblatex-abnt)
%%    ABNT NBR 6024:2012  numeração progressiva das seções
%%    ABNT NBR 6028:2003  resumo
%%    ABNT NBR 10520:2023  citações (citação longa, autor-data, apud)
%%
%%  O arquivo .tex do artigo fica só com o conteúdo: não é preciso escrever
%%  espaçamentos, caixas de legenda, recuos nem marcas de formatação.
%%
%%  TODAS AS MEDIDAS ESTÃO EM "bp" (pontos PostScript, 1/72 polegada), que é a
%%  unidade do PDF gerado pelo Word.  NÃO troque por "pt" (1/72,27 polegada): o
%%  documento inteiro encolheria 0,37 % e todas as linhas mudariam de lugar.
%% ===========================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pfccefsa}[2026/09/16 v1.0 Artigo científico PFC/TCC CEFSA-FESA (ABNT)]

% O kernel lê arquivos .cls com @ como letra; a linha abaixo deixa isso explícito.
\makeatletter

%% ---------------------------------------------------------------------------
%%  1. OPÇÕES
%%     alf  (padrão) — citações no sistema autor-data  (NBR 10520:2023)
%%     num           — citações no sistema numérico
%%     outras opções são repassadas para a classe base "article"
%% ---------------------------------------------------------------------------
\newif\ifpfc@autorano
\pfc@autoranotrue
\DeclareOption{alf}{\pfc@autoranotrue}
\DeclareOption{num}{\pfc@autoranofalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[12pt,a4paper]{article}

%% ---------------------------------------------------------------------------
%%  2. FONTES
%%     O modelo oficial usa Times New Roman (texto), Century Gothic (o "V-05"
%%     do rodapé) e Calibri (apenas dentro do gráfico de exemplo, que é um PDF).
%%     Se o Times New Roman não estiver instalado, usa-se a Liberation Serif,
%%     que tem as MESMAS larguras de glifo; em último caso, o TeX Gyre Termes.
%% ---------------------------------------------------------------------------
\RequirePackage{iftex}
\ifPDFTeX
  \RequirePackage[T1]{fontenc}
  \RequirePackage[utf8]{inputenc}
  \RequirePackage{mathptmx}             % Times — mesmas métricas do TNR
  \IfFileExists{tgadventor.sty}
    {\RequirePackage{tgadventor}%
     \newcommand{\fontecentury}{\fontfamily{qag}\selectfont}}
    {\newcommand{\fontecentury}{\fontfamily{phv}\selectfont}}
\else
  \RequirePackage{fontspec}
  \IfFontExistsTF{Times New Roman}
    {\setmainfont{Times New Roman}}
    {\IfFontExistsTF{Liberation Serif}
       {\setmainfont{Liberation Serif}}          % metricamente idêntica ao TNR
       {\setmainfont{texgyretermes-regular.otf}}}% clone do Times do TeX Live
  \IfFontExistsTF{Century Gothic}
    {\newfontfamily\fontecentury{Century Gothic}}
    {\newfontfamily\fontecentury{texgyreadventor-regular.otf}}
\fi

%% ---------------------------------------------------------------------------
%%  3. TAMANHOS E ENTRELINHAS (idênticos ao Word)
%%      12 bp -> 13,8 bp   texto corrido
%%      11 bp -> 12,65 bp  título/fonte de figura, tabela e quadro
%%      10 bp -> 11,5 bp   nota de rodapé e número de página
%%       8 bp -> 9,2 bp    "V-05" do rodapé (Century Gothic)
%% ---------------------------------------------------------------------------
\newcommand{\corpo}   {\fontsize{12bp}{13.8bp}\selectfont}
\newcommand{\tamonze} {\fontsize{11bp}{12.65bp}\selectfont}
\newcommand{\tamdez}  {\fontsize{10bp}{11.5bp}\selectfont}
\newcommand{\tamoito} {\fontsize{8bp}{9.2bp}\selectfont}
\renewcommand{\normalsize}{\corpo}
\normalsize

%% ---------------------------------------------------------------------------
%%  4. PÁGINA E MARGENS
%%     Papel A4 exatamente como no Word (595,32 x 842,04 bp).
%%     margens: superior 97,8 bp (3,45 cm) | inferior 74,5 bp
%%              esquerda 42,55 bp (1,5 cm) | direita 42,55 bp (1,5 cm)
%%     A área de texto termina em 767,54 bp — acima do bloco do rodapé, que é o
%%     mesmo critério de quebra de página usado pelo Word.
%% ---------------------------------------------------------------------------
\RequirePackage{geometry}
\geometry{
  paperwidth=595.32bp, paperheight=842.04bp,
  left=42.55bp, right=42.55bp, top=97.8bp, bottom=74.5bp,
  headheight=0pt, headsep=0pt, footskip=0pt,
  nohead, nofoot
}

%% ---------------------------------------------------------------------------
%%  5. IDIOMA
%%     Português do Brasil para as expressões geradas automaticamente
%%     ("e" entre autores, "et al.", datas) e para a hifenização.
%% ---------------------------------------------------------------------------
\RequirePackage[brazil]{babel}

%% ---------------------------------------------------------------------------
%%  6. PARÁGRAFOS
%%     recuo da 1ª linha = 1,25 cm (709 twips) | sem espaço entre parágrafos
%%     \frenchspacing: no Word o espaço depois do ponto final tem a MESMA
%%     largura dos demais; sem isso o TeX alarga esse espaço (o "espaço extra
%%     de fim de frase") e a linha fica visivelmente diferente do modelo.
%% ---------------------------------------------------------------------------
\RequirePackage{indentfirst}
\setlength{\parindent}{35.45bp}
\setlength{\parskip}{0pt}
\setlength{\topskip}{11.18bp}   % 1ª linha da página começa em 97,8+11,18 bp
\setlength{\lineskip}{0pt}
\lineskiplimit=0pt
\clubpenalty=10000              % sem linhas viúvas/órfãs (o Word também não as cria)
\widowpenalty=10000
\hyphenpenalty=10000            % o Word NÃO hifeniza este template
\exhyphenpenalty=10000
\tolerance=2000
\emergencystretch=1em
\frenchspacing

%% ---------------------------------------------------------------------------
%%  7. PACOTES DE APOIO
%%     xcolor[table] precisa vir antes do tikz; biblatex antes do hyperref.
%% ---------------------------------------------------------------------------
\RequirePackage[table]{xcolor}  % cores nas linhas de tabela (colortbl)
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{enumitem}
\RequirePackage{etoolbox}
\RequirePackage{environ}        % ambientes que capturam o próprio corpo
\RequirePackage{titlesec}
\RequirePackage{tikz}
\RequirePackage{eso-pic}
\RequirePackage[
  backend=biber,
  style=\ifpfc@autorano abnt\else abnt-numeric\fi,
  justify=false,               % referências alinhadas à esquerda (como no Word)
  indent=false,                % sem recuo pendente (NBR 6023: sem deslocamento)
  oldspacing,                  % o espaço entre referências é definido abaixo
  repeatfields=false,
]{biblatex}
\RequirePackage{hyperref}
\hypersetup{hidelinks}

%% ---------------------------------------------------------------------------
%%  7.1 ESTILO ABNT VENDORIZADO — ajustes para a NBR 6023:2025
%%      O biblatex-abnt está vendorizado em biblatex-abnt/ (upstream 4.0, sem
%%      alteração nos arquivos dele) e é ele que o XeLaTeX/biber usam — ver
%%      docs/vendorizacao-biblatex-abnt.md.  O upstream 4.0 implementa a
%%      NBR 6023:2018; a edição de 2025 é a consolidação de 2018 com a
%%      Emenda 1, e os ajustes abaixo aplicam os pontos que ela muda:
%%        8.1.1.2  quatro ou mais autores: "convém indicar todos" (a lista
%%                 passa a indicar todos; a citação segue com "et al.")
%%        7.7.6    DOI como endereço completo — DOI: https://doi.org/...
%%        8.4.4    local desconhecido: [S. l.] em itálico
%%        8.6.1.4  intervalos com hífen (p. 12-34; 2018-2022), como nos
%%                 exemplos da norma — se a instituição exigir travessão,
%%                 troque o "-" do \AtBeginDocument abaixo por \textendash
%% ---------------------------------------------------------------------------

% 8.1.1.2 — a lista de referências indica todos os autores (a citação, que
% segue a NBR 10520:2023, continua com "et al." a partir de quatro).
\ExecuteBibliographyOptions{maxbibnames=99}

% 7.7.6 — o DOI vira https://doi.org/... (o campo continua sendo só o
% identificador: doi = {10.1007/s11116-023-10391-3}).
\DeclareFieldFormat{doi}{\mkbibacro{DOI}\addcolon\space\url{https://doi.org/#1}}

% 8.6.1.4 — intervalos (páginas e anos) com hífen, como nos exemplos da norma.
% \bibrangedash é restabelecido pelo biblatex no começo do documento, por isso
% a redefinição vai em \AtBeginDocument; \bibdaterangesep herda dele.
\AtBeginDocument{\renewcommand*{\bibrangedash}{-}}

% 8.4.4 — "sine loco" abreviado e em itálico.  O .bib pode trazer [S.l.] ou
% [S. l.] (é o que o modelo Word escreve); a lista sempre imprime [S. l.].
% A definição repete a de biblatex.def (list:delim/list:andothers) para não
% perder a pontuação entre locais.
\DeclareListFormat{location}{%
  \usebibmacro{list:delim}{#1}%
  \ifboolexpr{test {\ifstrequal{#1}{[S.l.]}} or test {\ifstrequal{#1}{[S. l.]}}}%
    {\mkbibemph{[S. l.]}\isdot}%
    {#1\isdot}%
  \usebibmacro{list:andothers}%
}

% Os exemplos da norma escrevem os intervalos de páginas com hífen; o "--" do
% .bib viraria travessão na composição.  O "final" é o que faz o passo só valer
% quando o casamento acontece — num \DeclareSourcemap os passos seguintes rodam
% mesmo sem casar se o passo não tiver "final" (e um "]" literal dentro de
% \step[...] fecharia o argumento antes da hora; em regexp se usa \x5D).
\DeclareSourcemap{
  \maps[datatype=bibtex]{
    \map[overwrite]{
      \step[fieldsource=pages, match=\regexp{--}, final, replace={-}]
    }
  }
}

%% ---------------------------------------------------------------------------
%%  8. SEÇÕES — 12 bp, negrito, alinhadas à esquerda,
%%     uma linha em branco (13,8 bp) antes e depois (NBR 6024 + modelo CEFSA).
%%     O modelo usa maiúsculas e minúsculas (o padrão CEFSA prevalece sobre a
%%     NBR 6024, que grafaria o título da seção primária em maiúsculas).
%% ---------------------------------------------------------------------------
\titleformat{\section}[block]      {\normalfont\corpo\bfseries}{\thesection}{0.25em}{}
\titleformat{\subsection}[block]   {\normalfont\corpo\bfseries}{\thesubsection}{0.25em}{}
\titleformat{\subsubsection}[block]{\normalfont\corpo\bfseries}{\thesubsubsection}{0.25em}{}
\titleformat{\paragraph}[runin]    {\normalfont\corpo\bfseries}{\theparagraph}{0.25em}{}
\titleformat{\subparagraph}[runin] {\normalfont\corpo\bfseries}{\thesubparagraph}{0.25em}{}
\titlespacing*{\section}{0bp}{13.8bp}{13.8bp}
\titlespacing*{\subsection}{0bp}{13.8bp}{13.8bp}
\titlespacing*{\subsubsection}{0bp}{13.8bp}{13.8bp}
\titlespacing*{\paragraph}{0bp}{13.8bp}{0.5em}
\titlespacing*{\subparagraph}{0bp}{13.8bp}{0.5em}
% "Referências" não é numerada e o espaço antes do 1º parágrafo é dado
% explicitamente (12 bp), como no Word.
\titlespacing*{name=\section,numberless}{0bp}{0bp}{0bp}
\setcounter{secnumdepth}{5}

%% ---------------------------------------------------------------------------
%%  9. PRIMITIVAS DO MODELO OFICIAL
%%     São o "motor" tipográfico extraído do PDF canônico.  Os comandos de
%%     alto nível da classe (ambientes figura/tabela/quadro, resumo, etc.) já
%%     os usam; eles continuam públicos para casos especiais.
%% ---------------------------------------------------------------------------

% LINHA EM BRANCO DO WORD (parágrafo vazio de 12 bp).
% A regra tem exatamente a altura/profundidade da linha de 12 bp do Times New
% Roman (11,2 bp acima da linha de base e 2,6 bp abaixo), para que o
% espaçamento entre linhas não seja alterado por glifos altos.
\newcommand{\linhavazia}{\par\noindent\rule[-2.6bp]{0pt}{13.8bp}\par}

% TÍTULO E FONTE DE FIGURAS / TABELAS / QUADROS (11 bp, centralizados).
% O Word coloca uma linha de 12 bp antes do título; o ajuste de 0,22 bp
% corrige a diferença de entrelinha entre 12 bp e 11 bp.
\newcommand{\caixatitulo}[1]{%
  \par\vspace{0.22bp}{\centering\tamonze #1\rule[-2.38bp]{0pt}{0pt}\par}}
\newcommand{\caixafonte}[1]{%
  \par{\centering\tamonze\rule{0pt}{10.27bp}#1\rule[-2.38bp]{0pt}{0pt}\par}\vspace{-0.22bp}}
% Nome no estilo abnTeX2 para a fonte de uma ilustração.
\newcommand{\fonte}[1]{\caixafonte{#1}}

% BLOCOS (figura / tabela / quadro de altura qualquer)
% O Word empilha blocos assim: o bloco começa exatamente no fim da caixa da
% linha anterior e a linha de texto seguinte começa a uma "ascensão" da fonte
% (10,27 bp em 11 bp) depois do fim do bloco.  As linhas abaixo reproduzem esse
% comportamento para qualquer altura de bloco.
\newsavebox{\caixabloco}
\newcommand{\bloco}[1]{%
  \sbox{\caixabloco}{\ignorespaces#1\unskip}%
  \par
  % Os \nobreak impedem que o TeX quebre a página no meio do bloco.  Sem eles,
  % uma figura alta que não caiba na página faz o TeX quebrar no \vspace do
  % avanço: o bloco vai para a página seguinte sem o avanço e a imagem é
  % desenhada a partir da linha de base, com a altura para fora do papel.
  \nobreak
  \vspace{\dimexpr\prevdepth-\baselineskip+\ht\caixabloco\relax}%
  \nobreak
  \noindent\smash{\usebox{\caixabloco}}%
  \par
  \nobreak
  \vspace{\dimexpr\dp\caixabloco-2.50bp\relax}% V2
}

%% ---------------------------------------------------------------------------
%% 10. NOTA DE RODAPÉ (igual ao Word: 10 bp, marca sobrescrita de 6,48 bp,
%%     linha separadora de 2 polegadas = 144 bp)
%% ---------------------------------------------------------------------------
\newlength{\larguramarcanota}
\renewcommand{\footnoterule}{%
  \kern-4.21bp\hrule\@width 2in\@height 0.6bp\@depth\z@\kern 6.41bp}
\setlength{\skip\footins}{23.95bp}
\newcommand{\marcanotacorpo}[1]{\smash{\raisebox{3.96bp}{\fontsize{8.04bp}{8.04bp}\selectfont #1}}}
\renewcommand{\@makefnmark}{\marcanotacorpo{\@thefnmark}}
% A marca da nota é desenhada elevada em 3,5 bp, mas dentro de uma caixa de
% largura zero (senão o TeX a contabilizaria na profundidade da linha e o
% espaço entre as duas linhas da nota passaria de 11,5 bp para 15 bp).
\renewcommand{\@makefntext}[1]{%
  \raggedright\parindent\z@\noindent
  \settowidth{\larguramarcanota}{\fontsize{6.48bp}{6.48bp}\selectfont\@thefnmark}%
  \raisebox{3.5bp}{\hbox to \z@{\fontsize{6.48bp}{6.48bp}\selectfont\@thefnmark\hss}}%
  \hspace{\larguramarcanota}\hspace{0.25em}#1}
\renewcommand{\footnotesize}{\tamdez}

%% ---------------------------------------------------------------------------
%% 11. DADOS DO TRABALHO (usados no rodapé, no cabeçalho e no PDF)
%%     Os padrões reproduzem exatamente os textos do modelo oficial.
%% ---------------------------------------------------------------------------
\newcommand{\pfc@titulo}{}
\newcommand{\pfc@autores}{}
\newcommand{\pfc@orientador}{}
\newcommand{\pfc@curso}{\textless{}nome do curso\textgreater{}}
\newcommand{\pfc@diainicial}{[DIA INICIAL]}
\newcommand{\pfc@diafinal}{[DIA FINAL]}
\newcommand{\pfc@mes}{[MÊS]}
\newcommand{\pfc@ano}{2026}
\newcommand{\pfc@versao}{V-05}

\newcommand{\titulo}[1]{\gdef\pfc@titulo{#1}\hypersetup{pdftitle={#1}}}
\newcommand{\autores}[1]{\gdef\pfc@autores{#1}}
\newcommand{\orientador}[1]{\gdef\pfc@orientador{#1}}
\newcommand{\curso}[1]{\gdef\pfc@curso{#1}}
\newcommand{\diainicial}[1]{\gdef\pfc@diainicial{#1}}
\newcommand{\diafinal}[1]{\gdef\pfc@diafinal{#1}}
\newcommand{\mes}[1]{\gdef\pfc@mes{#1}}
\newcommand{\ano}[1]{\gdef\pfc@ano{#1}}
\newcommand{\versaomodelo}[1]{\gdef\pfc@versao{#1}}

%% ---------------------------------------------------------------------------
%% 12. CABEÇALHO E RODAPÉ — posicionados em coordenadas absolutas, exatamente
%%     como no PDF oficial (as imagens são ancoradas pelo canto superior
%%     esquerdo; as posições vêm do arquivo .docx, em EMU/twips convertidos).
%%     As imagens image1.png e image2.png são procuradas na pasta do .tex.
%% ---------------------------------------------------------------------------
\newcommand{\cabecalhoerodape}{%
  \begin{tikzpicture}[remember picture,overlay]
    % logotipo horizontal (image2): x=42,55  y=25,15  6,31 x 1,98 cm
    \node[anchor=north west,inner sep=0pt,outer sep=0pt]
      at ([xshift=42.55bp,yshift=-25.15bp]current page.north west)
      {\includegraphics[width=178.85bp]{image2.png}};
    % marca da faculdade (image1): x=434,05 y=16,60  4,70 x 2,60 cm
    \node[anchor=north west,inner sep=0pt,outer sep=0pt]
      at ([xshift=434.05bp,yshift=-16.60bp]current page.north west)
      {\includegraphics[width=133.35bp]{image1.png}};
    % número de página: 10 bp, alinhado à direita, base em y=776,88
    \node[anchor=base east,inner sep=0pt,outer sep=0pt]
      at ([xshift=552.77bp,yshift=-776.88bp]current page.north west)
      {\tamdez\thepage};
    % texto do rodapé: 12 bp, centralizado, 2 linhas (bases 790,20 e 804,00)
    \node[anchor=north,inner sep=0pt,outer sep=0pt,text width=510.24bp]
      at ([xshift=297.66bp,yshift=-781.88bp]current page.north west)
      {\centering\corpo\spaceskip=0pt
       Curso Superior de Bacharelado em \pfc@curso{} -
       São Bernardo do Campo - SP - Brasil -
       \pfc@diainicial{} a \pfc@diafinal{} de \pfc@mes{} de \pfc@ano{}\par};
    % caixa branca + versão do modelo (Century Gothic 8 bp), base em y=822,00
    \fill[white] ([xshift=548.25bp,yshift=-810.00bp]current page.north west)
      rectangle ([xshift=584.25bp,yshift=-829.50bp]current page.north west);
    \node[anchor=base west,inner sep=0pt,outer sep=0pt]
      at ([xshift=555.45bp,yshift=-822.00bp]current page.north west)
      {\fontecentury\tamoito\pfc@versao};
  \end{tikzpicture}%
}
\AddToShipoutPictureBG{\cabecalhoerodape}
\pagestyle{empty}

%% ---------------------------------------------------------------------------
%% 13. ELEMENTOS PRÉ-TEXTUAIS DO ARTIGO
%%     \imprimirtitulo · \imprimirautores · \imprimirorientador
%%     ambientes resumo / abstract · \palavraschave · \keywords
%% ---------------------------------------------------------------------------
\newcommand{\imprimirtitulo}{%
  \par{\centering\bfseries\pfc@titulo\par}%
  \linhavazia}
\newcommand{\imprimirautores}{%
  \par\noindent\textbf{\pfc@autores}\par
  \linhavazia}
\newcommand{\imprimirorientador}{%
  \par\noindent\textbf{\pfc@orientador}\par
  \linhavazia}

\NewEnviron{resumo}{%
  \par\noindent\textbf{Resumo}\par
  \linhavazia
  \begingroup\setlength{\parindent}{0pt}\BODY\par\endgroup}
\RenewEnviron{abstract}{%
  \par\noindent\textbf{Abstract}\par
  \linhavazia
  \begingroup\setlength{\parindent}{0pt}\BODY\par\endgroup}

\newcommand{\palavraschave}[1]{%
  \linhavazia\noindent\textbf{Palavras-chave}: #1\par
  \linhavazia}
% No modelo oficial a lista de palavras-chave em inglês é seguida imediatamente
% pela primeira seção: o espaço que separa as duas é o próprio espaço anterior
% do título da seção.  Por isso \keywords NÃO acrescenta a linha em branco
% final (ao contrário de \palavraschave).  Atenção: no modelo oficial o
% dois-pontos de "Keywords:" é negrito (o de "Palavras-chave:" não é).
\newcommand{\keywords}[1]{%
  \linhavazia\noindent\textbf{Keywords:} #1\par}

%% ---------------------------------------------------------------------------
%% 14. FIGURAS, TABELAS E QUADROS
%%     Uso:  \begin{figura}[número]{legenda}{fonte} ... \end{figura}
%%     O número é automático (1, 2, 3, …); use [n] para fixar um número —
%%     é o que o arquivo do modelo oficial faz, para reproduzir a numeração
%%     do documento original (Figura 9, Tabela 8, Quadro 6).
%%     O corpo do ambiente é a imagem ou o tabular; a legenda fica acima e a
%%     fonte abaixo, centralizadas em 11 bp, com o espaçamento do Word.
%%     Colunas: use o tipo C (centralizado, largura já ajustada ao modelo).
%% ---------------------------------------------------------------------------
\newcounter{figura}
\newcounter{tabela}
\newcounter{quadro}
\newlength{\pfc@largcol}
\newcolumntype{C}{>{\centering\arraybackslash}p{\pfc@largcol}}
\newcommand{\pfc@rotulo}[1]{\csname pfc@rotulo@#1\endcsname}
\newcommand{\pfc@rotulo@figura}{Figura}
\newcommand{\pfc@rotulo@tabela}{Tabela}
\newcommand{\pfc@rotulo@quadro}{Quadro}
\newcommand{\pfc@automatico}{\pfc@automatico}
% \pfc@legenda{tipo}{número ou \pfc@automatico}{texto}
\newcommand{\pfc@legenda}[3]{%
  \def\pfc@tempnum{#2}%
  \ifx\pfc@tempnum\pfc@automatico
    \refstepcounter{#1}%
    \edef\pfc@num{\csname the#1\endcsname}%
  \else
    \setcounter{#1}{#2}%
    \edef\pfc@num{#2}%
  \fi
  \caixatitulo{\pfc@rotulo{#1} \pfc@num\ --\ #3}}

\NewEnviron{figura}[3][\pfc@automatico]{%
  \linhavazia
  \pfc@legenda{figura}{#1}{#2}%
  \begingroup\bloco{\makebox[\textwidth][c]{\BODY}}\endgroup
  \caixafonte{#3}%
  \linhavazia}

\NewEnviron{tabela}[3][\pfc@automatico]{%
  \linhavazia
  \pfc@legenda{tabela}{#1}{#2}%
  \begingroup
    \tamonze
    \setlength{\arrayrulewidth}{0.5bp}%
    \arrayrulecolor[gray]{0.498}%
    \setlength{\tabcolsep}{5.4bp}%
    \setbox\strutbox=\hbox{\rule[-2.38bp]{0pt}{12.65bp}}%
    \setlength{\pfc@largcol}{140.2bp}%
    \bloco{\BODY}%
  \endgroup
  \caixafonte{#3}%
  \linhavazia}

\NewEnviron{quadro}[3][\pfc@automatico]{%
  \linhavazia
  \pfc@legenda{quadro}{#1}{#2}%
  \begingroup
    \tamonze
    \setlength{\arrayrulewidth}{0.5bp}%
    \arrayrulecolor{black}%
    \setlength{\tabcolsep}{0bp}%
    \setbox\strutbox=\hbox{\rule[-2.38bp]{0pt}{12.65bp}}%
    \setlength{\pfc@largcol}{150.6bp}%
    \bloco{\makebox[\textwidth][c]{\BODY}}%
  \endgroup
  \caixafonte{#3}%
  \linhavazia}

%% ---------------------------------------------------------------------------
%% 15. LISTAS (NBR 6024: incisos, alíneas e itens)
%%     O modelo oficial usa "1." com \labelindent=18bp, \leftmargin=36bp; esse
%%     passa a ser o padrão de qualquer \begin{enumerate} do documento.
%% ---------------------------------------------------------------------------
\setlist[enumerate]{label=\arabic*.,align=left,labelindent=18bp,labelwidth=9bp,
  labelsep=9bp,leftmargin=36bp,topsep=0pt,partopsep=0pt,parsep=0pt,itemsep=0pt}
\newlist{itens}{enumerate}{1}
\setlist[itens]{label=\arabic*.,align=left,labelindent=18bp,labelwidth=9bp,
  labelsep=9bp,leftmargin=36bp,topsep=0pt,partopsep=0pt,parsep=0pt,itemsep=0pt}
\newlist{alineas}{enumerate}{1}
\setlist[alineas]{label=\alph*),align=left,labelindent=18bp,labelwidth=9bp,
  labelsep=9bp,leftmargin=36bp,topsep=0pt,partopsep=0pt,parsep=0pt,itemsep=0pt}
\newlist{incisos}{enumerate}{1}
\setlist[incisos]{label=\Roman*.,align=left,labelindent=18bp,labelwidth=9bp,
  labelsep=9bp,leftmargin=36bp,topsep=0pt,partopsep=0pt,parsep=0pt,itemsep=0pt}
\setlist[itemize]{topsep=0pt,partopsep=0pt,parsep=0pt,itemsep=0pt}

%% ---------------------------------------------------------------------------
%% 16. CITAÇÃO DIRETA COM MAIS DE TRÊS LINHAS (NBR 10520:2023)
%%     recuo de 4 cm, letra menor que a do texto, espaço simples, sem aspas.
%% ---------------------------------------------------------------------------
\NewEnviron{citacao}{%
  \par
  \begingroup
    \setlength{\parindent}{0pt}%
    \leftskip=113.39bp            % 4 cm
    \tamdez                       % 10 bp / 11,5 bp
    \BODY\par
  \endgroup
  \linhavazia}

%% ---------------------------------------------------------------------------
%% 17. REFERÊNCIAS (ABNT NBR 6023) — a partir de um arquivo .bib
%%     No preâmbulo:  \addbibresource{referencias.bib}
%%     No fim:        \imprimirreferencias
%%     Citações: \citeonline{chave} -> "Autor (ano)"   (sistema autor-data)
%%               \cite{chave}       -> "(Autor, ano)"
%%               \cite[p.~12]{chave} acrescenta a página (NBR 10520).
%% ---------------------------------------------------------------------------
\newcommand{\citeonline}{\textcite}   % nome consagrado pelo abnTeX2
% Espaçamentos da lista de referências (NBR 6023 + modelo oficial):
%   * 14 bp entre referências (espaço "duplo" do Word);
%   * 12 bp entre o título "Referências" e a primeira referência — o biblatex
%     usa \bibinitsep também antes do primeiro item, e o \addvspace interno
%     mantém o maior valor, de modo que 12 bp vale só para a primeira.
\setlength{\bibitemsep}{14bp}
\setlength{\bibinitsep}{12bp}
\setlength{\bibnamesep}{0pt}          % não acrescentar nada entre autores
\setlength{\bibhang}{0pt}             % NBR 6023: sem deslocamento da 1ª linha

% Referências escritas à mão (sem arquivo .bib): página nova, título
% "Referências" sem número, uma referência por \item, alinhadas à esquerda,
% sem recuo de primeira linha, espaço simples dentro de cada referência e
% espaço duplo (14 bp) entre elas — e 12 bp entre o título e a primeira.
\newenvironment{referencias}{%
  \newpage
  \section*{Referências}%
  % O \vspace explícito (12 bp, como no modelo) é necessário porque o LaTeX
  % suprime o \topsep de uma lista que vem logo depois de um título.
  \vspace{12bp}%
  \begin{list}{}{%
    \raggedright
    \setlength{\leftmargin}{0pt}%
    \setlength{\labelwidth}{0pt}%
    \setlength{\labelsep}{0pt}%
    \setlength{\itemindent}{0pt}%
    \setlength{\listparindent}{0pt}%
    \setlength{\rightmargin}{0pt}%
    \setlength{\topsep}{0pt}%
    \setlength{\partopsep}{0pt}%
    \setlength{\parsep}{0pt}%
    \setlength{\itemsep}{14bp}%
  }%
}{\end{list}}

% Referências geradas a partir do arquivo .bib (biblatex-abnt), na mesma
% página e com o mesmo espaçamento do ambiente "referencias" acima.
\newcommand{\imprimirreferencias}{%
  \newpage
  \section*{Referências}%
  \printbibliography[heading=none]}

%% ---------------------------------------------------------------------------
%% Fim da classe.
%% ---------------------------------------------------------------------------
\makeatother
\endinput
