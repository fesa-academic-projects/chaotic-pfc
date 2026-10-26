#!/usr/bin/env python3
"""Confere o estilo ABNT vendorizado (NBR 6023:2025) contra o PDF de teste.

Uso:
    python3 tools/conferir_norma.py [--pdf teste_abnt2025.pdf] [--log ...]

O texto do PDF é extraído com o Ghostscript (`gs -sDEVICE=txtwrite`), então o
conferidor não depende de PyMuPDF e roda também na CI.  Ele confere:

  0. o estilo usado é o vendorizado em `biblatex-abnt/` (lido do .log);
  1. [8.1.1.2] quatro autores → todos indicados na lista de referências;
  2. [10520]   a citação continua com "et al." a partir de quatro autores;
  3. [7.7.6]   DOI → "DOI: https://doi.org/..." (endereço completo);
  4. [8.4.4]   local desconhecido → "[S. l.]" (itálico, com espaço);
  5. [8.4]     local real preservado (não vira [S. l.]);
  6. [8.6.1.4] intervalos com hífen (p. 302-314; 2018-2022), sem travessão;
  7. [7.7.6]   número de artigo (campo `eid`) sem o "p.";
  8. [8.13]    url diferente do DOI continua impressa.

Sai com código 1 se qualquer item falhar - é o que garante, na CI, que o estilo
vendorizado está sendo usado e que os ajustes de 2025 continuam valendo.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# (cláusula, descrição, exige?, alvo, onde)
#   exige?  = "tem" (o alvo precisa aparecer) ou "nao_tem"
#   onde    = "bruto" (texto com espaços, como o gs escreve) ou "sem" (sem espaços)
CHECAGENS = [
    (
        "8.1.1.2",
        "quatro autores: todos indicados na lista",
        "tem",
        "CHATTERJEE,Kiron;MARTIN,Adam;DAVIS,Adrian",
        "sem",
    ),
    ("10520", "citação continua com et al.", "tem", "Clarketal.", "sem"),
    ("7.7.6", "DOI como endereço completo", "tem", "DOI:https://doi.org/", "sem"),
    ("7.7.6", "DOI não sai na forma curta", "nao_tem", "DOI:10.", "sem"),
    ("8.4.4", "[S. l.] com espaço (forma da norma)", "tem", "[S. l.]", "bruto"),
    ("8.4", "local real preservado (Belo Horizonte)", "tem", "BeloHorizonte", "sem"),
    ("8.4", "local real preservado (London)", "tem", "London", "sem"),
    ("8.6.1.4", "intervalo de páginas com hífen", "tem", "p.302-314", "sem"),
    ("8.6.1.4", "intervalo de anos com hífen", "tem", "2018-2022", "sem"),
    ("8.6.1.4", "sem travessão de intervalo", "nao_tem", "\u2013", "bruto"),
    ("7.7.6", "número de artigo (eid) sem o p.", "tem", "e39114141", "sem"),
    ("7.7.6", "número de artigo não sai como página", "nao_tem", "p.e39114141", "sem"),
    ("8.13", "url diferente do DOI continua impressa", "tem", "tandfonline.com", "sem"),
]

# A contagem de "sine loco" é conferida à parte: o gs às vezes come o espaço
# depois do ponto ("[S.l.]"), então o teste do espaço vale para o texto bruto e
# a contagem para o texto sem espaços.
SINELOCO_ESPERADO = 3  # ACHEAMPONG e CLARK (literal no .bib) + GRI (sem local)


def texto_do_pdf(pdf: Path, gs: str) -> str:
    """Extrai o texto de *pdf* com Ghostscript."""
    r = subprocess.run(  # noqa: S603
        [
            gs,
            "-q",
            "-dNOSAFER",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=txtwrite",
            "-sOutputFile=-",
            str(pdf),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        print(f"ERRO: não consegui extrair o texto de {pdf} com o Ghostscript")
        print(r.stderr.strip()[:800])
        sys.exit(1)
    return r.stdout


def main(argv: list[str] | None = None) -> int:
    """Executa as checagens da norma e devolve o código de saída."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf", default="teste_abnt2025.pdf", help="PDF do documento de teste")
    ap.add_argument("--log", default=None, help=".log do XeLaTeX (padrão: mesmo nome do PDF)")
    ap.add_argument("--gs", default="gs", help="executável do Ghostscript")
    args = ap.parse_args(argv)

    pdf = Path(args.pdf)
    log = Path(args.log) if args.log else pdf.with_suffix(".log")
    if not pdf.exists():
        print(f"ERRO: {pdf} não existe - rode `make conferir-norma` (ou `make teste`) antes")
        return 1

    texto = texto_do_pdf(pdf, args.gs)
    bruto = " ".join(texto.split())
    sem_espaco = re.sub(r"\s+", "", texto)

    falhas = 0
    print(f"PDF conferido: {pdf}")

    # Qual abnt.bbx o XeLaTeX carregou?  O log registra o caminho entre
    # parênteses; o do TeX Live contém "texmf-dist", o vendorizado não.
    if not log.exists():
        print("ERRO: .log não encontrado, não dá para conferir qual estilo foi usado")
        falhas += 1
    else:
        conteudo = log.read_text(encoding="utf-8", errors="replace")
        caminhos = re.findall(r"\(([^()\s]*abnt\.bbx)", conteudo)
        vendorizado = [c for c in caminhos if "texmf-dist" not in c]
        if not caminhos:
            print("ERRO: o .log não diz qual abnt.bbx foi carregado")
            falhas += 1
        elif not vendorizado:
            print(
                f"ERRO: o estilo vendorizado em biblatex-abnt/ NÃO foi usado (log: {caminhos[0]})"
            )
            falhas += 1
        else:
            print(f"estilo: vendorizado ({vendorizado[0]})")

    for clausula, descricao, exige, alvo, onde in CHECAGENS:
        haystack = bruto if onde == "bruto" else sem_espaco
        achou = alvo in haystack
        ok = achou if exige == "tem" else not achou
        if not ok:
            falhas += 1
        verbo = "tem" if exige == "tem" else "não tem"
        print(
            f"  [{'ok ' if ok else 'FALHA'}] {clausula:8s} {descricao} (esperado {verbo}: {alvo!r})"
        )

    quantos = sem_espaco.count("[S.l.]")
    ok = quantos == SINELOCO_ESPERADO
    if not ok:
        falhas += 1
    print(
        f"  [{'ok ' if ok else 'FALHA'}] 8.4.4    sine loco nas 3 referências sem local "
        f"(esperado {SINELOCO_ESPERADO}, achou {quantos})"
    )

    if falhas:
        print(f"\nRESULTADO: {falhas} item(ns) fora da NBR 6023:2025 (veja acima).")
        return 1
    print("\nRESULTADO: o estilo vendorizado aplica os pontos conferidos da NBR 6023:2025.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
