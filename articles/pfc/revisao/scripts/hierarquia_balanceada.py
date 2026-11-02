#!/usr/bin/env python3
"""Eta quadrado parcial em desenho balanceado (revisao do orientador, item 15).

(a) 24 configuracoes de janelas fixas: tipo (4) x janela (6), sem Kaiser.
    A interacao tipo x janela e o residual do modelo aditivo.
(b) 160 configuracoes da Kaiser: beta_K (40) com o tipo como bloco (4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hierarquia import BETAS, FIXED, TYPES, configs, rho


def anova(mat: np.ndarray) -> tuple[float, float, float, float]:
    """SS do primeiro fator, do segundo, do residuo e total."""
    grand = mat.mean()
    a, b = mat.shape
    ss_a = b * float(((mat.mean(axis=1) - grand) ** 2).sum())
    ss_b = a * float(((mat.mean(axis=0) - grand) ** 2).sum())
    total = float(((mat - grand) ** 2).sum())
    return ss_a, ss_b, total - ss_a - ss_b, total


def main() -> int:
    """Imprime o eta quadrado parcial dos dois desenhos balanceados."""
    values = {key: rho(h, True, taps) for key, (h, taps) in configs().items()}
    labels = [label for _, label in FIXED]
    fixed = np.array([[values[(t, label, None)] for label in labels] for t in TYPES])
    ss_t, ss_w, ss_int, ss_tot = anova(fixed)
    print("janelas fixas (24 configuracoes, tipo x janela)")
    print(f"  eta2 tipo   = {ss_t / (ss_t + ss_int):.4f}")
    print(f"  eta2 janela = {ss_w / (ss_w + ss_int):.4f}")
    print(f"  interacao   = {ss_int / ss_tot:.4f} da variacao")
    kaiser = np.array([[values[(t, "Kaiser", beta)] for beta in BETAS] for t in TYPES])
    ss_t, ss_b, ss_err, ss_tot = anova(kaiser)
    print("Kaiser (160 configuracoes, beta_K com tipo como bloco)")
    print(f"  eta2 beta_K = {ss_b / (ss_b + ss_err):.4f}")
    print(f"  eta2 tipo   = {ss_t / (ss_t + ss_err):.4f}")
    print(f"  erro        = {ss_err / ss_tot:.4f} da variacao")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
