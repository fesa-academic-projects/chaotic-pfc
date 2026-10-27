#!/usr/bin/env python3
"""Verifica o mecanismo da falha do enlace sem filtro (Secao 4.2).

Tres checagens, com as mesmas sementes e a mesma grade das figuras:

a) o atraso de grupo do canal (201 coeficientes, fase linear) vale 100 amostras e
   e compensado na comparacao, nao no sinal que aciona o receptor;
b) a fracao de energia de ``x1(n)`` do mapa sem filtro acima de ``0,99 pi``;
c) um canal que so atrasa 100 amostras, sem limitar a banda, mantem a
   sincronizacao, o que separa atraso de perda de banda.

Uso::

    python articles/pfc/revisao/scripts/verificacao_canal.py
"""

from __future__ import annotations

import numpy as np

from chaotic_pfc.comms.channel import fir_channel
from chaotic_pfc.comms.receiver import receive
from chaotic_pfc.comms.transmitter import transmit
from chaotic_pfc.config import DEFAULT_CONFIG as CFG
from chaotic_pfc.dynamics.signals import text_message
from chaotic_pfc.dynamics.spectral import psd_normalised

N0 = CFG.comm.transient
DELAY = (CFG.channel.num_taps - 1) // 2
BAND_EDGE = 0.99


def an_error(recovered: np.ndarray, message: np.ndarray, delay: int) -> np.ndarray:
    """Return ``mu * |m_hat - m|`` aligned by the channel delay."""
    return CFG.comm.mu * np.abs(recovered[delay:] - message[:-delay])


def main() -> int:
    """Run the three checks and print the numbers the section quotes."""
    message = text_message(CFG.comm.message_text, CFG.comm.N, bit_period=CFG.comm.message_period)
    sent = transmit(message, mu=CFG.comm.mu, alpha=1.4, beta=0.3, x0=0.0, y0=0.0)

    print(f"(a) canal: {CFG.channel.num_taps} coeficientes, atraso de grupo {DELAY} amostras")
    print(f"    atraso compensado na comparacao: {DELAY} == (num_taps - 1) // 2")

    spec = CFG.spectral
    x1 = sent - CFG.comm.mu * message
    subset = x1[:65536]
    omega, values = psd_normalised(
        subset,
        spec.nfft,
        spec.window_length,
        window=spec.window,
        kaiser_beta=spec.kaiser_beta,
    )
    total = float(np.sum(values))
    above = float(np.sum(values[omega > BAND_EDGE]))
    print(
        f"(b) energia de x1 acima de {BAND_EDGE} pi: {100 * above / total:.2f}% "
        f"({omega.size} bins, {subset.size} amostras)"
    )

    received, _ = fir_channel(sent, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    rng = np.random.default_rng(CFG.seed)
    recovered = receive(
        received, mu=CFG.comm.mu, alpha=1.4, beta=0.3, y0=rng.random(), z0=rng.random()
    )
    err_channel = an_error(recovered, message, DELAY)

    pure = np.concatenate((np.zeros(DELAY), sent[:-DELAY]))
    rng = np.random.default_rng(CFG.seed)
    recovered_pure = receive(
        pure, mu=CFG.comm.mu, alpha=1.4, beta=0.3, y0=rng.random(), z0=rng.random()
    )
    err_pure = an_error(recovered_pure, message, DELAY)

    print(f"(c) canal de banda limitada: maximo {np.max(err_channel[N0:]):.3e} depois de n0={N0}")
    print(
        f"    canal so de atraso: maximo depois de n0={N0}: "
        f"{np.max(err_pure[N0:]):.3e}; depois de n0={N0 + DELAY}: "
        f"{np.max(err_pure[N0 + DELAY :]):.3e}; "
        f"ultimas 10 mil amostras: {np.mean(err_pure[-10_000:]):.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
