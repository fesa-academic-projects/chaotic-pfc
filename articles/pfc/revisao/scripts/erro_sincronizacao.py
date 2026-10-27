#!/usr/bin/env python3
r"""Mede o erro de sincronizacao dos tres cenarios da Secao 4.3.

O criterio e o erro entre o estado do transmissor e o do receptor depois do
transiente, sem usar MSE nem BER. Os tres cenarios usam o mesmo pipeline das
figuras (mesmas sementes, \(10^6\) amostras, \(n_0 = 200\)):

* sem filtro, canal ideal: ``transmit`` + ``receive`` (2-D), sem canal;
* sem filtro, canal de banda limitada: ``transmit`` + ``fir_channel`` +
  ``receive`` (2-D);
* com filtro, canal de banda limitada: ``transmit_order_n`` + ``fir_channel`` +
  ``receive_order_n``, com o filtro Hamming de 9 coeficientes e corte 0,52 pi.

Nos dois primeiros, a mensagem recuperada satisfaz
``m_hat[n] = (r[n] - y1[n]) / mu`` com ``r[n] = x1[n - delay] + mu * m[n - delay]``,
de modo que o erro de estado vale ``|x1 - y1| = mu * |m_hat - m|``. O script
aplica essa identidade em vez de repetir o laco do receptor. No terceiro, o erro
e medido diretamente entre ``state[2]`` do transmissor e do receptor.

Ao final, o script testa uma grade de limiares e mostra a faixa em que a
classificacao sincroniza/nao sincroniza dos tres cenarios nao muda.

Uso::

    python articles/pfc/revisao/scripts/erro_sincronizacao.py
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin

from chaotic_pfc.comms.channel import fir_channel
from chaotic_pfc.comms.receiver import receive, receive_order_n
from chaotic_pfc.comms.transmitter import transmit, transmit_order_n
from chaotic_pfc.config import DEFAULT_CONFIG as CFG
from chaotic_pfc.dynamics.signals import text_message

N0 = CFG.comm.transient
THRESHOLD = 1e-3
FILTER_CUTOFF = 0.52
FILTER_TAPS = 9
DELAY = (CFG.channel.num_taps - 1) // 2


def summary(name: str, error: np.ndarray, start: int) -> float:
    """Print the post-transient statistics of one scenario and return the maximum."""
    tail = np.abs(error[start:])
    below = tail < THRESHOLD
    print(
        f"  {name:<32} max {tail.max():.3e}  mediana {np.median(tail):.3e}  "
        f"abaixo de {THRESHOLD:g} em {100 * below.mean():5.1f}% das amostras"
    )
    return float(tail.max())


def main() -> int:
    """Measure the state error of the three links and the threshold sensitivity."""
    message = text_message(CFG.comm.message_text, CFG.comm.N, bit_period=CFG.comm.message_period)

    sent = transmit(message, mu=CFG.comm.mu, alpha=1.4, beta=0.3, x0=0.0, y0=0.0)
    rng = np.random.default_rng(CFG.seed)
    recovered_ideal = receive(
        sent, mu=CFG.comm.mu, alpha=1.4, beta=0.3, y0=rng.random(), z0=rng.random()
    )
    err_ideal = CFG.comm.mu * np.abs(recovered_ideal - message)

    received, _ = fir_channel(sent, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    rng = np.random.default_rng(CFG.seed)
    recovered = receive(
        received, mu=CFG.comm.mu, alpha=1.4, beta=0.3, y0=rng.random(), z0=rng.random()
    )
    err_2d = CFG.comm.mu * np.abs(recovered[DELAY:] - message[:-DELAY])

    coeffs = firwin(
        numtaps=FILTER_TAPS, cutoff=FILTER_CUTOFF, window="hamming", pass_zero=True, fs=2.0
    )
    rng = np.random.default_rng(CFG.seed)
    x0 = 0.5 * rng.random(len(coeffs))
    y0 = 0.5 * rng.random(len(coeffs))
    sent_f, state_t = transmit_order_n(message, coeffs, mu=CFG.comm.mu, alpha=1.4, beta=0.3, x0=x0)
    received_f, _ = fir_channel(sent_f, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    recovered_f, state_r = receive_order_n(
        received_f, coeffs, mu=CFG.comm.mu, alpha=1.4, beta=0.3, y0=y0
    )
    err_fir = np.abs(state_t[2, :-DELAY] - state_r[2, DELAY:])
    err_msg = CFG.comm.mu * np.abs(recovered_f[DELAY:] - message[:-DELAY])

    print(f"amostras {CFG.comm.N}, n0 {N0}, mu {CFG.comm.mu}, semente {CFG.seed}")
    maxima = {
        "sem filtro, canal ideal": summary("sem filtro, canal ideal", err_ideal, N0),
        "sem filtro, banda limitada": summary("sem filtro, banda limitada", err_2d, N0),
        "com filtro, banda limitada": summary("com filtro, banda limitada", err_fir, N0),
    }
    summary("com filtro, conferencia pela mensagem", err_msg, N0)

    print()
    print("sensibilidade ao limiar (sincroniza quando o maximo e menor que o limiar):")
    low = max(maxima["sem filtro, canal ideal"], maxima["com filtro, banda limitada"])
    high = maxima["sem filtro, banda limitada"]
    for threshold in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        classes = ["sincroniza" if maxima[key] < threshold else "nao sincroniza" for key in maxima]
        stable = classes == ["sincroniza", "nao sincroniza", "sincroniza"]
        print(
            f"  {threshold:8.0e}  ideal {classes[0]:<15} "
            f"limitado {classes[1]:<15} filtrado {classes[2]:<15} "
            f"{'estavel' if stable else 'MUDA'}"
        )
    print(f"  faixa estavel medida: acima de {low:.3e} e abaixo de {high:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
