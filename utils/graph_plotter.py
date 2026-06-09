import matplotlib.pyplot as plt
import numpy as np

plt.ion()

fig, (ax1, ax2) = plt.subplots(2, 1)

def update_graph(signal):

    if len(signal) < 50:
        return

    ax1.clear()
    ax2.clear()

    ax1.plot(signal)
    ax1.set_title("Heart Rate Signal")

    signal_np = np.array(signal)

    fft = np.abs(np.fft.rfft(signal_np))
    freqs = np.fft.rfftfreq(
        len(signal_np),
        1/30
    )

    ax2.plot(freqs, fft)
    ax2.set_title("Frequency Spectrum")

    plt.tight_layout()
    plt.pause(0.01)