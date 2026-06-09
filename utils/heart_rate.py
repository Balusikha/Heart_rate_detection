import numpy as np
from scipy.signal import butter, filtfilt

signal_buffer = []

def calculate_bpm(mean_green):

    global signal_buffer

    signal_buffer.append(mean_green)

    if len(signal_buffer) > 300:
        signal_buffer.pop(0)

    if len(signal_buffer) < 150:
        return 0

    signal = np.array(signal_buffer)

    signal = signal - np.mean(signal)

    fs = 30

    low = 0.8
    high = 3.0

    b, a = butter(
        3,
        [low/(fs/2), high/(fs/2)],
        btype="band"
    )

    filtered = filtfilt(b, a, signal)

    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), 1/fs)

    peak = freqs[np.argmax(fft)]

    bpm = peak * 60

    return int(bpm)

def get_signal():
    return signal_buffer