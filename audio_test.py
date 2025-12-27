import sounddevice as sd
import numpy as np

# Test playing a single tone
def play_tone(frequency, duration=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = 0.3 * np.sin(2 * np.pi * frequency * t)
    sd.play(wave, sample_rate)
    sd.wait()

# Test it
play_tone(261.63)  # Play A4 for 1 second