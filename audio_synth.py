import sounddevice as sd
import numpy as np
import threading

class AudioSynthesizer:
    def __init__(self, sample_rate=44100, amplitude=0.2, blocksize=2048):
        """Initialize the audio synthesizer"""
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.blocksize = blocksize
        
        # Shared state (thread-safe)
        self.audio_lock = threading.Lock()
        self.left_freq = None
        self.right_freq = None
        
        # Phase tracking for smooth audio
        self.left_phase = 0.0
        self.right_phase = 0.0
        
        # Audio stream
        self.stream = None
    
    def _audio_callback(self, outdata, frames, time, status):
        """This function is called by sounddevice to fill the audio buffer"""
        if status:
            print(status)
        
        # Get current frequencies (thread-safe)
        with self.audio_lock:
            l_freq = self.left_freq
            r_freq = self.right_freq
        
        # Generate time array for this chunk
        t = np.arange(frames) / self.sample_rate
        
        # Initialize output
        output = np.zeros(frames)
        
        # Generate left hand note
        if l_freq:
            left_wave = self.amplitude * np.sin(2 * np.pi * l_freq * t + self.left_phase)
            output += left_wave
            # Update phase for next callback
            self.left_phase = (self.left_phase + 2 * np.pi * l_freq * frames / self.sample_rate) % (2 * np.pi)
        else:
            self.left_phase = 0.0
        
        # Generate right hand note
        if r_freq:
            right_wave = self.amplitude * np.sin(2 * np.pi * r_freq * t + self.right_phase)
            output += right_wave
            # Update phase for next callback
            self.right_phase = (self.right_phase + 2 * np.pi * r_freq * frames / self.sample_rate) % (2 * np.pi)
        else:
            self.right_phase = 0.0
        
        # Write to output buffer
        outdata[:, 0] = output
    
    def start(self):
        """Start the audio stream"""
        self.stream = sd.OutputStream(
            channels=1,
            callback=self._audio_callback,
            samplerate=self.sample_rate,
            blocksize=self.blocksize
        )
        self.stream.start()
        print("Audio synthesizer started")
    
    def stop(self):
        """Stop the audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Audio synthesizer stopped")
    
    def set_notes(self, left_freq, right_freq):
        """Update the frequencies being played (thread-safe)"""
        with self.audio_lock:
            self.left_freq = left_freq
            self.right_freq = right_freq
    
    def set_left_note(self, freq):
        """Update just the left hand frequency"""
        with self.audio_lock:
            self.left_freq = freq
    
    def set_right_note(self, freq):
        """Update just the right hand frequency"""
        with self.audio_lock:
            self.right_freq = freq