import sounddevice as sd
import numpy as np
import threading

class AudioSynthesizer:
    def __init__(self, sample_rate=44100, amplitude=0.2, blocksize=2048, glide_time=0.3):
        """Initialize the audio synthesizer
        
            Args:
            sample_rate: Audio sample rate (Hz)
            amplitude: Volume level (0.0 to 1.0)
            blocksize: Audio buffer size
            glide_time: Time to glide between notes (seconds)
        """
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.blocksize = blocksize
        self.glide_time = glide_time
        
        # Shared state (thread-safe)
        self.audio_lock = threading.Lock()
        self.left_freq = None
        self.right_freq = None
        
        # Phase tracking for smooth audio
        self.left_phase = 0.0
        self.right_phase = 0.0
        
        self.current_left_freq = None
        self.current_right_freq = None

        # Audio stream
        self.stream = None
    
    def _audio_callback(self, outdata, frames, time, status):
        """This function is called by sounddevice to fill the audio buffer"""
        if status:
            print(status)
        
        # Get current frequencies (thread-safe)
        with self.audio_lock:
            target_left = self.left_freq
            target_right = self.right_freq
        
        # Generate time array for this chunk
        t = np.arange(frames) / self.sample_rate
        
        # Initialize output
        output = np.zeros(frames)
        
        # GENERATE LEFT HAND SOUND 
        if target_left:
            # If no current frequency, jump to target immediately
            if self.current_left_freq is None:
                self.current_left_freq = target_left
            
            # Calculate glide rate (Hz per second)
            freq_diff = target_left - self.current_left_freq
            glide_rate = freq_diff / self.glide_time
            
            # Generate frequency array that smoothly glides
            freq_change_per_sample = glide_rate / self.sample_rate
            freq_array = self.current_left_freq + freq_change_per_sample * np.arange(frames)
            
            if freq_diff > 0:
                freq_array = np.minimum(freq_array, target_left)
            else:
                freq_array = np.maximum(freq_array, target_left)
            
            # Generate wave with gliding frequency
            # Using instantaneous frequency
            phase_increments = 2 * np.pi * freq_array / self.sample_rate
            phases = self.left_phase + np.cumsum(phase_increments)
            left_wave = self.amplitude * np.sin(phases)
            
            output += left_wave
            
            # Update phase and current frequency
            self.left_phase = phases[-1] % (2 * np.pi)
            self.current_left_freq = freq_array[-1]
            
        else:
            # No note - reset
            self.current_left_freq = None
            self.left_phase = 0.0
        
        #GENERATE RIGHT HAND SOUND 
        if target_right:
            # If no current frequency, jump to target immediately
            if self.current_right_freq is None:
                self.current_right_freq = target_right
            
            # Calculate glide rate (Hz per second)
            freq_diff = target_right - self.current_right_freq
            glide_rate = freq_diff / self.glide_time
            
            # Generate frequency array that smoothly glides
            freq_change_per_sample = glide_rate / self.sample_rate
            freq_array = self.current_right_freq + freq_change_per_sample * np.arange(frames)
            
            # Clamp to target (don't overshoot)
            if freq_diff > 0:
                freq_array = np.minimum(freq_array, target_right)
            else:
                freq_array = np.maximum(freq_array, target_right)
            
            # Generate wave with gliding frequency
            phase_increments = 2 * np.pi * freq_array / self.sample_rate
            phases = self.right_phase + np.cumsum(phase_increments)
            right_wave = self.amplitude * np.sin(phases)
            
            output += right_wave
            
            # Update phase and current frequency
            self.right_phase = phases[-1] % (2 * np.pi)
            self.current_right_freq = freq_array[-1]
            
        else:
            # No note - reset
            self.current_right_freq = None
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