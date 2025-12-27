import cv2
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sounddevice as sd
import numpy as np
import threading

# Working in 4th/5th octave range
NOTE_FREQS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00,
    'A4': 440.00, 'B4': 493.88, 'C5': 523.25, 'D5': 587.33, 'E5': 659.25
}

audio_lock = threading.Lock()
left_freq = None
right_freq = None

SAMPLE_RATE = 44100     # this is like framerate for audio (samples per second)
AMPLITUDE = 0.2         # volume

left_phase = 0.0
right_phase = 0.0


def audio_callback(outdata, frames, time, status):
    """This function is called by sounddevice to fill the audio buffer"""
    global left_freq, right_freq, left_phase, right_phase
    
    if status:
        print(status)
    
    with audio_lock:
        l_freq = left_freq
        r_freq = right_freq
    
    # Generate time array for this chunk
    t = np.arange(frames) / SAMPLE_RATE
    
    output = np.zeros(frames)
    
    # Generate left hand note
    # Currently using a sine wave formula
    if l_freq:
        left_wave = AMPLITUDE * np.sin(2 * np.pi * l_freq * t + left_phase)
        output += left_wave
        # Update phase for next callback (keep it continuous)
        left_phase = (left_phase + 2 * np.pi * l_freq * frames / SAMPLE_RATE) % (2 * np.pi)
    else:
        # Reset phase when note stops (prevents phase accumulation)
        left_phase = 0.0
    
    # Generate right hand note
    if r_freq:
        right_wave = AMPLITUDE * np.sin(2 * np.pi * r_freq * t + right_phase)
        output += right_wave
        # Update phase for next callback
        right_phase = (right_phase + 2 * np.pi * r_freq * frames / SAMPLE_RATE) % (2 * np.pi)
    else:
        # Reset phase when note stops
        right_phase = 0.0
    
    # Write to output buffer (speakers)
    outdata[:, 0] = output

def main():
    global left_freq, right_freq
    
    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Start audio stream
    stream = sd.OutputStream(
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=2048
    )
    stream.start()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    print("Hand Synthesizer Started!")
    print("Left hand: Pinky=C4, Ring=D4, Middle=E4, Index=F4, Thumb=G4")
    print("Right hand: Thumb=A4, Index=B4, Middle=C5, Ring=D5, Pinky=E5")
    print("Press 'q' to quit")
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            # Flip and convert
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            results = detector.detect(mp_image)
            
            # Initialize notes for this frame
            left_note = None
            right_note = None
            
            if results.hand_landmarks and results.handedness:
                for idx, hand in enumerate(results.hand_landmarks):
                    # Get hand type (right, left)
                    hand_type = results.handedness[idx][0].category_name

                    # NOTE: Need to flip handedness logic due to selfie view
                    # The following chunk is actually for Left hand due to the flip
                    if hand_type == "Right":
                        # Get all finger landmarks
                        thumb_tip, thumb_ip = hand[4], hand[3]
                        index_tip, index_pip = hand[8], hand[6]
                        middle_tip, middle_pip = hand[12], hand[10]
                        ring_tip, ring_pip = hand[16], hand[14]
                        pinky_tip, pinky_pip = hand[20], hand[18]
                        
                        # Check each finger (thumb is special - horizontal movement)
                        thumb_up = thumb_tip.x > thumb_ip.x  # Left thumb points left
                        index_up = index_tip.y < index_pip.y
                        middle_up = middle_tip.y < middle_pip.y
                        ring_up = ring_tip.y < ring_pip.y
                        pinky_up = pinky_tip.y < pinky_pip.y
                        
                        # Determine which note (priority: thumb > index > middle > ring > pinky)
                        if thumb_up:
                            left_note = NOTE_FREQS['G4']
                        elif index_up:
                            left_note = NOTE_FREQS['F4']
                        elif middle_up:
                            left_note = NOTE_FREQS['E4']
                        elif ring_up:
                            left_note = NOTE_FREQS['D4']
                        elif pinky_up:
                            left_note = NOTE_FREQS['C4']
                        else:
                            left_note = None

                    elif hand_type == "Left":
                        thumb_tip, thumb_ip = hand[4], hand[3]
                        index_tip, index_pip = hand[8], hand[6]
                        middle_tip, middle_pip = hand[12], hand[10]
                        ring_tip, ring_pip = hand[16], hand[14]
                        pinky_tip, pinky_pip = hand[20], hand[18]
                        
                        thumb_up = thumb_tip.x < thumb_ip.x
                        index_up = index_tip.y < index_pip.y
                        middle_up = middle_tip.y < middle_pip.y
                        ring_up = ring_tip.y < ring_pip.y
                        pinky_up = pinky_tip.y < pinky_pip.y
                        
                        # Priority is reversed for right hand
                        if pinky_up:
                            right_note = NOTE_FREQS['E5']
                        elif ring_up:
                            right_note = NOTE_FREQS['D5']
                        elif middle_up:
                            right_note = NOTE_FREQS['C5']
                        elif index_up:
                            right_note = NOTE_FREQS['B4']
                        elif thumb_up:
                            right_note = NOTE_FREQS['A4']
                        else:
                            right_note = None
                    
                    # Draw landmarks
                    for landmark in hand:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Update audio frequencies (thread-safe)
            with audio_lock:
                left_freq = left_note
                right_freq = right_note
            
            # Display notes status on screen
            left_note_name = [k for k, v in NOTE_FREQS.items() if v == left_note]
            right_note_name = [k for k, v in NOTE_FREQS.items() if v == right_note]
            
            status_left = f"Left: {left_note_name[0] if left_note_name else 'silence'}"
            status_right = f"Right: {right_note_name[0] if right_note_name else 'silence'}"
            
            cv2.putText(frame, status_left, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, status_right, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow('Hand Synthesizer', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        stream.stop()
        stream.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()