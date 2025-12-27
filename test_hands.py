import cv2
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download the hand landmarker model first
# You can download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task


#Working in 4th/5th octave range
NOTE_FREQS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00,
    'A4': 440.00, 'B4': 493.88, 'C5': 523.25, 'D5': 587.33, 'E5': 659.25
}

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        continue
    
    # Flip for selfie view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    results = detector.detect(mp_image)
    
    # Draw landmarks
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw each landmark
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    
    if results.handedness: 
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
                    print("LEFT: G4")
                elif index_up:
                    print("LEFT: F4")
                elif middle_up:
                    print("LEFT: E4")
                elif ring_up:
                    print("LEFT: D4")
                elif pinky_up:
                    print("LEFT: C4")
                else:
                    print("LEFT: silence")

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
                    print("RIGHT: E5")
                elif ring_up:
                    print("RIGHT: D5")
                elif middle_up:
                    print("RIGHT: C5")
                elif index_up:
                    print("RIGHT: B4")
                elif thumb_up:
                    print("RIGHT: A4")
                else:
                    print("RIGHT: silence")

    cv2.imshow('Hand Tracking', frame)
    
    #Press q to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()