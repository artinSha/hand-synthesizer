import cv2
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download the hand landmarker model first
# You can download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

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
            
            # Draw connections (you'll need to define these)
            # For now, just showing the points
    
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()