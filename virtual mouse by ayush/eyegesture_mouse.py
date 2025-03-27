import cv2
import dlib
import pyautogui
import numpy as np

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib

# Get screen size
screen_width, screen_height = pyautogui.size()

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Define eye landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def get_eye_center(landmarks, eye_points):
    """Calculate the center of the eye."""
    eye = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in eye_points])
    return np.mean(eye, axis=0).astype(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get eye centers
        left_eye_center = get_eye_center(landmarks, LEFT_EYE)
        right_eye_center = get_eye_center(landmarks, RIGHT_EYE)
        
        # Average eye position
        eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                      (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Normalize coordinates (Assuming 640x480 webcam resolution)
        x_ratio = eye_center[0] / 640
        y_ratio = eye_center[1] / 480
        
        # Map to screen resolution
        screen_x = int(screen_width * x_ratio)
        screen_y = int(screen_height * y_ratio)
        
        # Move mouse
        pyautogui.moveTo(screen_x, screen_y)
        
        # Display eye tracking
        cv2.circle(frame, tuple(left_eye_center), 5, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_eye_center), 5, (0, 255, 0), -1)
        cv2.circle(frame, eye_center, 5, (255, 0, 0), -1)
    
    cv2.imshow("Eye Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
