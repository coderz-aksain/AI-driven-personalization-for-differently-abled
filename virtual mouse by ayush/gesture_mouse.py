import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)  # Increased accuracy

# Get screen size
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# Gesture accuracy tracking
correct_actions = 0
total_gestures = 0

# Cursor smoothing parameters
alpha = 0.2  # Smoothing factor
prev_x, prev_y = 0, 0

# Gesture cooldown timers
last_click_time = 0
last_scroll_time = 0

# Instructions for user
instructions = [
    "Virtual Mouse Controls:",
    "Move Index Finger → Move Mouse",
    "Index & Thumb Close → Left Click",
    "Middle Finger & Thumb Close → Right Click",
    "Move Index Finger Up → Scroll Up",
    "Move Index Finger Down → Scroll Down",
    "Index & Middle Fingers Close → Zoom In",
    "Index & Middle Fingers Apart → Zoom Out",
    "Press 'Q' to Exit"
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    total_gestures += 1  # Count total gestures

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[8]  # Index Finger Tip
            thumb_tip = landmarks[4]  # Thumb Tip
            middle_finger_tip = landmarks[12]  # Middle Finger Tip

            # Convert to screen coordinates
            x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

            # Apply cursor smoothing using Exponential Moving Average (EMA)
            smoothed_x = int(alpha * x + (1 - alpha) * prev_x)
            smoothed_y = int(alpha * y + (1 - alpha) * prev_y)

            # Move Mouse
            if abs(smoothed_x - prev_x) > 3 or abs(smoothed_y - prev_y) > 3:
                pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.08)  # Smoother movement

            prev_x, prev_y = smoothed_x, smoothed_y  # Update previous position

            # Left Click: Thumb & Index Finger Close (Adjusted Distance)
            thumb_index_distance = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y])
            )
            if thumb_index_distance < 0.04 and (time.time() - last_click_time) > 0.5:  # Avoid false triggers
                pyautogui.click()
                correct_actions += 1
                last_click_time = time.time()  # Cooldown

            # Right Click: Thumb & Middle Finger Close (Adjusted Distance)
            middle_thumb_distance = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_finger_tip.x, middle_finger_tip.y])
            )
            if middle_thumb_distance < 0.04 and (time.time() - last_click_time) > 0.5:
                pyautogui.rightClick()
                correct_actions += 1
                last_click_time = time.time()  # Cooldown

            # Scroll Up: Move Index Finger Up
            if index_finger_tip.y < landmarks[6].y - 0.05 and (time.time() - last_scroll_time) > 0.3:
                pyautogui.scroll(10)
                correct_actions += 1
                last_scroll_time = time.time()

            # Scroll Down: Move Index Finger Down
            elif index_finger_tip.y > landmarks[6].y + 0.05 and (time.time() - last_scroll_time) > 0.3:
                pyautogui.scroll(-10)
                correct_actions += 1
                last_scroll_time = time.time()

            # Zoom In: Index & Middle Fingers Close
            if abs(landmarks[8].x - landmarks[12].x) < 0.04 and abs(landmarks[8].y - landmarks[12].y) < 0.04:
                pyautogui.hotkey("ctrl", "+")
                correct_actions += 1
                time.sleep(0.3)

            # Zoom Out: Index & Middle Fingers Apart
            if abs(landmarks[8].x - landmarks[12].x) > 0.1:
                pyautogui.hotkey("ctrl", "-")
                correct_actions += 1
                time.sleep(0.3)

    # Calculate accuracy
    accuracy = (correct_actions / total_gestures) * 100 if total_gestures > 0 else 0

    # Display instructions on screen
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display accuracy on screen
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show webcam feed
    cv2.imshow("Gesture-Based Virtual Mouse (High Accuracy)", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
