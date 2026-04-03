# camera.py

import cv2
import mediapipe as mp
import requests
import numpy as np

# ── Setup ────────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # Only detect one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

BACKEND_URL = "http://127.0.0.1:5000/predict"

# ── Start Camera ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Camera started! Show your hand...")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB (MediaPipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    result = hands.process(rgb_frame)

    letter = ""
    sentence = ""
    confidence = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand skeleton on screen
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 63 values (21 landmarks × x, y, z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            # Send to backend
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"landmarks": landmarks},
                    timeout=0.5
                )
                data = response.json()
                letter = data.get("letter", "")
                confidence = data.get("confidence", 0)
                sentence = data.get("sentence", "")

            except Exception as e:
                print(f"Backend error: {e}")

    # ── Display on screen ─────────────────────────────────────────────────────
    # Background box for text
    cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)

    # Show current letter
    cv2.putText(frame, f"Letter: {letter}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Show confidence
    cv2.putText(frame, f"Confidence: {int(confidence * 100)}%",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    # Show sentence
    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {sentence}",
                (10, 440), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    # Show the camera window
    cv2.imshow("Gesture2Talk", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera stopped!")                                                        