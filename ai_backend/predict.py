import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# 1. Load the AI Brain and the Labels
print("🧠 Loading AI model...")
model = tf.keras.models.load_model("gesture_model.h5")
classes = np.load("classes.npy", allow_pickle=True)

# 2. Setup MediaPipe for Live Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False because we are using live video now!
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# 3. Open Webcam
cap = cv2.VideoCapture(0)
print("🎥 Webcam opened. Show your signs! (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally like a mirror (feels more natural)
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the cool skeleton on your hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the 42 2D coordinates (Exactly like our training data)
            pts = []
            for lm in hand_landmarks.landmark:
                pts.extend([lm.x, lm.y]) # ❌ Dropped Z!

            pts = np.array(pts, dtype=np.float32)

            # ✅ Normalization (Must match training EXACTLY)
            pts = pts.reshape(21, 2)
            pts = pts - pts[0]
            pts = pts / (np.max(np.abs(pts)) + 1e-6)
            
            # Reshape it so the AI understands it's looking at 1 image with 42 points
            pts = pts.flatten().reshape(1, 42) 

            # 4. Make the Prediction!
            predictions = model.predict(pts, verbose=0)
            class_index = np.argmax(predictions[0])
            confidence = predictions[0][class_index]
            predicted_letter = classes[class_index]

            # 5. Show the result on the screen if the AI is mostly confident
            if confidence > 0.6: # Only show if it's more than 60% sure
                cv2.putText(frame, f"Sign: {predicted_letter} ({confidence*100:.1f}%)", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show the video
    cv2.imshow("Gesture2Talk - Live Translator", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()