# debug.py
import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2

model   = tf.keras.models.load_model("best_model.h5")
classes = np.load("classes_v2.npy", allow_pickle=True)
print("Classes order:", classes)

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Webcam chalu — 'Q' dabao band karne ke liye")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    res   = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        pts = np.array([[l.x, l.y, l.z]
              for l in res.multi_hand_landmarks[0].landmark],
              dtype=np.float32)

        # Normalization — EXACTLY wahi jo build_dataset.py mein thi
        pts -= pts[0]
        pts /= (np.max(np.abs(pts)) + 1e-6)

        preds = model.predict(pts.flatten().reshape(1, -1), verbose=0)[0]

        # Top 3 predictions dikhao
        top3_idx = preds.argsort()[-3:][::-1]
        for i, idx in enumerate(top3_idx):
            label = f"#{i+1}: {classes[idx]} ({preds[idx]*100:.1f}%)"
            cv2.putText(frame, label, (10, 40 + i*35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0,255,0) if i==0 else (180,180,180), 2)

    cv2.imshow("Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()