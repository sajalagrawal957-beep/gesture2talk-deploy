import mediapipe as mp
import cv2, csv, os, numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATASET_DIR = "dataset"   # check your folder name
OUTPUT = "landmarks.csv"

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)

    for label in sorted(os.listdir(DATASET_DIR)):

        # ❌ skip useless folders
        if label.lower() in ["nothing", "space", "del"]:
            continue

        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue

        print(f"Processing {label}...")

        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)

            img = cv2.imread(path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if not result.multi_hand_landmarks:
                continue

            pts = []
            for lm in result.multi_hand_landmarks[0].landmark:
                # ❌ Dropped lm.z, only extracting x and y for better real-life accuracy
                pts.extend([lm.x, lm.y])

            pts = np.array(pts, dtype=np.float32)

            # ✅ normalization (VERY IMPORTANT - Updated for 2D)
            pts = pts.reshape(21, 2)
            pts = pts - pts[0]
            pts = pts / (np.max(np.abs(pts)) + 1e-6)
            pts = pts.flatten()

            # ✅ write clean row
            writer.writerow(list(pts) + [label])

print("✅ Clean landmarks.csv created")