import cv2
import mediapipe as mp
import os
import csv

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATASET_PATH = "dataset"
OUTPUT_FILE = "landmarks.csv"

# Check if folder exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: '{DATASET_PATH}' folder not found!")
    exit()

print("🚀 Starting conversion... This will take a few minutes.")

with open(OUTPUT_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Loop through each folder (A, B, C...)
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        
        if not os.path.isdir(label_path):
            continue
            
        print(f"Processing letter: {label}")
        
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z]) # 21 points * 3 (x,y,z) = 63 values
                    row.append(label)
                    writer.writerow(row)

print("✅ Success! landmarks.csv created.")