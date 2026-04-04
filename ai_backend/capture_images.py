import cv2
import os

# Create the main dataset folder if it doesn't exist
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Ask which alphabet you want to record right now
label = input("Which alphabet are you recording? (e.g., A, B, C): ").upper()

# Create the subfolder for that specific alphabet
save_path = os.path.join(DATASET_DIR, label)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print(f"\n📷 Recording for '{label}'.")
print("👉 Press 's' to SAVE an image.")
print("👉 Press 'q' to QUIT when done with this letter.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Show the live webcam feed
    cv2.imshow("Data Collection - Press 's' to save, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # If 's' is pressed, save the frame
    if key == ord('s'):
        count += 1
        img_name = os.path.join(save_path, f"{label}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"✅ Saved image {count} for letter {label}")
        
    # If 'q' is pressed, quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {count} images for {label}.")