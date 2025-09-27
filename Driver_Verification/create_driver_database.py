import face_recognition
import pickle
import cv2
import os

# --- Constants ---
AUTHORIZED_DRIVERS_DIR = "authorized_drivers"
DATABASE_FILE = "driver_database.pickle"

print("[INFO] Starting to process authorized driver faces...")
known_encodings = []
known_names = []

# Loop through each person in the authorized drivers directory
for person_name in os.listdir(AUTHORIZED_DRIVERS_DIR):
    person_dir = os.path.join(AUTHORIZED_DRIVERS_DIR, person_name)
    
    # Skip non-directory files
    if not os.path.isdir(person_dir):
        continue

    # Loop through each image of the person
    for filename in os.listdir(person_dir):
        print(f"[INFO] Processing image {filename} for {person_name}...")
        
        # Load the image
        image_path = os.path.join(person_dir, filename)
        image = cv2.imread(image_path)
        # Convert from BGR (OpenCV) to RGB (face_recognition)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        boxes = face_recognition.face_locations(rgb_image, model="hog")
        
        # Compute facial embeddings
        encodings = face_recognition.face_encodings(rgb_image, boxes)
        
        # Store the encodings and names
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save the encodings to a file
print("[INFO] Saving encodings to database file...")
data = {"encodings": known_encodings, "names": known_names}
with open(DATABASE_FILE, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] Driver database created successfully at {DATABASE_FILE}")
