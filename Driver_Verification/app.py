import face_recognition
import pickle
import cv2
import numpy as np
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # This is to allow requests from the HTML file

# --- Constants ---
DATABASE_FILE = "driver_database.pickle"
DETECTION_MODEL = "hog"

# --- Load Database ---
print("[INFO] Loading driver database...")
with open(DATABASE_FILE, "rb") as f:
    data = pickle.load(f)
print("[INFO] Driver database loaded successfully.")

# --- API Endpoint for Verification ---
@app.route('/verify', methods=['POST'])
def verify_driver():
    """
    Receives an image frame from the frontend, performs face recognition,
    and returns the verification status.
    """
    # Get the image data from the request
    if 'image' not in request.json:
        return jsonify({"status": "error", "message": "No image data found"}), 400

    # The image data is a Base64 encoded string
    image_data_url = request.json['image']
    
    # Decode the Base64 string
    # It will look like "data:image/jpeg;base64,..." so we split on ","
    header, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    
    # Convert bytes to an image that OpenCV can use
    image = Image.open(io.BytesIO(image_bytes))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # --- Face Recognition Logic (adapted from your script) ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame, model=DETECTION_MODEL)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    response_status = "SCANNING"
    response_name = "Unknown"

    if not face_locations:
        return jsonify({"status": "SCANNING", "name": "No face detected"})

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        
        # Default to unverified unless a match is found
        response_status = "UNVERIFIED"

        if True in matches:
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            
            response_name = max(counts, key=counts.get)
            response_status = "VERIFIED"
            # If we find one verified person, we can stop and return
            break 
            
    return jsonify({"status": response_status, "name": response_name})

# --- Run the App ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=True)
