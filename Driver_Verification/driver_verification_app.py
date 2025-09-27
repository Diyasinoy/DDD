import face_recognition
import cv2
import pickle
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

# --- Constants ---
DATABASE_FILE = "driver_database.pickle"
DETECTION_MODEL = "hog"

# --- GUI Colors and Fonts ---
COLOR_VERIFIED = "#2ecc71"  # Green
COLOR_UNVERIFIED = "#e74c3c" # Red
COLOR_SCANNING = "#3498db"  # Blue
FONT_LARGE = ("Helvetica", 24, "bold")
FONT_NORMAL = ("Helvetica", 16)

class DriverVerificationApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        print("[INFO] Loading driver database...")
        with open(DATABASE_FILE, "rb") as f:
            self.data = pickle.load(f)

        print("[INFO] Starting video stream...")
        self.video_source = 0  # Use webcam
        self.vid = cv2.VideoCapture(self.video_source)

        # Create GUI elements
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.status_label = tk.Label(window, text="STATUS: SCANNING", font=FONT_LARGE, bg=COLOR_SCANNING, fg="white")
        self.status_label.pack(fill=tk.X, expand=True)

        self.name_label = tk.Label(window, text="", font=FONT_NORMAL)
        self.name_label.pack()
        
        # Set the update interval (in milliseconds)
        self.delay = 15
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        
        if ret:
            # Process the frame for face recognition
            processed_frame, name, status = self.process_frame(frame)
            
            # Update GUI
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(processed_frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.update_status(name, status)

        self.window.after(self.delay, self.update)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame, model=DETECTION_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        current_name = "Unknown"
        status = "UNVERIFIED"

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare face with known faces
            matches = face_recognition.compare_faces(self.data["encodings"], face_encoding)
            
            if True in matches:
                # Find the best match
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                
                current_name = max(counts, key=counts.get)
                status = "VERIFIED"
            
            # Draw rectangle around the face
            top, right, bottom, left = face_location
            color = COLOR_VERIFIED if status == "VERIFIED" else COLOR_UNVERIFIED
            cv_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0)) # Convert hex to BGR
            cv2.rectangle(frame, (left, top), (right, bottom), cv_color, 2)
            cv2.putText(frame, current_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cv_color, 2)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), current_name, status if face_locations else "SCANNING"

    def update_status(self, name, status):
        if status == "VERIFIED":
            self.status_label.config(text="STATUS: VERIFIED", bg=COLOR_VERIFIED)
            self.name_label.config(text=f"Welcome, {name}!")
        elif status == "UNVERIFIED":
            self.status_label.config(text="STATUS: UNAUTHORIZED DRIVER", bg=COLOR_UNVERIFIED)
            self.name_label.config(text="Access Denied")
        else: # SCANNING
            self.status_label.config(text="STATUS: SCANNING", bg=COLOR_SCANNING)
            self.name_label.config(text="Please position your face in the camera")

    def on_closing(self):
        print("[INFO] Closing application...")
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

# Create a window and pass it to the application class
if __name__ == "__main__":
    root = tk.Tk()
    app = DriverVerificationApp(root, "Driver Verification System")
