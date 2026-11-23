# ======================
# IMPORTS AND LIBRARIES
# ======================
import cv2
import numpy as np
import pyautogui
import time
from mediapipe import solutions
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import keyboard
import sys
import os
import json
from datetime import datetime
import face_recognition  # For face recognition


# ======================
# MAIN APPLICATION CLASS
# ======================
class PoseGestureControl:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Pose PowerPoint Control")

        # ======================
        # APPLICATION SETTINGS
        # ======================
        self.slide_delay = 1.0  # Delay between slide changes
        self.width, self.height = 640, 480  # Camera resolution

        # Presenter identification settings
        self.presenter_profile = None
        self.profile_dir = "presenter_profiles"
        self.current_profile_id = None
        self.identification_samples = 30  # Increased to 30 samples
        self.identification_delay = 0.1  # Reduced delay for faster capture
        self.face_match_threshold = 0.6  # Threshold for face matching

        # Face recognition
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_this_frame = True

        # Create profile directory if it doesn't exist
        if not os.path.exists(self.profile_dir):
            os.makedirs(self.profile_dir)

        # ======================
        # APPLICATION STATE
        # ======================
        self.interface_visible = True
        self.running = True
        self.control_active = False
        self.show_camera = False
        self.last_gesture_time = 0
        self.identifying_presenter = False
        self.samples_captured = 0
        self.presenter_detected = False
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # Time to wait before considering presenter lost

        # ======================
        # INITIALIZE COMPONENTS
        # ======================
        self.setup_hotkeys()
        self.setup_camera()
        self.create_widgets()
        self.start_video_thread()

    # ======================
    # SETUP METHODS
    # ======================
    def setup_hotkeys(self):
        keyboard.add_hotkey('f12', self.toggle_interface)
        keyboard.add_hotkey('shift+f12', self.quit_application)

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

    # ======================
    # GUI METHODS
    # ======================
    def create_widgets(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Video display
        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(pady=10)
        self.video_label.pack_forget()

        # Status label
        self.status_var = tk.StringVar()
        self.set_status("Ready to start control")
        self.status_label = tk.Label(
            self.main_frame,
            textvariable=self.status_var,
            font=('Helvetica', 12),
            wraplength=400
        )
        self.status_label.pack(pady=5)

        # Profile selection
        self.create_profile_selector()

        # Slide delay control
        self.create_delay_control()

        # Control buttons
        self.create_control_buttons()

    def create_profile_selector(self):
        profile_frame = tk.Frame(self.main_frame)
        profile_frame.pack(pady=5)

        tk.Label(profile_frame, text="Presenter Profile:").pack(side=tk.LEFT)

        self.profile_var = tk.StringVar()
        self.profile_dropdown = ttk.Combobox(profile_frame, textvariable=self.profile_var)
        self.profile_dropdown.pack(side=tk.LEFT, padx=5)
        self.profile_dropdown.bind('<<ComboboxSelected>>', self.load_profile)

        self.update_profile_list()

        self.new_profile_btn = ttk.Button(
            profile_frame,
            text="New Profile",
            command=self.create_new_profile
        )
        self.new_profile_btn.pack(side=tk.LEFT, padx=5)

    def create_delay_control(self):
        delay_frame = tk.Frame(self.main_frame)
        delay_frame.pack(pady=5)

        tk.Label(delay_frame, text="Slide Delay:").pack(side=tk.LEFT)
        self.delay_slider = tk.Scale(
            delay_frame,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_slide_delay
        )
        self.delay_slider.set(self.slide_delay)
        self.delay_slider.pack(side=tk.LEFT, padx=5)

    def create_control_buttons(self):
        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(
            btn_frame,
            text="Start Control",
            command=self.start_control
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            btn_frame,
            text="Stop Control",
            command=self.stop_control,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update()

    def update_profile_list(self):
        profiles = []
        if os.path.exists(self.profile_dir):
            for file in os.listdir(self.profile_dir):
                if file.endswith('.json'):
                    profiles.append(file[:-5])  # Remove .json extension

        self.profile_dropdown['values'] = profiles
        if profiles:
            self.profile_var.set(profiles[0])
            self.load_profile()

    # ======================
    # PRESENTER IDENTIFICATION METHODS
    # ======================
    def create_new_profile(self):
        profile_name = f"presenter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_profile_id = profile_name
        self.presenter_profile = {
            'id': profile_name,
            'created': datetime.now().isoformat(),
            'face_encodings': [],
        }
        self.set_status("Please stand in front of the camera for identification")
        self.identifying_presenter = True
        self.samples_captured = 0

    def capture_presenter_samples(self, frame):
        if self.samples_captured >= self.identification_samples:
            self.identifying_presenter = False
            self.save_profile()
            self.set_status(f"Presenter profile created: {self.current_profile_id}")
            self.update_profile_list()
            self.profile_var.set(self.current_profile_id)
            return

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all faces and face encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if face_encodings:
            # Use the first face found
            self.presenter_profile['face_encodings'].append(face_encodings[0].tolist())
            self.samples_captured += 1
            self.set_status(f"Capturing presenter... {self.samples_captured}/{self.identification_samples}")

            # Add a small delay between samples
            time.sleep(self.identification_delay)

    def save_profile(self):
        if not self.presenter_profile:
            return

        # Save to file
        profile_path = os.path.join(self.profile_dir, f"{self.current_profile_id}.json")
        with open(profile_path, 'w') as f:
            json.dump(self.presenter_profile, f)

    def load_profile(self, event=None):
        profile_name = self.profile_var.get()
        if not profile_name:
            return

        profile_path = os.path.join(self.profile_dir, f"{profile_name}.json")
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)

            # Load face encodings
            self.known_face_encodings = [np.array(encoding) for encoding in profile_data.get('face_encodings', [])]
            self.known_face_names = [profile_name] * len(self.known_face_encodings)

            self.current_profile_id = profile_name
            self.set_status(f"Loaded profile: {profile_name}")

    def recognize_presenter(self, frame):
        """Recognize if the current face matches the presenter using face encodings"""
        if not self.known_face_encodings:
            return False

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all faces and face encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < self.face_match_threshold:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Check if presenter is detected
        return self.current_profile_id in face_names

    # ======================
    # CORE FUNCTIONALITY
    # ======================
    def start_video_thread(self):
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def update_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                if self.control_active:
                    frame = self.process_control_mode(frame)

                if self.show_camera:
                    self.update_display(frame)
                else:
                    self.video_label.pack_forget()

            time.sleep(0.03)  # Reduce CPU usage
            self.root.update()

    def process_control_mode(self, frame):
        # If we're identifying the presenter, capture samples
        if self.identifying_presenter:
            self.capture_presenter_samples(frame)
            cv2.putText(frame, f"IDENTIFYING: {self.samples_captured}/{self.identification_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        # Check for presenter
        current_time = time.time()

        # Only check for presenter every few frames to reduce processing load
        if current_time - self.last_detection_time > 0.5:  # Check every 0.5 seconds
            self.last_detection_time = current_time
            is_presenter = self.recognize_presenter(frame)

            if is_presenter:
                self.presenter_detected = True
                self.set_status("Presenter detected - Control active")
            else:
                # If presenter was previously detected but not now, wait for timeout
                if self.presenter_detected and current_time - self.last_detection_time > self.detection_timeout:
                    self.presenter_detected = False
                    self.set_status("Presenter not detected - Control paused")

        # Draw presenter status
        status_text = "Presenter: " + ("Detected" if self.presenter_detected else "Not Detected")
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if self.presenter_detected else (0, 0, 255), 2)

        # Only detect gestures if presenter is detected
        if self.presenter_detected:
            self.detect_gestures(frame)

        return frame

    def detect_gestures(self, frame):
        """Detect hand gestures for slide control using face position"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all faces
        face_locations = face_recognition.face_locations(rgb_small_frame)

        current_time = time.time()

        if face_locations and current_time - self.last_gesture_time > self.slide_delay:
            # Get the first face found
            top, right, bottom, left = face_locations[0]

            # Scale back up face locations since the frame was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Calculate face center
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2

            # Draw a circle at the center
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)

            # Get hand positions (simplified approach)
            # For a real implementation, you would use hand detection
            frame_height, frame_width, _ = frame.shape

            # Simple gesture detection based on face position
            if face_center_x < frame_width // 3:  # Left side of screen
                pyautogui.press('left')
                self.last_gesture_time = current_time
                cv2.putText(frame, "PREVIOUS SLIDE", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif face_center_x > 2 * frame_width // 3:  # Right side of screen
                pyautogui.press('right')
                self.last_gesture_time = current_time
                cv2.putText(frame, "NEXT SLIDE", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def update_display(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.video_label.pack()

    # ======================
    # CONTROL METHODS
    # ======================
    def start_control(self):
        # Check if we have a presenter profile
        if not self.presenter_profile:
            # Create a new profile automatically
            self.create_new_profile()
            self.set_status("Creating new presenter profile. Please stand in front of the camera.")
        else:
            self.control_active = True
            self.show_camera = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.set_status(
                "Control active - Move your face to the left or right to navigate\nLeft side: Previous slide\nRight side: Next slide")

    def stop_control(self):
        self.control_active = False
        self.show_camera = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.set_status("Control stopped")
        self.presenter_detected = False

    def update_slide_delay(self, value):
        self.slide_delay = float(value)

    # ======================
    # APPLICATION CONTROL
    # ======================
    def toggle_interface(self):
        if self.interface_visible:
            self.root.withdraw()
        else:
            self.root.deiconify()
        self.interface_visible = not self.interface_visible

    def quit_application(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
        sys.exit()

    def on_closing(self):
        self.quit_application()


# ======================
# APPLICATION ENTRY POINT
# ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = PoseGestureControl(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()