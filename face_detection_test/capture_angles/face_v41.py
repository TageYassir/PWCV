# ======================
# POSE GESTURE CONTROL WITH PERSON RECOGNITION
# ======================
import cv2
import numpy as np
import pyautogui
import time
from mediapipe import solutions
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import keyboard
import sys
import os
import json
from datetime import datetime
import shutil
import tempfile


class PoseGestureControl:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Pose PowerPoint Control")

        # ======================
        # APPLICATION SETTINGS
        # ======================
        self.slide_delay = 1.0  # Delay between slide changes
        self.width, self.height = 640, 480  # Camera resolution
        self.pose_threshold = 0.7  # Pose detection confidence
        self.gesture_threshold = 0.5  # Gesture sensitivity

        # Tracking settings
        self.tracking_mode = "face"  # Can be "face" or "body"
        self.tracked_position = None
        self.tracking_history = []
        self.max_history = 5
        self.tracking_confidence = 0
        self.min_face_confidence = 0.5
        self.min_pose_confidence = 0.5

        # Person recognition settings
        self.registered_persons_dir = "registered_persons"
        self.registered_person = None
        self.face_match_threshold = 0.6  # Threshold for face matching
        self.recognition_interval = 2.0  # Seconds between recognition checks
        self.last_recognition_time = 0

        # Create directory if it doesn't exist
        if not os.path.exists(self.registered_persons_dir):
            os.makedirs(self.registered_persons_dir)

        # ======================
        # APPLICATION STATE
        # ======================
        self.interface_visible = True
        self.running = True
        self.control_active = False
        self.show_camera = False
        self.last_gesture_time = 0

        # ======================
        # INITIALIZE COMPONENTS
        # ======================
        self.setup_hotkeys()
        self.setup_camera()
        self.setup_detection_models()
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

    def setup_detection_models(self):
        # Pose detection
        self.mp_pose = solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = solutions.drawing_utils

        # Face detection
        self.mp_face = solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

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
        self.set_status("Please select a registered person to start control")
        self.status_label = tk.Label(
            self.main_frame,
            textvariable=self.status_var,
            font=('Helvetica', 12),
            wraplength=400
        )
        self.status_label.pack(pady=5)

        # Person selection
        self.create_person_selector()

        # Slide delay control
        self.create_delay_control()

        # Control buttons
        self.create_control_buttons()

    def create_person_selector(self):
        person_frame = tk.Frame(self.main_frame)
        person_frame.pack(pady=5)

        tk.Label(person_frame, text="Registered Person:").pack(side=tk.LEFT)

        self.person_var = tk.StringVar()
        self.person_dropdown = ttk.Combobox(person_frame, textvariable=self.person_var, state="readonly")
        self.person_dropdown.pack(side=tk.LEFT, padx=5)
        self.person_dropdown.bind('<<ComboboxSelected>>', self.load_person)

        self.update_person_list()

        self.register_btn = ttk.Button(
            person_frame,
            text="Register New Person",
            command=self.open_registration_app
        )
        self.register_btn.pack(side=tk.LEFT, padx=5)

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
            command=self.start_control,
            state=tk.DISABLED
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

    def update_person_list(self):
        persons = []
        if os.path.exists(self.registered_persons_dir):
            for file in os.listdir(self.registered_persons_dir):
                if file.endswith('.json'):
                    persons.append(file[:-5])  # Remove .json extension

        self.person_dropdown['values'] = persons
        if persons:
            self.person_var.set(persons[0])
            self.load_person()

    def open_registration_app(self):
        # This would launch the registration app
        # In a real implementation, you might use subprocess to run the other script
        self.set_status("Please run the PersonRegistrationApp separately to register a new person")

    # ======================
    # PERSON RECOGNITION METHODS
    # ======================
    def load_person(self, event=None):
        person_id = self.person_var.get()
        if not person_id:
            return

        json_path = os.path.join(self.registered_persons_dir, f"{person_id}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.registered_person = json.load(f)

            # Load reference images
            self.reference_faces = []
            for angle, images in self.registered_person['images'].items():
                for img_info in images[:2]:  # Use first 2 images from each angle
                    img_path = os.path.join(self.registered_persons_dir, img_info['filename'])
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize and convert to grayscale for easier comparison
                            img = cv2.resize(img, (100, 100))
                            self.reference_faces.append(img)

            self.start_btn.config(state=tk.NORMAL)
            self.set_status(f"Loaded {person_id}. You can now start control.")

    def is_registered_person(self, frame, face_detection):
        if not self.registered_person or not face_detection or not self.reference_faces:
            return False

        # Extract current face ROI
        x, y, w, h = face_detection['bbox']
        current_face = frame[y:y + h, x:x + w]
        if current_face.size == 0:
            return False

        # Resize to match reference images
        current_face = cv2.resize(current_face, (100, 100))

        # Compare with reference images
        matches = []
        for ref_face in self.reference_faces:
            # Simple histogram comparison
            hist1 = cv2.calcHist([current_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([ref_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            match = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            matches.append(match)

        # Check if any match is above threshold
        if matches and max(matches) > self.face_match_threshold:
            return True

        return False

    # ======================
    # TRACKING METHODS
    # ======================
    def detect_face(self, image_rgb):
        """Detect faces in the image and return the best candidate"""
        face_results = self.face_detection.process(image_rgb)
        faces = []

        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]

                if confidence > self.min_face_confidence:
                    x = int(bbox.xmin * self.width)
                    y = int(bbox.ymin * self.height)
                    w = int(bbox.width * self.width)
                    h = int(bbox.height * self.height)
                    center = (x + w // 2, y + h // 2)

                    faces.append({
                        'center': center,
                        'bbox': (x, y, w, h),
                        'confidence': confidence
                    })

        # Return the face with highest confidence
        if faces:
            return max(faces, key=lambda x: x['confidence'])
        return None

    def detect_body(self, image_rgb):
        """Detect body pose and return the center of the shoulders"""
        pose_results = self.pose.process(image_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Get shoulder landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Calculate center point between shoulders
            center_x = int((left_shoulder.x + right_shoulder.x) / 2 * self.width)
            center_y = int((left_shoulder.y + right_shoulder.y) / 2 * self.height)

            # Use visibility of shoulders as confidence
            confidence = (left_shoulder.visibility + right_shoulder.visibility) / 2

            if confidence > self.min_pose_confidence:
                return {
                    'center': (center_x, center_y),
                    'confidence': confidence
                }

        return None

    def update_tracking(self, face_detection, body_detection, frame):
        """Update the tracking position based on available detections"""
        # Check if we have a registered person and verify the detected person
        is_registered = True
        current_time = time.time()

        # Only check recognition at intervals to reduce CPU usage
        if (self.registered_person and face_detection and
                current_time - self.last_recognition_time > self.recognition_interval):
            is_registered = self.is_registered_person(frame, face_detection)
            self.last_recognition_time = current_time

        if not is_registered:
            self.set_status("Warning: Unregistered person detected. Only the registered person can control slides.")
            return

        # Prioritize face tracking for the verified person
        if face_detection:
            self.tracked_position = face_detection['center']
            self.tracking_confidence = face_detection['confidence']
            self.tracking_mode = "face"

            # Add to history for smoothing
            self.tracking_history.append(face_detection['center'])
            if len(self.tracking_history) > self.max_history:
                self.tracking_history.pop(0)

        # Fall back to body tracking if no face detected
        elif body_detection:
            self.tracked_position = body_detection['center']
            self.tracking_confidence = body_detection['confidence']
            self.tracking_mode = "body"

            # Add to history for smoothing
            self.tracking_history.append(body_detection['center'])
            if len(self.tracking_history) > self.max_history:
                self.tracking_history.pop(0)

        # If no detection, use prediction based on history
        elif self.tracking_history:
            # Simple prediction: use the average of recent positions
            avg_x = sum(pos[0] for pos in self.tracking_history) / len(self.tracking_history)
            avg_y = sum(pos[1] for pos in self.tracking_history) / len(self.tracking_history)
            self.tracked_position = (int(avg_x), int(avg_y))
            self.tracking_confidence *= 0.9  # Reduce confidence over time

        # Apply smoothing if we have history
        if len(self.tracking_history) > 1:
            avg_x = sum(pos[0] for pos in self.tracking_history) / len(self.tracking_history)
            avg_y = sum(pos[1] for pos in self.tracking_history) / len(self.tracking_history)
            self.tracked_position = (int(avg_x), int(avg_y))

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
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face and body
        face_detection = self.detect_face(image_rgb)
        body_detection = self.detect_body(image_rgb)

        # Update tracking
        self.update_tracking(face_detection, body_detection, frame)

        # Draw tracking information
        if self.tracked_position:
            # Draw tracking point
            cv2.circle(frame, self.tracked_position, 10, (0, 255, 0), -1)

            # Draw detection source
            mode_text = f"Tracking: {self.tracking_mode} ({self.tracking_confidence:.2f})"
            cv2.putText(frame, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show registered person ID if available
            if self.registered_person:
                person_text = f"Person: {self.registered_person['person_id']}"
                cv2.putText(frame, person_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detect gestures for slide control
        self.detect_gestures(frame, image_rgb)

        return frame

    def detect_gestures(self, frame, image_rgb):
        """Detect hand gestures for slide control"""
        pose_results = self.pose.process(image_rgb)

        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            landmarks = pose_results.pose_landmarks.landmark
            left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            eye_level = (left_eye.y + right_eye.y) / 2
            current_time = time.time()

            if current_time - self.last_gesture_time > self.slide_delay:
                if right_wrist.y < eye_level and right_wrist.visibility > 0.5:
                    pyautogui.press('right')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "NEXT SLIDE", (50, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif left_wrist.y < eye_level and left_wrist.visibility > 0.5:
                    pyautogui.press('left')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "PREVIOUS SLIDE", (50, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
        if not self.registered_person:
            self.set_status("Please select a registered person first")
            return

        self.control_active = True
        self.show_camera = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_status(
            "Control active - Raise your hand above eye level to navigate\nLeft hand up: Previous slide\nRight hand up: Next slide")

    def stop_control(self):
        self.control_active = False
        self.show_camera = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.set_status("Control stopped")
        # Reset tracking
        self.tracked_position = None
        self.tracking_history = []
        self.tracking_mode = "face"

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