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

        # Presenter identification settings
        self.presenter_profile = None
        self.profile_dir = "presenter_profiles"
        self.current_profile_id = None
        self.identification_samples = 5  # Number of samples to capture
        self.identification_delay = 0.5  # Delay between sample captures
        self.face_match_threshold = 0.6  # Threshold for face matching
        self.color_match_threshold = 0.7  # Threshold for color matching

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

        # Face recognition (using face landmarks for simple matching)
        self.mp_face_mesh = solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Holistic model for combined face and pose tracking
        self.mp_holistic = solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
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
            'face_samples': [],
            'clothing_colors': [],
            'typical_position': None
        }
        self.set_status("Please stand in front of the camera for identification")
        self.identifying_presenter = True
        self.samples_captured = 0

    def capture_presenter_samples(self, frame, image_rgb):
        if self.samples_captured >= self.identification_samples:
            self.identifying_presenter = False
            self.save_profile()
            self.set_status(f"Presenter profile created: {self.current_profile_id}")
            self.update_profile_list()
            self.profile_var.set(self.current_profile_id)
            return

        # Capture face sample
        face_detection = self.detect_face(image_rgb)
        if face_detection:
            x, y, w, h = face_detection['bbox']
            face_roi = frame[y:y + h, x:x + w]

            if face_roi.size > 0:
                # Resize to standard size for comparison
                face_roi = cv2.resize(face_roi, (100, 100))
                self.presenter_profile['face_samples'].append(face_roi)

                # Capture clothing color from upper body area
                clothing_y = min(y + h + 50, self.height - 1)
                clothing_h = min(150, self.height - clothing_y - 1)
                if clothing_h > 0:
                    clothing_roi = frame[clothing_y:clothing_y + clothing_h, x:x + w]
                    if clothing_roi.size > 0:
                        avg_color = np.mean(clothing_roi, axis=(0, 1))
                        self.presenter_profile['clothing_colors'].append(avg_color)

                self.samples_captured += 1
                self.set_status(f"Capturing presenter... {self.samples_captured}/{self.identification_samples}")

                # Add a small delay between samples
                time.sleep(self.identification_delay)

    def save_profile(self):
        if not self.presenter_profile:
            return

        # Convert numpy arrays to lists for JSON serialization
        profile_data = self.presenter_profile.copy()

        # Convert face samples to base64
        profile_data['face_samples'] = []
        for sample in self.presenter_profile['face_samples']:
            _, buffer = cv2.imencode('.jpg', sample)
            profile_data['face_samples'].append(buffer.tobytes().hex())

        # Save clothing colors as lists
        profile_data['clothing_colors'] = [color.tolist() for color in self.presenter_profile['clothing_colors']]

        # Save to file
        profile_path = os.path.join(self.profile_dir, f"{self.current_profile_id}.json")
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f)

    def load_profile(self, event=None):
        profile_name = self.profile_var.get()
        if not profile_name:
            return

        profile_path = os.path.join(self.profile_dir, f"{profile_name}.json")
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)

            # Convert back to numpy arrays
            self.presenter_profile = profile_data
            self.presenter_profile['face_samples'] = []
            for sample_hex in profile_data['face_samples']:
                sample_bytes = bytes.fromhex(sample_hex)
                sample_array = np.frombuffer(sample_bytes, dtype=np.uint8)
                sample_img = cv2.imdecode(sample_array, cv2.IMREAD_COLOR)
                self.presenter_profile['face_samples'].append(sample_img)

            self.presenter_profile['clothing_colors'] = [
                np.array(color) for color in profile_data['clothing_colors']
            ]

            self.current_profile_id = profile_name
            self.set_status(f"Loaded profile: {profile_name}")

    def is_presenter(self, frame, face_detection):
        if not self.presenter_profile or not face_detection:
            return False

        # Check if we have any face samples to compare against
        if not self.presenter_profile.get('face_samples'):
            return False  # No samples to compare with

        # Extract current face ROI
        x, y, w, h = face_detection['bbox']
        current_face = frame[y:y + h, x:x + w]
        if current_face.size == 0:
            return False

        # Resize to match stored samples
        current_face = cv2.resize(current_face, (100, 100))

        # Compare with stored samples
        matches = []
        for sample in self.presenter_profile['face_samples']:
            # Simple histogram comparison
            hist1 = cv2.calcHist([current_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([sample], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            match = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            matches.append(match)

        # Check if we have any matches and if the best match is above threshold
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
        # Check if we're identifying the presenter
        if self.identifying_presenter:
            return

        # Check if we have a presenter profile and verify the detected person
        is_presenter = True
        if self.presenter_profile and face_detection:
            # Only check if we have face samples to compare with
            if self.presenter_profile.get('face_samples'):
                is_presenter = self.is_presenter(frame, face_detection)
            else:
                # If no face samples, assume it's the presenter
                is_presenter = True

        if not is_presenter:
            self.set_status("Warning: Unknown person detected. Only the presenter can control slides.")
            return

        # ... rest of the method remains the same

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

        # If we're identifying the presenter, capture samples
        if self.identifying_presenter:
            self.capture_presenter_samples(frame, image_rgb)
            cv2.putText(frame, f"IDENTIFYING: {self.samples_captured}/{self.identification_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

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