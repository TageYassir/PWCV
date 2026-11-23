# ======================
# IMPORTS AND LIBRARIES
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
        self.profile_dir = tempfile.mkdtemp(prefix="pose_control_", dir=".")
        self.current_profile_id = None
        self.identification_samples = 10  # Number of initial samples to capture
        self.continuous_capture = True  # Continuously capture presenter data
        self.capture_interval = 2.0  # Seconds between continuous captures
        self.last_capture_time = 0
        self.face_match_threshold = 0.6  # Threshold for face matching

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

        # Slide delay control
        self.create_delay_control()

        # Control buttons
        self.create_control_buttons()

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

    # ======================
    # PRESENTER IDENTIFICATION METHODS
    # ======================
    def create_presenter_profile(self):
        profile_name = f"presenter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_profile_id = profile_name
        self.presenter_profile = {
            'id': profile_name,
            'created': datetime.now().isoformat(),
            'face_samples': [],
            'sample_count': 0,
            'last_updated': datetime.now().isoformat()
        }
        self.set_status("Creating presenter profile. Please stand in front of the camera.")
        self.identifying_presenter = True
        self.samples_captured = 0

    def capture_presenter_samples(self, frame, image_rgb):
        if self.identifying_presenter and self.samples_captured >= self.identification_samples:
            self.identifying_presenter = False
            self.save_profile()
            self.set_status(f"Presenter profile created. Continuous capture enabled.")
            return

        # Continuous capture logic
        current_time = time.time()
        if (self.continuous_capture and self.presenter_profile and
                current_time - self.last_capture_time > self.capture_interval):
            self.capture_sample(frame, image_rgb)
            self.last_capture_time = current_time

        # Initial identification capture
        elif self.identifying_presenter:
            self.capture_sample(frame, image_rgb)
            time.sleep(0.2)  # Small delay between initial samples

    def capture_sample(self, frame, image_rgb):
        face_detection = self.detect_face(image_rgb)
        if face_detection:
            x, y, w, h = face_detection['bbox']
            face_roi = frame[y:y + h, x:x + w]

            if face_roi.size > 0:
                # Resize to standard size for comparison
                face_roi = cv2.resize(face_roi, (100, 100))

                # Save sample image
                sample_filename = f"sample_{self.presenter_profile['sample_count']:04d}.jpg"
                sample_path = os.path.join(self.profile_dir, sample_filename)
                cv2.imwrite(sample_path, face_roi)

                # Update profile
                self.presenter_profile['face_samples'].append(sample_filename)
                self.presenter_profile['sample_count'] += 1
                self.presenter_profile['last_updated'] = datetime.now().isoformat()

                if self.identifying_presenter:
                    self.samples_captured += 1
                    self.set_status(f"Capturing presenter... {self.samples_captured}/{self.identification_samples}")

    def save_profile(self):
        if not self.presenter_profile:
            return

        profile_path = os.path.join(self.profile_dir, "presenter_profile.json")
        with open(profile_path, 'w') as f:
            json.dump(self.presenter_profile, f)

    def load_profile(self):
        profile_path = os.path.join(self.profile_dir, "presenter_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                self.presenter_profile = json.load(f)
            self.current_profile_id = self.presenter_profile['id']
            self.set_status(f"Loaded existing presenter profile with {self.presenter_profile['sample_count']} samples")

    def is_presenter(self, frame, face_detection):
        if not self.presenter_profile or not face_detection or len(self.presenter_profile['face_samples']) == 0:
            return False

        # Extract current face ROI
        x, y, w, h = face_detection['bbox']
        current_face = frame[y:y + h, x:x + w]
        if current_face.size == 0:
            return False

        # Resize to match stored samples
        current_face = cv2.resize(current_face, (100, 100))

        # Compare with a subset of stored samples (for performance)
        sample_count = min(5, len(self.presenter_profile['face_samples']))
        matches = []

        for i in range(sample_count):
            sample_path = os.path.join(self.profile_dir, self.presenter_profile['face_samples'][-i - 1])
            if os.path.exists(sample_path):
                sample = cv2.imread(sample_path)

                # Simple histogram comparison
                hist1 = cv2.calcHist([current_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([sample], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
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
        # Check if we're identifying the presenter
        if self.identifying_presenter:
            return

        # Check if we have a presenter profile and verify the detected person
        is_presenter = True
        if self.presenter_profile and face_detection:
            is_presenter = self.is_presenter(frame, face_detection)

        if not is_presenter:
            self.set_status("Warning: Unknown person detected. Only the presenter can control slides.")
            return

        # Prioritize face tracking for the verified presenter
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

        # Capture presenter samples (initial identification or continuous)
        self.capture_presenter_samples(frame, image_rgb)

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

            # Show sample count if we have a profile
            if self.presenter_profile:
                sample_text = f"Samples: {self.presenter_profile['sample_count']}"
                cv2.putText(frame, sample_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # If we're identifying the presenter, show progress
        if self.identifying_presenter:
            cv2.putText(frame, f"IDENTIFYING: {self.samples_captured}/{self.identification_samples}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
            self.create_presenter_profile()
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

    def cleanup(self):
        """Delete all temporary files when the application exits"""
        try:
            if os.path.exists(self.profile_dir):
                shutil.rmtree(self.profile_dir)
                print(f"Deleted temporary directory: {self.profile_dir}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def quit_application(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.cleanup()
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

    # Try to load existing profile
    try:
        app.load_profile()
    except:
        pass  # No existing profile

    root.mainloop()