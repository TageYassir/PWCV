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
            model_selection=1,  # Use full-range model for better detection
            min_detection_confidence=0.5
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

    def update_tracking(self, face_detection, body_detection):
        """Update the tracking position based on available detections"""
        # Prioritize face tracking
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
        self.update_tracking(face_detection, body_detection)

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