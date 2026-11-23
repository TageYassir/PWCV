# ======================
# IMPORTS AND LIBRARIES
# ======================
import cv2
import numpy as np
import pyautogui
import time
from mediapipe import solutions
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import keyboard
import sys
import os
import pickle
from datetime import datetime
import face_recognition


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
        self.base_zoom_factor = 1.5  # Base zoom level
        self.dynamic_zoom = True  # Enable dynamic zoom
        self.width, self.height = 640, 480  # Camera resolution
        self.pose_threshold = 0.7  # Pose detection confidence
        self.gesture_threshold = 0.5  # Gesture sensitivity
        self.face_auth_enabled = True  # Face authentication flag (enabled by default)
        self.known_faces_dir = "known_faces"  # Directory to store known faces
        self.face_distance_threshold = 0.6  # Face recognition threshold
        self.face_detection_timeout = 2.0  # Time before stopping if face is lost

        # Create directory for known faces if it doesn't exist
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)

        # ======================
        # APPLICATION STATE
        # ======================
        self.interface_visible = True
        self.running = True
        self.control_active = False
        self.show_camera = False
        self.last_gesture_time = 0
        self.known_face_encodings = []
        self.known_face_names = []
        self.current_user = None
        self.shoulder_distance_ref = None  # Reference shoulder distance for dynamic zoom
        self.face_detection_active = False
        self.last_face_detection_time = 0

        # ======================
        # INITIALIZE COMPONENTS
        # ======================
        self.setup_hotkeys()
        self.setup_camera()
        self.setup_pose_detection()
        self.load_known_faces()
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

    def setup_pose_detection(self):
        self.mp_pose = solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = solutions.drawing_utils

    def load_known_faces(self):
        """Load known face encodings from the directory"""
        self.known_face_encodings = []
        self.known_face_names = []

        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(".pkl"):
                filepath = os.path.join(self.known_faces_dir, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings.append(data['encoding'])
                    self.known_face_names.append(data['name'])

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

        # User info label
        self.user_var = tk.StringVar()
        self.user_var.set("Current user: None")
        self.user_label = tk.Label(
            self.main_frame,
            textvariable=self.user_var,
            font=('Helvetica', 10, 'bold'),
            fg='blue'
        )
        self.user_label.pack(pady=2)

        # Slide delay control
        self.create_delay_control()

        # Zoom control
        self.create_zoom_control()

        # Face authentication controls
        self.create_face_auth_controls()

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

    def create_zoom_control(self):
        zoom_frame = tk.Frame(self.main_frame)
        zoom_frame.pack(pady=5)

        tk.Label(zoom_frame, text="Base Zoom Level (x):").pack(side=tk.LEFT)
        self.zoom_slider = tk.Scale(
            zoom_frame,
            from_=1.0,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_zoom_level
        )
        self.zoom_slider.set(self.base_zoom_factor)
        self.zoom_slider.pack(side=tk.LEFT, padx=5)

        # Dynamic zoom checkbox
        self.dynamic_zoom_var = tk.BooleanVar(value=self.dynamic_zoom)
        self.dynamic_zoom_cb = tk.Checkbutton(
            zoom_frame,
            text="Dynamic Zoom",
            variable=self.dynamic_zoom_var,
            command=self.toggle_dynamic_zoom
        )
        self.dynamic_zoom_cb.pack(side=tk.LEFT, padx=10)

    def create_face_auth_controls(self):
        auth_frame = tk.Frame(self.main_frame)
        auth_frame.pack(pady=10)

        # Face registration controls
        reg_frame = tk.Frame(auth_frame)
        reg_frame.pack(pady=5)

        tk.Label(reg_frame, text="Register New User:").pack(side=tk.LEFT)
        self.new_user_name = tk.StringVar()
        tk.Entry(reg_frame, textvariable=self.new_user_name, width=20).pack(side=tk.LEFT, padx=5)

        self.register_btn = ttk.Button(
            reg_frame,
            text="Take Photo",
            command=self.register_new_user
        )
        self.register_btn.pack(side=tk.LEFT, padx=5)

        # Face authentication toggle
        auth_toggle_frame = tk.Frame(auth_frame)
        auth_toggle_frame.pack(pady=5)

        self.auth_toggle_btn = ttk.Button(
            auth_toggle_frame,
            text="Disable Face Authentication",
            command=self.toggle_face_auth
        )
        self.auth_toggle_btn.pack(side=tk.LEFT, padx=5)

        # Known faces list
        list_frame = tk.Frame(auth_frame)
        list_frame.pack(pady=5)

        tk.Label(list_frame, text="Registered Users:").pack()
        self.known_faces_list = tk.Listbox(list_frame, height=4, width=30)
        self.known_faces_list.pack(side=tk.LEFT, fill=tk.BOTH)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        scrollbar.config(command=self.known_faces_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.known_faces_list.config(yscrollcommand=scrollbar.set)

        self.update_known_faces_list()

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

        self.show_camera_btn = ttk.Button(
            btn_frame,
            text="Show Camera",
            command=self.toggle_camera_display
        )
        self.show_camera_btn.pack(side=tk.LEFT, padx=5)

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update()

    def update_user_label(self):
        if self.current_user:
            self.user_var.set(f"Current user: {self.current_user}")
            self.user_label.config(fg='green')
        else:
            self.user_var.set("Current user: None")
            self.user_label.config(fg='red')

    def update_known_faces_list(self):
        self.known_faces_list.delete(0, tk.END)
        for name in self.known_face_names:
            self.known_faces_list.insert(tk.END, name)

    # ======================
    # FACE AUTHENTICATION
    # ======================
    def register_new_user(self):
        name = self.new_user_name.get().strip()
        if not name:
            self.set_status("Please enter a name for the new user")
            return

        # Capture current frame
        ret, frame = self.cap.read()
        if not ret:
            self.set_status("Failed to capture image")
            return

        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            self.set_status("No face detected in the frame")
            return

        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            self.set_status("Failed to extract face features")
            return

        # Save the face encoding
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
        filepath = os.path.join(self.known_faces_dir, filename)

        data = {
            'name': name,
            'encoding': face_encodings[0],
            'timestamp': timestamp
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        # Update known faces
        self.load_known_faces()
        self.update_known_faces_list()
        self.set_status(f"Successfully registered user: {name}")
        self.new_user_name.set("")

    def recognize_face(self, rgb_frame):
        """Recognize faces in the frame and return the best match"""
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return None

        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            return None

        # Compare with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding,
                                                     tolerance=self.face_distance_threshold)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                return self.known_face_names[best_match_index]

        return None

    def toggle_face_auth(self):
        self.face_auth_enabled = not self.face_auth_enabled
        if self.face_auth_enabled:
            self.auth_toggle_btn.config(text="Disable Face Authentication")
            self.set_status("Face authentication enabled - Only registered users can control")
        else:
            self.auth_toggle_btn.config(text="Enable Face Authentication")
            self.set_status("Face authentication disabled - Anyone can control")
            self.current_user = None
            self.face_detection_active = False
            self.update_user_label()

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
                elif self.face_auth_enabled and not self.current_user:
                    # Try to recognize face if authentication is enabled but no user is logged in
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    recognized_user = self.recognize_face(rgb_frame)
                    if recognized_user:
                        self.current_user = recognized_user
                        self.last_face_detection_time = time.time()
                        self.face_detection_active = True
                        self.update_user_label()
                        self.set_status(f"User recognized: {recognized_user}")

                if self.show_camera:
                    self.update_display(frame)
                else:
                    self.video_label.pack_forget()

            self.root.update()

    def calculate_shoulder_distance(self, landmarks, frame_width):
        """Calculate normalized shoulder distance"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate pixel distance between shoulders
        shoulder_pixel_distance = abs(left_shoulder.x - right_shoulder.x) * frame_width

        return shoulder_pixel_distance

    def process_control_mode(self, frame):
        # Convert to RGB for both MediaPipe and face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # First check for face authentication if enabled
        if self.face_auth_enabled:
            current_time = time.time()
            recognized_user = self.recognize_face(rgb_frame)

            if recognized_user:
                self.current_user = recognized_user
                self.last_face_detection_time = current_time
                self.face_detection_active = True
                self.update_user_label()
            else:
                if current_time - self.last_face_detection_time > self.face_detection_timeout:
                    self.face_detection_active = False
                    self.current_user = None
                    self.update_user_label()
                    self.stop_control()
                    cv2.putText(frame, "AUTHORIZED USER NOT DETECTED", (50, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return frame

        # Only proceed with pose detection if face authentication is satisfied
        if not self.face_auth_enabled or (self.face_auth_enabled and self.face_detection_active):
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                h, w, _ = frame.shape

                # Get key body points
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
                right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
                left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                # Calculate zoom center (between shoulders)
                center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
                center_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)

                # === Dynamic Zoom Calculation ===
                if self.dynamic_zoom:
                    # Calculate current shoulder distance
                    current_shoulder_dist = self.calculate_shoulder_distance(landmarks, w)

                    # Set reference distance if not set
                    if self.shoulder_distance_ref is None:
                        self.shoulder_distance_ref = current_shoulder_dist

                    # Calculate zoom factor based on distance change
                    distance_ratio = self.shoulder_distance_ref / current_shoulder_dist
                    zoom_factor = self.base_zoom_factor * distance_ratio
                    zoom_factor = max(1.0, min(zoom_factor, 10.0))  # Clamp between 1.0 and 10.0
                else:
                    zoom_factor = self.base_zoom_factor

                # === Apply Zoom ===
                crop_w = int(self.width / zoom_factor)
                crop_h = int(self.height / zoom_factor)

                # Clamp crop window within frame
                x1 = max(0, min(center_x - crop_w // 2, w - crop_w))
                y1 = max(0, min(center_y - crop_h // 2, h - crop_h))

                # Crop and resize for zoom effect
                cropped_frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]
                frame = cv2.resize(cropped_frame, (self.width, self.height))

                # Display zoom level
                cv2.putText(frame, f"Zoom: {zoom_factor:.1f}x", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Gesture detection still based on original normalized coordinates
                eye_level = (left_eye.y + right_eye.y) / 2
                current_time = time.time()

                if current_time - self.last_gesture_time > self.slide_delay:
                    if right_wrist.y < eye_level:
                        pyautogui.press('right')
                        self.last_gesture_time = current_time
                        cv2.putText(frame, "NEXT SLIDE", (50, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif left_wrist.y < eye_level:
                        pyautogui.press('left')
                        self.last_gesture_time = current_time
                        cv2.putText(frame, "PREVIOUS SLIDE", (50, 60),
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
        if self.face_auth_enabled and not self.current_user:
            self.set_status("Cannot start - No authenticated user detected")
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

    def toggle_camera_display(self):
        self.show_camera = not self.show_camera
        if self.show_camera:
            self.show_camera_btn.config(text="Hide Camera")
        else:
            self.show_camera_btn.config(text="Show Camera")

    def update_slide_delay(self, value):
        self.slide_delay = float(value)

    def update_zoom_level(self, value):
        self.base_zoom_factor = float(value)

    def toggle_dynamic_zoom(self):
        self.dynamic_zoom = self.dynamic_zoom_var.get()
        if self.dynamic_zoom:
            self.shoulder_distance_ref = None  # Reset reference distance
            self.set_status("Dynamic zoom enabled - Zoom will adjust based on distance")
        else:
            self.set_status("Dynamic zoom disabled - Using fixed zoom level")

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