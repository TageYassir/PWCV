


# A failing test the two featurs can't be on the same concept there are a lot of missdetections because the zoom doesn't frame
# always the head of the person



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
        self.locked_face_center = None
        self.lock_tolerance = 100  # pixels, adjust as needed

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
        self.setup_pose_detection()
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
        self.mp_face = solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

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

            self.root.update()

    def process_control_mode(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Face Detection
        face_results = self.face_detection.process(image_rgb)

        if face_results.detections:
            candidates = []

            for face in face_results.detections:
                bbox = face.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                center = (x + bw // 2, y + bh // 2)
                candidates.append((center, (x, y, bw, bh)))

            # Lock target if not already locked
            if self.locked_face_center is None:
                self.locked_face_center = candidates[0][0]

            # Find closest face to the locked center
            def distance(c1, c2):
                return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

            target = min(candidates, key=lambda c: distance(c[0], self.locked_face_center))

            # If face moved too far, unlock (optional)
            if distance(target[0], self.locked_face_center) > 200:
                self.locked_face_center = target[0]

            # Zoom in on locked target
            cx, cy = target[0]
            x, y, bw, bh = target[1]

            zoom_factor = 1.5
            crop_w = int(self.width / zoom_factor)
            crop_h = int(self.height / zoom_factor)
            x1 = max(0, min(cx - crop_w // 2, w - crop_w))
            y1 = max(0, min(cy - crop_h // 2, h - crop_h))

            cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            frame = cv2.resize(cropped, (self.width, self.height))

        # Pose Detection for gesture control
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
                if right_wrist.y < eye_level:
                    pyautogui.press('right')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "NEXT SLIDE", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif left_wrist.y < eye_level:
                    pyautogui.press('left')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "PREVIOUS SLIDE", (50, 50),
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