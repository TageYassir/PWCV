# ======================
# IMPORTS AND LIBRARIES
# ======================
import cv2
import numpy as np
import pyautogui
import pickle
import os
import time
from cvzone.HandTrackingModule import HandDetector
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
class FaceLockedGestureControl:
    def __init__(self, root):
        # Initialize main window
        self.root = root
        self.root.title("Face-Locked PowerPoint Control")

        # ======================
        # APPLICATION SETTINGS
        # ======================
        self.slide_delay = 1.0  # Default delay between slide changes (in seconds)
        self.width, self.height = 640, 480  # Camera resolution
        self.face_matching_threshold = 0.9  # Face tracking sensitivity
        self.profile_file = "user_profile.dat"  # User face data storage

        # ======================
        # APPLICATION STATE
        # ======================
        self.user_face_data = None  # Stores face tracking data
        self.face_tracked = False  # Face tracking status
        self.interface_visible = True  # UI visibility
        self.running = True  # Application running state
        self.calibrating = False  # Calibration mode
        self.control_active = False  # Gesture control active
        self.show_camera = False  # Camera display control

        # ======================
        # INITIALIZE COMPONENTS
        # ======================
        self.setup_hotkeys()
        self.setup_camera()
        self.setup_face_detection()
        self.setup_hand_detection()
        self.load_profile()
        self.create_widgets()

        # Start video processing thread
        self.start_video_thread()

    # ======================
    # SETUP METHODS
    # ======================
    def setup_hotkeys(self):
        """Configure keyboard shortcuts for the application"""
        keyboard.add_hotkey('f12', self.toggle_interface)
        keyboard.add_hotkey('shift+f12', self.quit_application)

    def setup_camera(self):
        """Initialize and configure the camera"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

    def setup_face_detection(self):
        """Initialize face detection using MediaPipe"""
        self.mp_face_detection = solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.8
        )

    def setup_hand_detection(self):
        """Initialize hand detection using cvzone"""
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)

    # ======================
    # GUI METHODS
    # ======================
    def create_widgets(self):
        """Create and arrange all GUI components"""
        # Main container
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Video display (initially hidden)
        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(pady=10)
        self.video_label.pack_forget()  # Hide initially

        # Status label
        self.status_var = tk.StringVar()
        self.set_status("Ready" if self.user_face_data else "Please calibrate your face")
        self.status_label = tk.Label(
            self.main_frame,
            textvariable=self.status_var,
            font=('Helvetica', 12),
            wraplength=400
        )
        self.status_label.pack(pady=5)

        # Slide delay control
        self.create_delay_control()

        # Buttons frame
        self.create_control_buttons()

    def create_delay_control(self):
        """Create slider for adjusting slide transition delay"""
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
        """Create action buttons for the application"""
        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(pady=10)

        # Calibrate button
        self.calibrate_btn = ttk.Button(
            btn_frame,
            text="Calibrate My Face",
            command=self.start_calibration
        )
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)

        # Start button
        self.start_btn = ttk.Button(
            btn_frame,
            text="Start Control",
            command=self.start_control,
            state=tk.NORMAL if self.user_face_data else tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

    def set_status(self, message):
        """Update the status message in the UI"""
        self.status_var.set(message)
        self.root.update()

    # ======================
    # CORE FUNCTIONALITY
    # ======================
    def start_video_thread(self):
        """Start the thread that processes video frames"""
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def update_video(self):
        """Main video processing loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                if self.calibrating or self.control_active:
                    display_frame = frame.copy()

                    if self.calibrating:
                        self.process_calibration_mode(frame, display_frame)
                    elif self.control_active:
                        self.process_control_mode(frame, display_frame)

                    # Update display if in calibration mode or control active with face tracked
                    if self.show_camera or (self.control_active and self.face_tracked):
                        self.update_display(display_frame)
                    elif not self.show_camera:
                        # Hide the video display when not needed
                        self.video_label.pack_forget()

            self.root.update()

    def process_calibration_mode(self, frame, display_frame):
        """Handle face calibration mode"""
        if keyboard.is_pressed('space'):
            if self.capture_face(frame):
                self.show_camera = False
                self.video_label.pack_forget()

        # Draw calibration instructions
        cv2.putText(display_frame, "Face the camera", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to capture", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def process_control_mode(self, frame, display_frame):
        """Handle gesture control mode"""
        # Check if the tracked face is present
        self.face_tracked = self.track_face(frame)

        if self.face_tracked:
            # Only detect hands if tracked face is present
            hands, _ = self.detector.findHands(frame, draw=False)
            if hands:
                self.process_hand_gestures(hands[0])

            # Draw tracking indicator
            cv2.putText(display_frame, "TRACKING YOUR FACE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Draw not tracking warning
            cv2.putText(display_frame, "FACE NOT DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def process_hand_gestures(self, hand):
        """Process detected hand gestures"""
        cx, cy = hand["center"]
        fingers = self.detector.fingersUp(hand)

        # Only respond when all fingers are open (full hand)
        if fingers == [1, 1, 1, 1, 1]:
            if cx < self.width // 2:  # Left side of screen - previous slide
                pyautogui.press('left')
                time.sleep(self.slide_delay)  # Apply configured delay
            else:  # Right side of screen - next slide
                pyautogui.press('right')
                time.sleep(self.slide_delay)  # Apply configured delay

    def update_display(self, frame):
        """Update the video display in the GUI"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.video_label.pack()  # Ensure it's visible

    # ======================
    # FACE TRACKING METHODS
    # ======================
    def capture_face(self, frame):
        """Capture and store face data for tracking"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)

        if results.detections:
            # Get the primary face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            landmarks = detection.location_data.relative_keypoints

            # Store face data
            self.user_face_data = {
                'bbox': (bbox.xmin, bbox.ymin, bbox.width, bbox.height),
                'landmarks': [(lm.x, lm.y) for lm in landmarks]
            }
            self.save_profile()
            self.calibrating = False
            self.set_status("Face captured! Click Start Control")
            self.calibrate_btn.config(state=tk.NORMAL)
            self.start_btn.config(state=tk.NORMAL)
            return True
        return False

    def track_face(self, frame):
        """Check if the current face matches the stored profile"""
        if not self.user_face_data:
            return False

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            landmarks = detection.location_data.relative_keypoints

            # Simple comparison with stored data
            stored_bbox = self.user_face_data['bbox']
            bbox_similarity = 1 - (abs(bbox.xmin - stored_bbox[0]) +
                                   abs(bbox.ymin - stored_bbox[1]) +
                                   abs(bbox.width - stored_bbox[2]) +
                                   abs(bbox.height - stored_bbox[3])) / 4

            return bbox_similarity > self.face_matching_threshold
        return False

    # ======================
    # PROFILE MANAGEMENT
    # ======================
    def save_profile(self):
        """Save face tracking data to file"""
        with open(self.profile_file, 'wb') as f:
            pickle.dump(self.user_face_data, f)

    def load_profile(self):
        """Load face tracking data from file"""
        if os.path.exists(self.profile_file):
            with open(self.profile_file, 'rb') as f:
                self.user_face_data = pickle.load(f)
            return True
        return False

    # ======================
    # CONTROL METHODS
    # ======================
    def start_calibration(self):
        """Enter face calibration mode"""
        self.calibrating = True
        self.control_active = False
        self.show_camera = True
        self.video_label.pack()  # Show camera feed
        self.set_status("Face the camera and press SPACE to capture your face")
        self.calibrate_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)

    def start_control(self):
        """Activate gesture control mode"""
        self.control_active = True
        self.calibrating = False
        self.show_camera = False
        self.video_label.pack_forget()  # Hide camera feed
        self.set_status(
            "Control active - Only your face can control slides\nF12 to hide/show interface, Shift+F12 to exit")

    def update_slide_delay(self, value):
        """Update the delay between slide changes"""
        self.slide_delay = float(value)

    # ======================
    # APPLICATION CONTROL
    # ======================
    def toggle_interface(self):
        """Toggle the application window visibility"""
        if self.interface_visible:
            self.root.withdraw()  # Hide the window
        else:
            self.root.deiconify()  # Show the window
        self.interface_visible = not self.interface_visible

    def quit_application(self):
        """Cleanly exit the application"""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
        sys.exit()

    def on_closing(self):
        """Handle window close event"""
        self.quit_application()


# ======================
# APPLICATION ENTRY POINT
# ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceLockedGestureControl(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()