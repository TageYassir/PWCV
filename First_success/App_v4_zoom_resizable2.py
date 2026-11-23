# ======================
# IMPORTS AND LIBRARIES
# ======================
import cv2
import numpy as np
import pyautogui
import time
from mediapipe import solutions
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import keyboard
import sys
import os
import webbrowser
from ctypes import windll  # For taskbar icon on Windows

# ======================
# COLOR THEME
# ======================
COLORS = {
    "primary": "#D8C7B9",
    "secondary": "#C0A08F",
    "accent": "#A09391",
    "light": "#E8D0B9",
    "neutral": "#C0C0C1",
    "text": "#333333",
    "error": "#FF3333",
    "social": "#4A6FA5"  # Color for social media buttons
}

# ======================
# MAIN APPLICATION CLASS
# ======================
class PoseGestureControl:
    def __init__(self, root):
        self.root = root
        self.root.title("PWCV")

        # Set window size and make it non-resizable
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        # Set application icons
        self.setup_icons()        # Load icons


        # ======================
        # APPLICATION SETTINGS
        # ======================
        self.slide_delay = 1.0  # Delay between slide changes
        self.base_zoom_factor = 1.5  # Base zoom level
        self.dynamic_zoom = True  # Enable dynamic zoom
        self.width, self.height = 640, 480  # Camera resolution
        self.pose_threshold = 0.7  # Pose detection confidence
        self.gesture_threshold = 0.5  # Gesture sensitivity

        # ======================
        # APPLICATION STATE
        # ======================
        self.running = True
        self.control_active = False
        self.show_camera = False
        self.last_gesture_time = 0
        self.shoulder_distance_ref = None

        # ======================
        # INITIALIZE COMPONENTS
        # ======================
        self.setup_hotkeys()
        self.setup_camera()
        self.setup_pose_detection()
        self.create_menu()
        self.create_welcome_screen()

        # Start video thread
        self.start_video_thread()

    def setup_icons(self):
        """Set up application icons for window and taskbar"""
        try:
            # This is for the window icon
            icon_path = os.path.join(os.path.dirname(__file__), r"C:\Users\fftt7\PycharmProjects\PWCV\large.ico")
            self.root.iconbitmap(icon_path)

            # This is for the taskbar icon (Windows only)
            if os.name == 'nt':
                app_id = 'com.tagemouati.pwcv'  # Arbitrary string
                windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception as e:
            print(f"Could not load icons: {e}")
            # Create a simple default icon
            try:
                # Create a simple default icon
                icon = tk.PhotoImage(width=16, height=16)
                self.root.iconphoto(False, icon)
            except:
                pass

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

    # ======================
    # GUI METHODS - MENU
    # ======================
    def create_menu(self):
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit_application)
        menubar.add_cascade(label="File", menu=file_menu)

        # Sections menu
        sections_menu = tk.Menu(menubar, tearoff=0)
        sections_menu.add_command(label="Main App", command=self.show_main_app)
        sections_menu.add_command(label="Settings", command=self.show_settings)
        sections_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Sections", menu=sections_menu)

        self.root.config(menu=menubar)

    # ======================
    # GUI METHODS - WELCOME SCREEN
    # ======================
    def create_welcome_screen(self):
        self.clear_frame()

        welcome_frame = tk.Frame(self.root, bg=COLORS["primary"])
        welcome_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = tk.Label(
            welcome_frame,
            text="Welcome to PWCV",
            font=("Helvetica", 24, "bold"),
            bg=COLORS["primary"],
            fg=COLORS["text"]
        )
        title_label.pack(pady=(40, 20))

        # Description
        desc_text = (
            "This application allows you to control your PowerPoint presentations\n"
            "using body gestures without any additional hardware.\n\n"
            "Key Features:\n"
            "• Navigate slides with hand gestures\n"
            "• Dynamic zoom based on your distance\n"
            "• Customizable settings for optimal experience\n\n"
            "Select a section from the menu above to get started."
        )

        desc_label = tk.Label(
            welcome_frame,
            text=desc_text,
            font=("Helvetica", 12),
            bg=COLORS["primary"],
            fg=COLORS["text"],
            justify=tk.LEFT
        )
        desc_label.pack(pady=20)

        # Start button
        start_btn = ttk.Button(
            welcome_frame,
            text="Go to Main App",
            command=self.show_main_app,
            style="Accent.TButton"
        )
        start_btn.pack(pady=30)

    # ======================
    # GUI METHODS - MAIN APP
    # ======================
    def create_main_app(self):
        self.clear_frame()

        main_frame = tk.Frame(self.root, bg=COLORS["primary"])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Camera and controls
        left_panel = tk.Frame(main_frame, bg=COLORS["secondary"])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video display
        self.video_label = tk.Label(left_panel, bg="black")
        self.video_label.pack(pady=10)

        # Control buttons
        btn_frame = tk.Frame(left_panel, bg=COLORS["secondary"])
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(
            btn_frame,
            text="Start Control",
            command=self.start_control,
            style="Accent.TButton"
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            btn_frame,
            text="Stop Control",
            command=self.stop_control,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Right panel - Status and info
        right_panel = tk.Frame(main_frame, bg=COLORS["light"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status label
        status_title = tk.Label(
            right_panel,
            text="Control Status",
            font=("Helvetica", 14, "bold"),
            bg=COLORS["light"],
            fg=COLORS["text"]
        )
        status_title.pack(pady=(10, 5))

        self.status_var = tk.StringVar()
        self.set_status("Ready to start control")

        self.status_label = tk.Label(
            right_panel,
            textvariable=self.status_var,
            font=("Helvetica", 12),
            bg=COLORS["light"],
            fg=COLORS["text"],
            wraplength=300,
            justify=tk.LEFT
        )
        self.status_label.pack(pady=5, padx=10)

        # Instructions
        instructions_title = tk.Label(
            right_panel,
            text="How to Use",
            font=("Helvetica", 14, "bold"),
            bg=COLORS["light"],
            fg=COLORS["text"]
        )
        instructions_title.pack(pady=(20, 5))

        instructions_text = (
            "1. Start the control system\n"
            "2. Stand facing the camera\n"
            "3. Raise your RIGHT hand above eye level for NEXT slide\n"
            "4. Raise your LEFT hand above eye level for PREVIOUS slide\n"
            "5. The system will automatically zoom based on your distance"
        )

        instructions_label = tk.Label(
            right_panel,
            text=instructions_text,
            font=("Helvetica", 11),
            bg=COLORS["light"],
            fg=COLORS["text"],
            justify=tk.LEFT
        )
        instructions_label.pack(pady=5, padx=10)

    # ======================
    # GUI METHODS - SETTINGS
    # ======================
    def create_settings(self):
        self.clear_frame()

        settings_frame = tk.Frame(self.root, bg=COLORS["primary"])
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = tk.Label(
            settings_frame,
            text="Presentation Settings",
            font=("Helvetica", 20, "bold"),
            bg=COLORS["primary"],
            fg=COLORS["text"]
        )
        title_label.pack(pady=(10, 20))

        # Slide delay control
        delay_frame = tk.Frame(settings_frame, bg=COLORS["primary"])
        delay_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            delay_frame,
            text="Slide Delay (seconds):",
            font=("Helvetica", 12),
            bg=COLORS["primary"],
            fg=COLORS["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.delay_slider = tk.Scale(
            delay_frame,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_slide_delay,
            bg=COLORS["primary"],
            fg=COLORS["text"],
            highlightbackground=COLORS["primary"]
        )
        self.delay_slider.set(self.slide_delay)
        self.delay_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Zoom controls
        zoom_frame = tk.Frame(settings_frame, bg=COLORS["primary"])
        zoom_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            zoom_frame,
            text="Base Zoom Level:",
            font=("Helvetica", 12),
            bg=COLORS["primary"],
            fg=COLORS["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.zoom_slider = tk.Scale(
            zoom_frame,
            from_=1.0,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_zoom_level,
            bg=COLORS["primary"],
            fg=COLORS["text"],
            highlightbackground=COLORS["primary"]
        )
        self.zoom_slider.set(self.base_zoom_factor)
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Dynamic zoom checkbox
        self.dynamic_zoom_var = tk.BooleanVar(value=self.dynamic_zoom)

        dynamic_zoom_frame = tk.Frame(settings_frame, bg=COLORS["primary"])
        dynamic_zoom_frame.pack(fill=tk.X, pady=10)

        self.dynamic_zoom_cb = tk.Checkbutton(
            dynamic_zoom_frame,
            text="Enable Dynamic Zoom",
            variable=self.dynamic_zoom_var,
            command=self.toggle_dynamic_zoom,
            font=("Helvetica", 12),
            bg=COLORS["primary"],
            fg=COLORS["text"],
            selectcolor=COLORS["secondary"]
        )
        self.dynamic_zoom_cb.pack(side=tk.LEFT, padx=10)

        # Description
        desc_frame = tk.Frame(settings_frame, bg=COLORS["light"])
        desc_frame.pack(fill=tk.X, pady=20, padx=20)

        desc_label = tk.Label(
            desc_frame,
            text="Adjust these settings to customize your presentation experience. The slide delay controls how quickly you can change slides, while the zoom settings determine how much the camera follows your movements.",
            font=("Helvetica", 11),
            bg=COLORS["light"],
            fg=COLORS["text"],
            wraplength=600,
            justify=tk.LEFT
        )
        desc_label.pack(pady=10, padx=10)

        # Back button
        back_btn = ttk.Button(
            settings_frame,
            text="Back to Main",
            command=self.show_main_app
        )
        back_btn.pack(pady=10)

    # ======================
    # GUI METHODS - ABOUT
    # ======================
    def create_about(self):
        self.clear_frame()

        about_frame = tk.Frame(self.root, bg=COLORS["primary"])
        about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = tk.Label(
            about_frame,
            text="About Gesture-Controlled Presentation",
            font=("Helvetica", 20, "bold"),
            bg=COLORS["primary"],
            fg=COLORS["text"]
        )
        title_label.pack(pady=(10, 20))

        # Description
        desc_frame = tk.Frame(about_frame, bg=COLORS["light"])
        desc_frame.pack(fill=tk.X, pady=10, padx=20)

        desc_text = (
            "This application allows you to control your PowerPoint presentations\n"
            "using body gestures without any additional hardware.\n\n"
            "This application was developed to provide a hands-free presentation experience, "
            "making it ideal for situations where you need to interact with your audience "
            "while maintaining full control of your slides."
        )

        desc_label = tk.Label(
            desc_frame,
            text=desc_text,
            font=("Helvetica", 11),
            bg=COLORS["light"],
            fg=COLORS["text"],
            wraplength=600,
            justify=tk.LEFT
        )
        desc_label.pack(pady=10, padx=10)

        # Developer info
        dev_frame = tk.Frame(about_frame, bg=COLORS["light"])
        dev_frame.pack(fill=tk.X, pady=10, padx=20)

        dev_label = tk.Label(
            dev_frame,
            text="Developed by Yassir Tagemouati",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["light"],
            fg=COLORS["text"]
        )
        dev_label.pack(pady=5)

        # Social media buttons
        social_frame = tk.Frame(about_frame, bg=COLORS["light"])
        social_frame.pack(fill=tk.X, pady=10, padx=20)

        # LinkedIn button
        linkedin_btn = ttk.Button(
            social_frame,
            text="LinkedIn",
            command=lambda: webbrowser.open("https://www.linkedin.com/in/yassir-tagemouati-471a6a2a2/"),
            style="Social.TButton"
        )
        linkedin_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Instagram button
        instagram_btn = ttk.Button(
            social_frame,
            text="Instagram",
            command=lambda: webbrowser.open("https://www.instagram.com/yassir_tage/"),
            style="Social.TButton"
        )
        instagram_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # github button
        github_btn = ttk.Button(
            social_frame,
            text="Github",
            command=lambda: webbrowser.open("https://github.com/TageYassir"),
            style="Social.TButton"
        )
        github_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Support button
        support_frame = tk.Frame(about_frame, bg=COLORS["light"])
        support_frame.pack(fill=tk.X, pady=10, padx=20)

        paypal_btn = ttk.Button(
            support_frame,
            text="Support via PayPal",
            command=lambda: webbrowser.open("https://paypal.me/YassirTagemouati?country.x=MA&locale.x=fr_XC"),
            style="Accent.TButton"
        )
        paypal_btn.pack(pady=10, fill=tk.X)

        # Support info
        support_label = tk.Label(
            support_frame,
            text="Support: tagemouatiyassir@gmail.com",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["light"],
            fg=COLORS["text"]
        )
        support_label.pack(pady=5)

        # Back button
        back_btn = ttk.Button(
            about_frame,
            text="Back to Main",
            command=self.show_main_app
        )
        back_btn.pack(pady=10)
    # ======================
    # UTILITY METHODS
    # ======================
    def clear_frame(self):
        for widget in self.root.winfo_children():
            if widget not in [self.root.children['!menu']]:
                widget.destroy()

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update()

    # ======================
    # SECTION NAVIGATION
    # ======================
    def show_main_app(self):
        self.create_main_app()

    def show_settings(self):
        self.create_settings()

    def show_about(self):
        self.create_about()

    # ======================
    # CORE FUNCTIONALITY
    # ======================
    def start_video_thread(self):
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def calculate_shoulder_distance(self, landmarks, frame_width):
        """Calculate normalized shoulder distance"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate pixel distance between shoulders
        shoulder_pixel_distance = abs(left_shoulder.x - right_shoulder.x) * frame_width

        return shoulder_pixel_distance

    def process_control_mode(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    pyautogui.press('left')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "NEXT SLIDE", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif left_wrist.y < eye_level:
                    pyautogui.press('right')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "PREVIOUS SLIDE", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def update_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                if self.control_active:
                    frame = self.process_control_mode(frame)

                if self.show_camera and hasattr(self, 'video_label'):
                    self.update_display(frame)

            self.root.update()

    def update_display(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

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
        if self.root.state() == 'normal':
            self.root.withdraw()
        else:
            self.root.deiconify()

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

    # Configure styles
    style = ttk.Style()
    style.theme_use('clam')

    # Configure colors
    style.configure('.', background=COLORS["primary"])
    style.configure('TFrame', background=COLORS["primary"])
    style.configure('TLabel', background=COLORS["primary"], foreground=COLORS["text"])
    style.configure('TButton', background=COLORS["secondary"], foreground=COLORS["text"])
    style.configure('Accent.TButton', background=COLORS["accent"], foreground="white")

    app = PoseGestureControl(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()