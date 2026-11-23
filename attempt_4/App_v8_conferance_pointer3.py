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

pyautogui.FAILSAFE = False
# ======================
# COLOR THEMES
# ======================
THEMES = {
    "Default": {
        "primary": "#D8C7B9",
        "secondary": "#C0A08F",
        "accent": "#A09391",
        "light": "#E8D0B9",
        "neutral": "#C0C0C1",
        "text": "#333333",
        "error": "#FF3333",
        "social": "#4A6FA5",
        "warning": "#FFA500",
        "font": ("Helvetica", 10)
    },
    "Dark Mode": {
        "primary": "#2D2D2D",
        "secondary": "#1E1E1E",
        "accent": "#3A3A3A",
        "light": "#4A4A4A",
        "neutral": "#5A5A5A",
        "text": "#E0E0E0",
        "error": "#FF5252",
        "social": "#64B5F6",
        "warning": "#FFB74D",
        "font": ("Consolas", 10, "bold")
    },
    "Ocean Blue": {
        "primary": "#E1F5FE",
        "secondary": "#B3E5FC",
        "accent": "#4FC3F7",
        "light": "#E1F5FE",
        "neutral": "#BBDEFB",
        "text": "#01579B",
        "error": "#D32F2F",
        "social": "#0288D1",
        "warning": "#FFA000",
        "font": ("Arial", 10, "italic")
    },
    "Forest Green": {
        "primary": "#E8F5E9",
        "secondary": "#C8E6C9",
        "accent": "#81C784",
        "light": "#E8F5E9",
        "neutral": "#A5D6A7",
        "text": "#1B5E20",
        "error": "#C62828",
        "social": "#2E7D32",
        "warning": "#EF6C00",
        "font": ("Verdana", 10)
    },
    "Sunset": {
        "primary": "#FFF3E0",
        "secondary": "#FFE0B2",
        "accent": "#FFB74D",
        "light": "#FFF3E0",
        "neutral": "#FFCC80",
        "text": "#E65100",
        "error": "#D84315",
        "social": "#FF6D00",
        "warning": "#FF8F00",
        "font": ("Comic Sans MS", 10)
    }
}


# ======================
# MAIN APPLICATION CLASS
# ======================
class PoseGestureControl:
    def __init__(self, root):
        self.root = root
        self.root.title("PWCV")

        # Set theme
        self.current_theme = "Default"
        self.colors = THEMES[self.current_theme]

        # Set window size and make it non-resizable
        self.root.geometry("1200x700")
        self.root.resizable(False, False)

        # Set application icons
        self.setup_icons()

        # ======================
        # APPLICATION SETTINGS
        # ======================
        self.slide_delay = 1.0  # Delay between slide changes
        self.base_zoom_factor = 1.5  # Base zoom level
        self.dynamic_zoom = True  # Enable dynamic zoom
        self.width, self.height = 640, 480  # Camera resolution
        self.pose_threshold = 0.7  # Pose detection confidence
        self.gesture_threshold = 0.5  # Gesture sensitivity
        self.posture_timeout = 3.0  # Time to show posture warning
        self.validate_posture = True  # Enable posture validation

        # ======================
        # APPLICATION STATE
        # ======================
        self.running = True
        self.control_active = False
        self.show_camera = False
        self.last_gesture_time = 0
        self.shoulder_distance_ref = None
        self.last_posture_warning_time = 0
        self.valid_posture = False

        # Pointer mode variables
        self.pointer_mode = False
        self.both_hands_up_start = 0
        self.both_hands_up_threshold = 3.0
        self.prev_x, self.prev_y = 0, 0
        self.smoothing_factor = 0.3
        self.click_threshold = 30
        self.pointer_speed_multiplier = 2.5
        self.finger_detected = False
        self.click_detected = False

        # ======================
        # INITIALIZE COMPONENTS
        # ======================
        self.setup_hotkeys()
        self.setup_camera()
        self.setup_pose_detection()
        self.create_menu()
        self.create_welcome_screen()
        self.configure_styles()

        # Start video thread
        self.start_video_thread()

        self.enable_toggle_mode = True  # Default to enabled

        self.status_var = tk.StringVar()
        self.status_var.set("Application initialized")

        # Pointer mode variables
        self.pointer_mode = False
        self.both_hands_up_start = 0
        self.both_hands_up_threshold = 3.0
        self.prev_x, self.prev_y = 0, 0
        self.smoothing_factor = 0.3
        self.click_threshold = 30
        self.pointer_speed_multiplier = 2.5
        self.finger_detected = False
        self.click_detected = False

        # New toggle mode variables
        self.enable_toggle_mode_nearby = True  # Default to enabled (2-10cm)
        self.enable_toggle_mode_conference = False  # Default to disabled (20-70cm)

    def configure_styles(self):
        """Configure tkinter styles based on current theme"""
        style = ttk.Style()
        style.theme_use('clam')

        # Base styles
        style.configure('.', background=self.colors["primary"], font=self.colors["font"])
        style.configure('TFrame', background=self.colors["primary"])
        style.configure('TLabel', background=self.colors["primary"], foreground=self.colors["text"])
        style.configure('TButton', background=self.colors["secondary"], foreground=self.colors["text"])
        style.configure('Accent.TButton', background=self.colors["accent"], foreground="white")
        style.configure('Social.TButton', background=self.colors["social"], foreground="white")
        style.configure('Toolbutton', background=self.colors["secondary"], foreground=self.colors["text"])

        # Style maps
        style.map('Toolbutton',
                  background=[('selected', self.colors["accent"]), ('active', self.colors["light"])],
                  foreground=[('selected', 'white'), ('active', self.colors["text"])])

    def map_value(self, value, in_min, in_max, out_min, out_max):
        """Map value with movement multiplier"""
        mapped_value = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        center = (out_max + out_min) / 2
        return center + (mapped_value - center) * self.pointer_speed_multiplier

    def setup_icons(self):
        """Set up application icons for window and taskbar"""
        try:
            icon_path = os.path.join(os.path.dirname(__file__), r"C:\Users\fftt7\PycharmProjects\PWCV\app.ico")
            self.root.iconbitmap(icon_path)

            if os.name == 'nt':
                try:
                    windll.shell32.SetCurrentProcessExplicitAppUserModelID('com.tagemouati.pwcv')
                except Exception as e:
                    print(f"Error setting taskbar icon: {e}")

        except Exception as e:
            print(f"Could not load icons: {e}")
            try:
                icon = tk.PhotoImage(width=16, height=16)
                self.root.iconphoto(False, icon)
            except:
                pass

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

        self.mp_hands = solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def validate_posture(self, landmarks, frame_width, frame_height):
        """Check if the person is in a proper standing posture"""
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

        for landmark in required_landmarks:
            if landmarks[landmark].visibility < self.pose_threshold:
                return False

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        knee_center_y = (left_knee.y + right_knee.y) / 2
        ankle_center_y = (left_ankle.y + right_ankle.y) / 2

        if not (shoulder_center_y < hip_center_y < knee_center_y < ankle_center_y):
            return False

        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        hip_slope = abs(left_hip.y - right_hip.y)
        if shoulder_slope > 0.1 or hip_slope > 0.1:
            return False

        hip_knee_dist = abs(hip_center_y - knee_center_y)
        knee_ankle_dist = abs(knee_center_y - ankle_center_y)

        if abs(hip_knee_dist - knee_ankle_dist) > 0.15:
            return False

        return True

    def show_main_app(self):
        self.current_screen_method = self.show_main_app
        self.create_main_app()

    def show_settings(self):
        self.current_screen_method = self.show_settings
        self.create_settings()

    def show_about(self):
        self.current_screen_method = self.show_about
        self.create_about()

    def clear_frame(self):
        for widget in self.root.winfo_children():
            if widget not in [self.root.children['!menu']]:
                widget.destroy()

    def set_status(self, message):
        if hasattr(self, 'status_var') and self.status_var is not None:
            try:
                self.status_var.set(message)
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.update()
            except Exception as e:
                print(f"Error updating status: {e}")

    def create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit_application)
        menubar.add_cascade(label="File", menu=file_menu)

        sections_menu = tk.Menu(menubar, tearoff=0)
        sections_menu.add_command(label="Main App", command=lambda: [self.stop_control(), self.show_main_app()])
        sections_menu.add_command(label="Settings", command=lambda: [self.stop_control(), self.show_settings()])
        sections_menu.add_command(label="About", command=lambda: [self.stop_control(), self.show_about()])
        menubar.add_cascade(label="Sections", menu=sections_menu)

        self.root.config(menu=menubar)

    def create_welcome_screen(self):
        self.clear_frame()

        welcome_frame = tk.Frame(self.root, bg=self.colors["primary"])
        welcome_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(
            welcome_frame,
            text="Welcome to PWCV",
            font=("Helvetica", 24, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        title_label.pack(pady=(40, 10))

        warning_label = tk.Label(
            welcome_frame,
            text="IMPORTANT: For full functionality, the camera must be directly facing the presenter with no obstacles in between.",
            font=("Helvetica", 11, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["error"],
            wraplength=600
        )
        warning_label.pack(pady=(0, 20))

        desc_text = (
            "This application allows you to control your PowerPoint presentations\n"
            "using body gestures without any additional hardware.\n\n"
            "Key Features:\n"
            "• Navigate slides with hand gestures\n"
            "• Dynamic zoom based on your distance\n"
            "• Posture validation for better accuracy\n"
            "• Customizable settings for optimal experience\n\n"
            "For best results, stand straight facing the camera with your full body visible."
        )

        desc_label = tk.Label(
            welcome_frame,
            text=desc_text,
            font=("Helvetica", 12),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            justify=tk.LEFT
        )
        desc_label.pack(pady=20)

        start_btn = ttk.Button(
            welcome_frame,
            text="Go to Main App",
            command=self.show_main_app,
            style="Accent.TButton"
        )
        start_btn.pack(pady=30)

    def create_main_app(self):
        self.clear_frame()

        # Initialize status_var before using it
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to start control")  # Set initial status

        main_frame = tk.Frame(self.root, bg=self.colors["primary"])
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_frame, bg=self.colors["secondary"])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = tk.Label(left_panel, bg="black")
        self.video_label.pack(pady=10)

        btn_frame = tk.Frame(left_panel, bg=self.colors["secondary"])
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

        right_panel = tk.Frame(main_frame, bg=self.colors["light"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        status_title = tk.Label(
            right_panel,
            text="Control Status",
            font=("Helvetica", 14, "bold"),
            bg=self.colors["light"],
            fg=self.colors["text"]
        )
        status_title.pack(pady=(10, 5))

        self.status_var = tk.StringVar()
        self.set_status("Ready to start control")

        self.status_label = tk.Label(
            right_panel,
            textvariable=self.status_var,
            font=("Helvetica", 12),
            bg=self.colors["light"],
            fg=self.colors["text"],
            wraplength=300,
            justify=tk.LEFT
        )
        self.status_label.pack(pady=5, padx=10)

        self.posture_status_var = tk.StringVar()
        self.posture_status_var.set("Posture: Not detected")

        self.posture_status = tk.Label(
            right_panel,
            textvariable=self.posture_status_var,
            font=("Helvetica", 12),
            bg=self.colors["light"],
            fg=self.colors["text"]
        )
        self.posture_status.pack(pady=5)

        instructions_title = tk.Label(
            right_panel,
            text="How to Use",
            font=("Helvetica", 14, "bold"),
            bg=self.colors["light"],
            fg=self.colors["text"]
        )
        instructions_title.pack(pady=(20, 5))

        instructions_text = (
            "1. Stand straight facing the camera\n"
            "2. Ensure your full body is visible\n"
            "3. Start the control system\n"
            "4. Raise your RIGHT hand for NEXT slide\n"
            "5. Raise your LEFT hand for PREVIOUS slide\n"
            "6. Hold both hands apart for 3s to toggle between slide/pointer modes"
        )

        instructions_label = tk.Label(
            right_panel,
            text=instructions_text,
            font=("Helvetica", 11),
            bg=self.colors["light"],
            fg=self.colors["text"],
            justify=tk.LEFT
        )
        instructions_label.pack(pady=5, padx=10)

    def create_settings(self):
        self.clear_frame()

        settings_frame = tk.Frame(self.root, bg=self.colors["primary"])
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for core settings
        left_panel = tk.Frame(settings_frame, bg=self.colors["primary"])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel for advanced settings
        right_panel = tk.Frame(settings_frame, bg=self.colors["primary"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Title spanning both panels
        title_frame = tk.Frame(settings_frame, bg=self.colors["primary"])
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(
            title_frame,
            text="Settings",
            font=("Helvetica", 20, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        title_label.pack()

        # ======================
        # LEFT PANEL - CORE SETTINGS
        # ======================

        # Mode presets
        mode_frame = tk.LabelFrame(
            left_panel,
            text="1. Mode Presets",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        mode_frame.pack(fill=tk.X, pady=5, padx=5)

        self.mode_var = tk.StringVar(value="custom")

        mode_btn_frame = tk.Frame(mode_frame, bg=self.colors["primary"])
        mode_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        nearby_btn = ttk.Radiobutton(
            mode_btn_frame,
            text="Nearby Mode",
            variable=self.mode_var,
            value="nearby",
            command=self.set_nearby_mode,
            style="Toolbutton"
        )
        nearby_btn.pack(side=tk.LEFT, padx=5, expand=True)

        conference_btn = ttk.Radiobutton(
            mode_btn_frame,
            text="Conference Mode",
            variable=self.mode_var,
            value="conference",
            command=self.set_conference_mode,
            style="Toolbutton"
        )
        conference_btn.pack(side=tk.LEFT, padx=5, expand=True)

        custom_btn = ttk.Radiobutton(
            mode_btn_frame,
            text="Custom Settings",
            variable=self.mode_var,
            value="custom",
            command=self.set_custom_mode,
            style="Toolbutton"
        )
        custom_btn.pack(side=tk.LEFT, padx=5, expand=True)

        # Slide control settings
        control_frame = tk.LabelFrame(
            left_panel,
            text="2. Slide Control",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        control_frame.pack(fill=tk.X, pady=5, padx=5)

        # Slide delay
        delay_frame = tk.Frame(control_frame, bg=self.colors["primary"])
        delay_frame.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(
            delay_frame,
            text="Slide Delay (seconds):",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.delay_slider = tk.Scale(
            delay_frame,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_slide_delay,
            bg=self.colors["primary"],
            fg=self.colors["text"],
            highlightbackground=self.colors["primary"],
            length=200
        )
        self.delay_slider.set(self.slide_delay)
        self.delay_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Appearance settings
        theme_frame = tk.LabelFrame(
            left_panel,
            text="3. Appearance",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        theme_frame.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(
            theme_frame,
            text="Application Theme:",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.theme_var = tk.StringVar(value=self.current_theme)
        theme_menu = ttk.Combobox(
            theme_frame,
            textvariable=self.theme_var,
            values=list(THEMES.keys()),
            state="readonly",
            width=15
        )
        theme_menu.pack(side=tk.LEFT, padx=5)
        theme_menu.bind("<<ComboboxSelected>>", self.change_theme)

        # ======================
        # RIGHT PANEL - ADVANCED SETTINGS
        # ======================

        # Pointer settings
        pointer_frame = tk.LabelFrame(
            right_panel,
            text="4. Pointer Mode",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        pointer_frame.pack(fill=tk.X, pady=5, padx=5)

        # Pointer speed control
        speed_frame = tk.Frame(pointer_frame, bg=self.colors["primary"])
        speed_frame.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(
            speed_frame,
            text="Pointer Speed:",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.speed_slider = tk.Scale(
            speed_frame,
            from_=1.0,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_pointer_speed,
            bg=self.colors["primary"],
            fg=self.colors["text"],
            highlightbackground=self.colors["primary"],
            length=200
        )
        self.speed_slider.set(self.pointer_speed_multiplier)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Warning for pointer speed
        warning_frame = tk.Frame(pointer_frame, bg=self.colors["primary"])
        warning_frame.pack(fill=tk.X, pady=(0, 5), padx=5)

        warning_label = tk.Label(
            warning_frame,
            text="Note: High speeds (>7) may cause instability at far distances",
            font=("Helvetica", 9),
            bg=self.colors["primary"],
            fg=self.colors["warning"],
            justify=tk.LEFT
        )
        warning_label.pack(side=tk.LEFT, padx=10)

        # Zoom settings
        zoom_frame = tk.LabelFrame(
            right_panel,
            text="5. Zoom Settings",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        zoom_frame.pack(fill=tk.X, pady=5, padx=5)

        # Base zoom level
        base_zoom_frame = tk.Frame(zoom_frame, bg=self.colors["primary"])
        base_zoom_frame.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(
            base_zoom_frame,
            text="Base Zoom Level:",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.zoom_slider = tk.Scale(
            base_zoom_frame,
            from_=1.0,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_zoom_level,
            bg=self.colors["primary"],
            fg=self.colors["text"],
            highlightbackground=self.colors["primary"],
            length=200
        )
        self.zoom_slider.set(self.base_zoom_factor)
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Dynamic zoom checkbox
        dynamic_zoom_frame = tk.Frame(zoom_frame, bg=self.colors["primary"])
        dynamic_zoom_frame.pack(fill=tk.X, pady=5, padx=5)

        self.dynamic_zoom_var = tk.BooleanVar(value=self.dynamic_zoom)

        self.dynamic_zoom_cb = tk.Checkbutton(
            dynamic_zoom_frame,
            text="Enable Dynamic Zoom Adjustment",
            variable=self.dynamic_zoom_var,
            command=self.toggle_dynamic_zoom,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        self.dynamic_zoom_cb.pack(side=tk.LEFT, padx=10)

        # Posture settings
        posture_frame = tk.LabelFrame(
            right_panel,
            text="6. Posture Settings",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        posture_frame.pack(fill=tk.X, pady=5, padx=5)

        # Posture validation checkbox
        posture_validation_frame = tk.Frame(posture_frame, bg=self.colors["primary"])
        posture_validation_frame.pack(fill=tk.X, pady=5, padx=5)

        self.posture_validation_var = tk.BooleanVar(value=self.validate_posture)

        self.posture_cb = tk.Checkbutton(
            posture_validation_frame,
            text="Enable Posture Validation",
            variable=self.posture_validation_var,
            command=self.toggle_posture_validation,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        self.posture_cb.pack(side=tk.LEFT, padx=10)

        # Back button at bottom
        back_frame = tk.Frame(settings_frame, bg=self.colors["primary"])
        back_frame.pack(fill=tk.X, pady=10)

        back_btn = ttk.Button(
            back_frame,
            text="Back to Main",
            command=self.show_main_app
        )
        back_btn.pack(pady=10)

        # Mode toggle - Nearby
        toggle_nearby_frame = tk.Frame(control_frame, bg=self.colors["primary"])
        toggle_nearby_frame.pack(fill=tk.X, pady=5, padx=5)

        self.toggle_mode_nearby_var = tk.BooleanVar(value=self.enable_toggle_mode_nearby)

        self.toggle_nearby_cb = tk.Checkbutton(
            toggle_nearby_frame,
            text="Enable Nearby Toggle Mode",
            variable=self.toggle_mode_nearby_var,
            command=self.toggle_mode_nearby_switch,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        self.toggle_nearby_cb.pack(side=tk.LEFT, padx=10)

        # Mode toggle - Conference
        toggle_conference_frame = tk.Frame(control_frame, bg=self.colors["primary"])
        toggle_conference_frame.pack(fill=tk.X, pady=5, padx=5)

        self.toggle_mode_conference_var = tk.BooleanVar(value=self.enable_toggle_mode_conference)

        self.toggle_conference_cb = tk.Checkbutton(
            toggle_conference_frame,
            text="Enable Conference Toggle Mode",
            variable=self.toggle_mode_conference_var,
            command=self.toggle_mode_conference_switch,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        self.toggle_conference_cb.pack(side=tk.LEFT, padx=10)

    def create_theme_settings(self, parent_frame):
        # Theme selection
        theme_frame = tk.LabelFrame(
            parent_frame,
            text="Appearance",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        theme_frame.pack(fill=tk.X, pady=10, padx=5)

        tk.Label(
            theme_frame,
            text="Application Theme:",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.theme_var = tk.StringVar(value=self.current_theme)
        theme_menu = ttk.Combobox(
            theme_frame,
            textvariable=self.theme_var,
            values=list(THEMES.keys()),
            state="readonly",
            width=15
        )
        theme_menu.pack(side=tk.LEFT, padx=5)
        theme_menu.bind("<<ComboboxSelected>>", self.change_theme)

    def create_mode_settings(self, parent_frame):
        # Mode selection
        mode_frame = tk.LabelFrame(
            parent_frame,
            text="Presentation Mode Presets",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        mode_frame.pack(fill=tk.X, pady=10, padx=5)

        self.mode_var = tk.StringVar(value="custom")

        mode_btn_frame = tk.Frame(mode_frame, bg=self.colors["primary"])
        mode_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        nearby_btn = ttk.Radiobutton(
            mode_btn_frame,
            text="Nearby Mode",
            variable=self.mode_var,
            value="nearby",
            command=self.set_nearby_mode,
            style="Toolbutton"
        )
        nearby_btn.pack(side=tk.LEFT, padx=5, expand=True)

        conference_btn = ttk.Radiobutton(
            mode_btn_frame,
            text="Conference Mode",
            variable=self.mode_var,
            value="conference",
            command=self.set_conference_mode,
            style="Toolbutton"
        )
        conference_btn.pack(side=tk.LEFT, padx=5, expand=True)

        custom_btn = ttk.Radiobutton(
            mode_btn_frame,
            text="Custom Settings",
            variable=self.mode_var,
            value="custom",
            command=self.set_custom_mode,
            style="Toolbutton"
        )
        custom_btn.pack(side=tk.LEFT, padx=5, expand=True)

        desc_frame = tk.Frame(mode_frame, bg=self.colors["primary"])
        desc_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Label(
            desc_frame,
            text="• Nearby: For small rooms with close distance\n• Conference: For large rooms with far distance",
            font=("Helvetica", 9),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            justify=tk.LEFT
        ).pack(side=tk.LEFT)

    def create_control_settings(self, parent_frame):
        # Control settings
        control_frame = tk.LabelFrame(
            parent_frame,
            text="Slide Control Settings",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        control_frame.pack(fill=tk.X, pady=10, padx=5)

        # Slide delay
        delay_frame = tk.Frame(control_frame, bg=self.colors["primary"])
        delay_frame.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(
            delay_frame,
            text="Slide Delay (seconds):",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.delay_slider = tk.Scale(
            delay_frame,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_slide_delay,
            bg=self.colors["primary"],
            fg=self.colors["text"],
            highlightbackground=self.colors["primary"],
            length=200
        )
        self.delay_slider.set(self.slide_delay)
        self.delay_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)


    def create_pointer_settings(self, parent_frame):
        # Pointer settings
        pointer_frame = tk.LabelFrame(
            parent_frame,
            text="Pointer Mode Settings",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        pointer_frame.pack(fill=tk.X, pady=10, padx=5)

        # Pointer speed control
        speed_frame = tk.Frame(pointer_frame, bg=self.colors["primary"])
        speed_frame.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(
            speed_frame,
            text="Pointer Speed Multiplier:",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.speed_slider = tk.Scale(
            speed_frame,
            from_=1.0,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_pointer_speed,
            bg=self.colors["primary"],
            fg=self.colors["text"],
            highlightbackground=self.colors["primary"],
            length=200
        )
        self.speed_slider.set(self.pointer_speed_multiplier)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Warning for pointer speed
        warning_frame = tk.Frame(pointer_frame, bg=self.colors["primary"])
        warning_frame.pack(fill=tk.X, pady=(0, 5), padx=5)

        warning_label = tk.Label(
            warning_frame,
            text="Note: Speeds above 7 may cause unstable behavior at far distances",
            font=("Helvetica", 9),
            bg=self.colors["primary"],
            fg=self.colors["warning"],
            justify=tk.LEFT
        )
        warning_label.pack(side=tk.LEFT, padx=10)

    def create_posture_settings(self, parent_frame):
        # Posture and zoom settings
        posture_frame = tk.LabelFrame(
            parent_frame,
            text="Posture & Zoom Settings",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        posture_frame.pack(fill=tk.X, pady=10, padx=5)

        # Zoom controls
        zoom_frame = tk.Frame(posture_frame, bg=self.colors["primary"])
        zoom_frame.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(
            zoom_frame,
            text="Base Zoom Level:",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        self.zoom_slider = tk.Scale(
            zoom_frame,
            from_=1.0,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_zoom_level,
            bg=self.colors["primary"],
            fg=self.colors["text"],
            highlightbackground=self.colors["primary"],
            length=200
        )
        self.zoom_slider.set(self.base_zoom_factor)
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Dynamic zoom checkbox
        self.dynamic_zoom_var = tk.BooleanVar(value=self.dynamic_zoom)

        dynamic_zoom_frame = tk.Frame(posture_frame, bg=self.colors["primary"])
        dynamic_zoom_frame.pack(fill=tk.X, pady=5, padx=5)

        self.dynamic_zoom_cb = tk.Checkbutton(
            dynamic_zoom_frame,
            text="Enable Dynamic Zoom (auto-adjust based on distance)",
            variable=self.dynamic_zoom_var,
            command=self.toggle_dynamic_zoom,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        self.dynamic_zoom_cb.pack(side=tk.LEFT, padx=10)

        # Posture validation checkbox
        self.posture_validation_var = tk.BooleanVar(value=self.validate_posture)

        posture_validation_frame = tk.Frame(posture_frame, bg=self.colors["primary"])
        posture_validation_frame.pack(fill=tk.X, pady=5, padx=5)

        self.posture_cb = tk.Checkbutton(
            posture_validation_frame,
            text="Enable Standing Posture Validation",
            variable=self.posture_validation_var,
            command=self.toggle_posture_validation,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        self.posture_cb.pack(side=tk.LEFT, padx=10)

    def create_about(self):
        self.clear_frame()

        about_frame = tk.Frame(self.root, bg=self.colors["primary"])
        about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(
            about_frame,
            text="About Gesture-Controlled Presentation",
            font=("Helvetica", 20, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        title_label.pack(pady=(10, 20))

        desc_frame = tk.Frame(about_frame, bg=self.colors["light"])
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
            bg=self.colors["light"],
            fg=self.colors["text"],
            wraplength=600,
            justify=tk.LEFT
        )
        desc_label.pack(pady=10, padx=10)

        dev_frame = tk.Frame(about_frame, bg=self.colors["light"])
        dev_frame.pack(fill=tk.X, pady=10, padx=20)

        dev_label = tk.Label(
            dev_frame,
            text="Developed by Yassir Tagemouati",
            font=("Helvetica", 11, "bold"),
            bg=self.colors["light"],
            fg=self.colors["text"]
        )
        dev_label.pack(pady=5)

        social_frame = tk.Frame(about_frame, bg=self.colors["light"])
        social_frame.pack(fill=tk.X, pady=10, padx=20)

        linkedin_btn = ttk.Button(
            social_frame,
            text="LinkedIn",
            command=lambda: webbrowser.open("https://www.linkedin.com/in/yassir-tagemouati-471a6a2a2/"),
            style="Social.TButton"
        )
        linkedin_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        instagram_btn = ttk.Button(
            social_frame,
            text="Instagram",
            command=lambda: webbrowser.open("https://www.instagram.com/yassir_tage/"),
            style="Social.TButton"
        )
        instagram_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        github_btn = ttk.Button(
            social_frame,
            text="Github",
            command=lambda: webbrowser.open("https://github.com/TageYassir"),
            style="Social.TButton"
        )
        github_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        support_frame = tk.Frame(about_frame, bg=self.colors["light"])
        support_frame.pack(fill=tk.X, pady=10, padx=20)

        paypal_btn = ttk.Button(
            support_frame,
            text="Support via PayPal",
            command=lambda: webbrowser.open("https://paypal.me/YassirTagemouati?country.x=MA&locale.x=fr_XC"),
            style="Accent.TButton"
        )
        paypal_btn.pack(pady=10, fill=tk.X)

        email_frame = tk.Frame(support_frame, bg=self.colors["light"])
        email_frame.pack(fill=tk.X, pady=5)

        support_label = tk.Label(
            email_frame,
            text="Support: ",
            font=("Helvetica", 11, "bold"),
            bg=self.colors["light"],
            fg=self.colors["text"]
        )
        support_label.pack(side=tk.LEFT)

        email_link = tk.Label(
            email_frame,
            text="tagemouatiyassir@gmail.com",
            font=("Helvetica", 11, "bold", "underline"),
            bg=self.colors["light"],
            fg="blue",
            cursor="hand2"
        )
        email_link.pack(side=tk.LEFT)
        email_link.bind("<Button-1>", lambda e: webbrowser.open("mailto:tagemouatiyassir@gmail.com"))

        back_btn = ttk.Button(
            about_frame,
            text="Back to Main",
            command=self.show_main_app
        )
        back_btn.pack(pady=10)

    def change_theme(self, event=None):
        theme_name = self.theme_var.get()
        self.current_theme = theme_name
        self.colors = THEMES[theme_name]

        # Reconfigure all styles
        self.configure_styles()

        # Determine current screen (simpler approach)
        current_screen = self.show_main_app  # default
        if hasattr(self, 'current_screen_method'):
            current_screen = self.current_screen_method

        # Recreate current screen with new theme
        current_screen()

        # Update status
        self.set_status(f"Theme changed to {theme_name}")

    def start_video_thread(self):
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def calculate_shoulder_distance(self, landmarks, frame_width):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return abs(left_shoulder.x - right_shoulder.x) * frame_width

    def check_posture(self, landmarks, frame_width, frame_height):
        if not self.validate_posture:
            return True

        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

        for landmark in required_landmarks:
            if landmarks[landmark].visibility < self.pose_threshold:
                return False

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        knee_center_y = (left_knee.y + right_knee.y) / 2
        ankle_center_y = (left_ankle.y + right_ankle.y) / 2

        if not (shoulder_center_y < hip_center_y < knee_center_y < ankle_center_y):
            return False

        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        hip_slope = abs(left_hip.y - right_hip.y)
        if shoulder_slope > 0.1 or hip_slope > 0.1:
            return False

        hip_knee_dist = abs(hip_center_y - knee_center_y)
        knee_ankle_dist = abs(knee_center_y - ankle_center_y)

        if abs(hip_knee_dist - knee_ankle_dist) > 0.15:
            return False

        return True

    def process_control_mode(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # Process pose detection
            pose_results = self.pose.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                # Get body center (between shoulders)
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                body_center_x = (left_shoulder.x + right_shoulder.x) / 2
                body_center_y = (left_shoulder.y + right_shoulder.y) / 2

                # Get wrist positions
                left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                # Calculate distance between hands
                pixel_distance = abs(left_wrist.x - right_wrist.x) * w
                cm_per_pixel = 40 / (0.2 * w)
                distance_cm = pixel_distance * cm_per_pixel

                # Check for toggle modes (swapped distances for more natural use)
                hands_nearby = (20 <= distance_cm <= 70 and  # Wider gesture for conference
                                    left_wrist.visibility > 0.7 and
                                    right_wrist.visibility > 0.7)
                hands_conference = (2 <= distance_cm <= 10 and  # Closer gesture for nearby
                                left_wrist.visibility > 0.7 and
                                right_wrist.visibility > 0.7)

                # Draw distance feedback
                left_x = int(left_wrist.x * w)
                right_x = int(right_wrist.x * w)
                cv2.line(frame, (left_x, 100), (right_x, 100), (0, 255, 255), 2)
                color = (0, 255, 0) if (hands_nearby or hands_conference) else (0, 0, 255)
                cv2.putText(frame, f"{distance_cm:.0f}cm", ((left_x + right_x) // 2, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                current_time = time.time()

                # Check for toggle mode activation
                if (self.enable_toggle_mode_nearby and hands_nearby) or \
                        (self.enable_toggle_mode_conference and hands_conference):
                    if self.both_hands_up_start == 0:
                        self.both_hands_up_start = current_time
                        cv2.putText(frame, "HOLD FOR 3s TO TOGGLE POINTER", (50, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    else:
                        elapsed = current_time - self.both_hands_up_start
                        if elapsed < self.both_hands_up_threshold:
                            remaining = int(self.both_hands_up_threshold - elapsed) + 1
                            cv2.putText(frame, f"{remaining}", (w // 2, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                        if elapsed >= self.both_hands_up_threshold:
                            self.pointer_mode = not self.pointer_mode
                            self.both_hands_up_start = 0

                            # Center cursor when entering pointer mode
                            if self.pointer_mode:
                                screen_width, screen_height = pyautogui.size()
                                pyautogui.moveTo(screen_width // 2, screen_height // 2)

                            if not self.pointer_mode:
                                pyautogui.hotkey('ctrl', 'l')

                            status_msg = "POINTER MODE ON" if self.pointer_mode else "SLIDE MODE ON"
                            cv2.putText(frame, status_msg, (w // 2 - 100, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.both_hands_up_start = 0

                # JOYSTICK CONTROL (Both Modes)
                if self.pointer_mode:
                    screen_width, screen_height = pyautogui.size()
                    screen_center_x = screen_width // 2
                    screen_center_y = screen_height // 2

                    # CONFERENCE MODE: Hand-as-joystick
                    if self.enable_toggle_mode_conference:
                        # Use whichever hand is higher (more raised)
                        active_hand = left_wrist if left_wrist.y < right_wrist.y else right_wrist
                        control_point_x = active_hand.x
                        control_point_y = active_hand.y
                        control_type = "HAND"

                        # Visual feedback
                        point_x = int(control_point_x * w)
                        point_y = int(control_point_y * h)
                        body_px = int(body_center_x * w)
                        body_py = int(body_center_y * h)
                        cv2.circle(frame, (point_x, point_y), 20, (0, 255, 255), -1)
                        cv2.line(frame, (body_px, body_py), (point_x, point_y), (0, 255, 0), 2)

                    # NEARBY MODE: Finger-as-joystick
                    elif self.enable_toggle_mode_nearby and hand_results.multi_hand_landmarks:
                        hand_landmarks = hand_results.multi_hand_landmarks[0]
                        index_finger = hand_landmarks.landmark[8]
                        control_point_x = index_finger.x
                        control_point_y = index_finger.y
                        control_type = "FINGER"

                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Visual feedback
                        point_x = int(control_point_x * w)
                        point_y = int(control_point_y * h)
                        cv2.circle(frame, (point_x, point_y), 10, (0, 255, 255), -1)

                    # Calculate relative movement (same for both modes)
                    rel_x = (control_point_x - 0.5) * 2  # Range: -1 to 1
                    rel_y = (control_point_y - 0.5) * 2

                    # Apply sensitivity
                    move_x = rel_x * self.pointer_speed_multiplier * 500
                    move_y = rel_y * self.pointer_speed_multiplier * 500

                    # Calculate new cursor position
                    cursor_x = screen_center_x + int(move_x)
                    cursor_y = screen_center_y + int(move_y)

                    # Clamp to screen bounds
                    cursor_x = max(0, min(screen_width, cursor_x))
                    cursor_y = max(0, min(screen_height, cursor_y))

                    # Move cursor
                    pyautogui.moveTo(cursor_x, cursor_y, _pause=False)

                    # Display mode status
                    cv2.putText(frame, f"JOYSTICK MODE ({control_type}) - Speed: {self.pointer_speed_multiplier:.1f}x",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # SLIDE CONTROL MODE
                else:
                    if self.validate_posture:
                        self.valid_posture = self.check_posture(landmarks, w, h)
                        if hasattr(self, 'posture_status'):
                            if self.valid_posture:
                                self.posture_status_var.set("Posture: Valid")
                                self.posture_status.config(fg="green")
                            else:
                                self.posture_status_var.set("Posture: Invalid - Stand straight")
                                self.posture_status.config(fg="red")

                                if current_time - self.last_posture_warning_time > self.posture_timeout:
                                    cv2.putText(frame, "STAND STRAIGHT", (50, 60),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    self.last_posture_warning_time = current_time
                    else:
                        self.valid_posture = True
                        if hasattr(self, 'posture_status'):
                            self.posture_status_var.set("Posture: Validation disabled")
                            self.posture_status.config(fg=self.colors["text"])

                    if self.valid_posture or not self.validate_posture:
                        # Dynamic zoom calculation
                        if self.dynamic_zoom:
                            current_shoulder_dist = self.calculate_shoulder_distance(landmarks, w)
                            if self.shoulder_distance_ref is None:
                                self.shoulder_distance_ref = current_shoulder_dist
                            zoom_factor = self.base_zoom_factor * (self.shoulder_distance_ref / current_shoulder_dist)
                            zoom_factor = max(1.0, min(zoom_factor, 10.0))
                        else:
                            zoom_factor = self.base_zoom_factor

                        # Apply zoom
                        crop_w = int(w / zoom_factor)
                        crop_h = int(h / zoom_factor)
                        center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
                        center_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
                        x1 = max(0, min(center_x - crop_w // 2, w - crop_w))
                        y1 = max(0, min(center_y - crop_h // 2, h - crop_h))
                        cropped_frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]
                        frame = cv2.resize(cropped_frame, (w, h))

                        cv2.putText(frame, "SLIDE MODE", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Zoom: {zoom_factor:.1f}x", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Slide control gestures
                        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
                        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
                        eye_level = (left_eye.y + right_eye.y) / 2

                        if current_time - self.last_gesture_time > self.slide_delay:
                            right_hand_raised = (right_wrist.y < eye_level and
                                                 right_wrist.visibility > self.gesture_threshold and
                                                 left_wrist.y > eye_level * 1.1)
                            left_hand_raised = (left_wrist.y < eye_level and
                                                left_wrist.visibility > self.gesture_threshold and
                                                right_wrist.y > eye_level * 1.1)

                            if right_hand_raised:
                                pyautogui.press('left')
                                self.last_gesture_time = current_time
                                cv2.putText(frame, "NEXT SLIDE", (w // 2 - 70, h - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            elif left_hand_raised:
                                pyautogui.press('right')
                                self.last_gesture_time = current_time
                                cv2.putText(frame, "PREV SLIDE", (w // 2 - 70, h - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            print(f"Error in process_control_mode: {e}")

        return frame

    def update_video(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)

                if self.control_active:
                    if self.pointer_mode:
                        if time.time() % 0.1 < 0.05:  # ~10fps processing in pointer mode
                            frame = self.process_control_mode(frame)
                    else:
                        frame = self.process_control_mode(frame)

                if self.show_camera and hasattr(self, 'video_label'):
                    self.update_display(frame)

            except Exception as e:
                print(f"Video processing error: {e}")
                time.sleep(0.1)

    def update_display(self, frame):
        if not hasattr(self, 'video_label') or not self.video_label.winfo_exists():
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_pointer_speed(self, value):
        speed = float(value)
        self.pointer_speed_multiplier = speed

        if speed >= 7:
            warning = " (High speed - Not recommended for far distances)"
            color = self.colors["warning"]
        else:
            warning = ""
            color = self.colors["text"]

        self.set_status(f"Pointer speed set to: {speed:.1f}x{warning}")

        if hasattr(self, 'speed_slider'):
            self.speed_slider.config(fg=color if speed >= 7 else self.colors["text"])

    def set_nearby_mode(self):
        self.base_zoom_factor = 1.0
        self.dynamic_zoom = False
        self.validate_posture = False
        self.pointer_speed_multiplier = 1.5
        self.enable_toggle_mode_nearby = True
        self.enable_toggle_mode_conference = False

        self.zoom_slider.set(self.base_zoom_factor)
        self.dynamic_zoom_var.set(False)
        self.posture_validation_var.set(False)
        self.speed_slider.set(self.pointer_speed_multiplier)
        self.toggle_mode_nearby_var.set(True)
        self.toggle_mode_conference_var.set(False)

        self.zoom_slider.config(state=tk.DISABLED)
        self.dynamic_zoom_cb.config(state=tk.DISABLED)
        self.posture_cb.config(state=tk.DISABLED)
        self.speed_slider.config(state=tk.DISABLED)
        self.toggle_nearby_cb.config(state=tk.NORMAL)
        self.toggle_conference_cb.config(state=tk.NORMAL)

        self.delay_slider.config(state=tk.NORMAL)

        self.set_status(
            "Nearby mode activated: Fixed zoom (1x), no dynamic zoom, no posture validation, lower pointer speed, nearby toggle mode enabled")

    def set_conference_mode(self):
        self.base_zoom_factor = 2.0
        self.dynamic_zoom = False
        self.validate_posture = True
        self.pointer_speed_multiplier = 7.0
        self.enable_toggle_mode_nearby = False
        self.enable_toggle_mode_conference = True

        self.zoom_slider.set(self.base_zoom_factor)
        self.dynamic_zoom_var.set(False)
        self.posture_validation_var.set(True)
        self.speed_slider.set(self.pointer_speed_multiplier)
        self.toggle_mode_nearby_var.set(False)
        self.toggle_mode_conference_var.set(True)

        self.zoom_slider.config(state=tk.NORMAL)
        self.dynamic_zoom_cb.config(state=tk.DISABLED)
        self.posture_cb.config(state=tk.DISABLED)
        self.speed_slider.config(state=tk.NORMAL)
        self.toggle_nearby_cb.config(state=tk.NORMAL)
        self.toggle_conference_cb.config(state=tk.NORMAL)
        self.delay_slider.config(state=tk.NORMAL)

        self.set_status(
            "Conference mode activated: Medium zoom (2x), no dynamic zoom, posture validation enabled, conference toggle mode enabled")

    def set_custom_mode(self):
        self.zoom_slider.config(state=tk.NORMAL)
        self.dynamic_zoom_cb.config(state=tk.NORMAL)
        self.posture_cb.config(state=tk.NORMAL)
        self.speed_slider.config(state=tk.NORMAL)
        self.toggle_nearby_cb.config(state=tk.NORMAL)
        self.toggle_conference_cb.config(state=tk.NORMAL)
        self.delay_slider.config(state=tk.NORMAL)

        self.set_status("Custom mode activated: All settings can be adjusted including both toggle modes")

    def start_control(self):
        self.control_active = True
        self.show_camera = True
        self.pointer_mode = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_status(
            "Control active - Stand straight and:\n"
            "• Left hand up: Previous slide\n"
            "• Right hand up: Next slide\n"
            "• Hold hands ~50cm apart for 3s: Toggle pointer mode")

    def stop_control(self):
        try:
            self.control_active = False
            self.show_camera = False

            if hasattr(self, 'start_btn') and self.start_btn.winfo_exists():
                self.start_btn.config(state=tk.NORMAL)
            if hasattr(self, 'stop_btn') and self.stop_btn.winfo_exists():
                self.stop_btn.config(state=tk.DISABLED)

            self.pointer_mode = False
            self.prev_x, self.prev_y = 0, 0

            self.set_status("Control stopped")

            if hasattr(self, 'posture_status') and self.posture_status.winfo_exists():
                self.posture_status_var.set("Posture: Not detected")
                self.posture_status.config(fg=self.colors["text"])
        except Exception as e:
            print(f"Error in stop_control: {e}")

    def update_slide_delay(self, value):
        self.slide_delay = float(value)

    def update_zoom_level(self, value):
        self.base_zoom_factor = float(value)

    def toggle_dynamic_zoom(self):
        self.dynamic_zoom = self.dynamic_zoom_var.get()
        if self.dynamic_zoom:
            self.shoulder_distance_ref = None
            self.set_status("Dynamic zoom enabled - Zoom will adjust based on distance")
        else:
            self.set_status("Dynamic zoom disabled - Using fixed zoom level")

    def toggle_posture_validation(self):
        self.validate_posture = self.posture_validation_var.get()
        if self.validate_posture:
            self.set_status("Posture validation enabled - System will only respond when standing straight")
        else:
            self.set_status("Posture validation disabled - System will respond to any posture")
            if hasattr(self, 'posture_status') and self.posture_status.winfo_exists():
                self.posture_status_var.set("Posture: Validation disabled")
                self.posture_status.config(fg=self.colors["text"])

    def toggle_interface(self):
        if self.root.state() == 'normal':
            self.root.withdraw()
        else:
            self.root.deiconify()

    def toggle_mode_switch(self):
        self.enable_toggle_mode = self.toggle_mode_var.get()
        if self.enable_toggle_mode:
            self.set_status("Mode toggle enabled - Hold hands apart for 3s to switch between slide and pointer modes")
        else:
            self.set_status("Mode toggle disabled - System will stay in current mode")
            self.pointer_mode = False  # Force back to slide mode

    def toggle_mode_nearby_switch(self):
        self.enable_toggle_mode_nearby = self.toggle_mode_nearby_var.get()
        if self.enable_toggle_mode_nearby:
            self.enable_toggle_mode_conference = False
            self.toggle_mode_conference_var.set(False)
            self.set_status("Nearby toggle mode enabled (2-10cm) - Hold hands close for 3s to switch modes")
        else:
            self.set_status("Nearby toggle mode disabled")

    def toggle_mode_conference_switch(self):
        self.enable_toggle_mode_conference = self.toggle_mode_conference_var.get()
        if self.enable_toggle_mode_conference:
            self.enable_toggle_mode_nearby = False
            self.toggle_mode_nearby_var.set(False)
            self.set_status("Conference toggle mode enabled (20-70cm) - Hold hands apart for 3s to switch modes")
        else:
            self.set_status("Conference toggle mode disabled")


    def quit_application(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
        sys.exit()

    def on_closing(self):
        self.quit_application()


if __name__ == "__main__":
    if os.name == 'nt':
        windll.shell32.SetCurrentProcessExplicitAppUserModelID('com.tagemouati.pwcv')

    root = tk.Tk()
    app = PoseGestureControl(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()