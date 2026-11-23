# ======================
# IMPORTS AND LIBRARIES
# ======================
import cv2
import pyautogui
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import keyboard
import sys
import os
import webbrowser
from ctypes import windll  # For taskbar icon on Windows
import math
import speech_recognition as sr

# Mediapipe imports
import mediapipe as mp
from mediapipe import solutions


# YOLO import
from ultralytics import YOLO  # For object detection

from attempt_15.AdaptiveVoiceRecognizer_v24 import AdaptiveVoiceRecognizer
from attempt_15.cookies import UserPreferences




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

        # Initialize preferences
        self.preferences = UserPreferences()
        self.preferences.load_preferences()

        # Set theme from preferences
        self.current_theme = self.preferences.get('theme', 'Default')
        self.colors = THEMES[self.current_theme]

        # Set window size and make it non-resizable
        self.root.geometry("1300x720")
        self.root.resizable(False, False)

        # Set application icons
        self.setup_icons()

        # ======================
        # APPLICATION SETTINGS (with defaults from preferences)
        # ======================
        self.slide_delay = self.preferences.get('slide_delay', 1.0)
        self.base_zoom_factor = self.preferences.get('base_zoom_factor', 1.5)
        self.dynamic_zoom = self.preferences.get('dynamic_zoom', True)
        self.width, self.height = 640, 480  # Camera resolution
        self.pose_threshold = 0.7  # Pose detection confidence
        self.gesture_threshold = 0.5  # Gesture sensitivity
        self.posture_timeout = 3.0  # Time to show posture warning
        self.validate_posture = self.preferences.get('validate_posture', True)

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
        self.pointer_speed_multiplier = self.preferences.get('pointer_speed', 2.5)
        self.finger_detected = False
        self.click_detected = False

        # Camera device handling
        self.available_cameras = []  # list of (index, name) tuples
        self.selected_camera_index = self.preferences.get('camera_index', 0)
        self.update_camera_list()  # populate using pygrabber
        self._camera_failure_count = 0
        self._camera_failure_threshold = 30  # consecutive failures before error
        self._camera_error_shown = False

        # Microphone device handling
        self.available_microphones = []  # list of (index, name) tuples
        self.selected_microphone_index = self.preferences.get('microphone_index', 0)
        self.update_microphone_list()  # populate using pyaudio

        # Toggle mode variables
        self.enable_toggle_mode_nearby = self.preferences.get('enable_toggle_mode_nearby', True)
        self.enable_toggle_mode_conference = self.preferences.get('enable_toggle_mode_conference', False)

        # Camera mode (normal/high)
        self.camera_mode_var = tk.StringVar(value=self.preferences.get('camera_mode', 'normal'))
        self._original_process_control_mode = self.process_control_mode_normal
        self._high_process_control_mode = self.process_control_mode_high
        if self.camera_mode_var.get() == 'high':
            self.process_control_mode = self._high_process_control_mode
        else:
            self.process_control_mode = self._original_process_control_mode

        # Pose estimation settings
        self.enable_pose_estimation = self.preferences.get('enable_pose_estimation', False)
        self.pose_estimation_model = None
        self.pose_landmarker = None

        # Furniture detection
        self.enable_furniture_detection = self.preferences.get('enable_furniture_detection', False)
        self.furniture_model = None
        self.chair_distance_threshold = 0.3
        self.table_distance_threshold = 0.4
        self.load_furniture_model()

        # Teacher/student mode
        self.teacher_mode_active = self.preferences.get('teacher_mode_active', False)

        # Voice control
        self.voice_recognizer = None
        self.enable_voice_control = self.preferences.get('enable_voice_control', False)
        self.voice_stop_event = threading.Event()
        self.voice_listening_thread = None
        self.voice_control_var = tk.BooleanVar(value=self.enable_voice_control)
        self.voice_lang_var = tk.StringVar(value=self.preferences.get('voice_language', 'en'))

        # ======================
        # INITIALIZE COMPONENTS
        # ======================
        self.setup_hotkeys()
        self.setup_camera()  # now uses self.selected_camera_index
        self.setup_pose_detection()
        self.create_menu()
        self.configure_styles()
        self.create_welcome_screen()

        # Start video thread
        self.start_video_thread()

        # Status variable
        self.status_var = tk.StringVar()
        self.status_var.set("Application initialized")

        self.icon_path = "app.ico"  # Update with your actual path

        # Initialize pose estimation if enabled
        if self.enable_pose_estimation:
            self.initialize_pose_estimation()

        # Apply any remaining UI preferences after a short delay (ensures widgets exist)
        self.root.after(100, lambda: self.preferences.apply_preferences(self))


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
            icon_path = os.path.join(os.path.dirname(__file__), r"C:\Users\fftt7\PyCharmMiscProject\PWCV\attempt_14\app.ico")
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

    def create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit_application)
        menubar.add_cascade(label="File", menu=file_menu)

        sections_menu = tk.Menu(menubar, tearoff=0)
        sections_menu.add_command(label="Main App", command=lambda: [self.stop_control(), self.show_main_app()])
        sections_menu.add_command(label="Settings", command=lambda: [self.stop_control(), self.show_settings()])
        sections_menu.add_command(label="Camera", command=lambda: [self.stop_control(), self.create_camera_settings()])
        sections_menu.add_command(label="Voice", command=lambda: [self.stop_control(), self.create_voice_settings()])
        sections_menu.add_command(label="About", command=lambda: [self.stop_control(), self.show_about()])
        menubar.add_cascade(label="Sections", menu=sections_menu)

        self.root.config(menu=menubar)

    def create_welcome_screen(self):
        self.clear_frame()

        welcome_frame = tk.Frame(self.root, bg=self.colors["primary"])
        welcome_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Check if we have existing preferences
        has_preferences = self.preferences.load_preferences()

        # If we have preferences, skip the mode selection and go directly to main app
        if has_preferences:
            # Only apply basic preferences that are safe to apply at this stage
            self.apply_basic_preferences()
            # Add a small delay to show the welcome screen briefly
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
                "Please select your presentation mode:"
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

            # Mode selection buttons frame
            mode_frame = tk.Frame(welcome_frame, bg=self.colors["primary"])
            mode_frame.pack(pady=20)

            # Nearby Mode Button
            nearby_btn = ttk.Button(
                mode_frame,
                text="Nearby Mode",
                command=lambda: [self.set_nearby_mode(), self.show_main_app()],
                style="Accent.TButton",
                width=15
            )
            nearby_btn.pack(side=tk.LEFT, padx=10, pady=10)

            # Conference Mode Button
            conference_btn = ttk.Button(
                mode_frame,
                text="Conference Mode",
                command=lambda: [self.set_conference_mode(), self.show_main_app()],
                style="Accent.TButton",
                width=15
            )
            conference_btn.pack(side=tk.LEFT, padx=10, pady=10)

            # Mode descriptions
            desc_frame = tk.Frame(welcome_frame, bg=self.colors["primary"])
            desc_frame.pack(pady=(0, 20))

            tk.Label(
                desc_frame,
                text="• Nearby Mode: For close-range control \n"
                     "• Conference Mode: For long-range control ",
                font=("Helvetica", 10),
                bg=self.colors["primary"],
                fg=self.colors["text"],
                justify=tk.LEFT
            ).pack()

            # Start button (optional - you can remove if you want immediate transition after mode selection)
            start_btn = ttk.Button(
                welcome_frame,
                text="Start Presentation",
                command=self.show_main_app,
                style="Accent.TButton"
            )
            start_btn.pack(pady=30)
            return
        else:
            # Only apply basic preferences that are safe to apply at this stage
            self.apply_basic_preferences()
            # Add a small delay to show the welcome screen briefly
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
                "Please select your presentation mode:"
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

            # Mode selection buttons frame
            mode_frame = tk.Frame(welcome_frame, bg=self.colors["primary"])
            mode_frame.pack(pady=20)

            # Nearby Mode Button
            nearby_btn = ttk.Button(
                mode_frame,
                text="Nearby Mode",
                command=lambda: [self.set_nearby_mode(), self.show_main_app()],
                style="Accent.TButton",
                width=15
            )
            nearby_btn.pack(side=tk.LEFT, padx=10, pady=10)

            # Conference Mode Button
            conference_btn = ttk.Button(
                mode_frame,
                text="Conference Mode",
                command=lambda: [self.set_conference_mode(), self.show_main_app()],
                style="Accent.TButton",
                width=15
            )
            conference_btn.pack(side=tk.LEFT, padx=10, pady=10)

            # Mode descriptions
            desc_frame = tk.Frame(welcome_frame, bg=self.colors["primary"])
            desc_frame.pack(pady=(0, 20))

            tk.Label(
                desc_frame,
                text="• Nearby Mode: For close-range control \n"
                     "• Conference Mode: For long-range control ",
                font=("Helvetica", 10),
                bg=self.colors["primary"],
                fg=self.colors["text"],
                justify=tk.LEFT
            ).pack()

            # Start button (optional - you can remove if you want immediate transition after mode selection)
            start_btn = ttk.Button(
                welcome_frame,
                text="Start Presentation",
                command=self.show_main_app,
                style="Accent.TButton"
            )
            start_btn.pack(pady=30)
            return


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

        # Set UI elements with preference values
        if hasattr(self, 'delay_slider'):
            self.delay_slider.set(self.slide_delay)
        if hasattr(self, 'zoom_slider'):
            self.zoom_slider.set(self.base_zoom_factor)
        if hasattr(self, 'speed_slider'):
            self.speed_slider.set(self.pointer_speed_multiplier)
        if hasattr(self, 'dynamic_zoom_var'):
            self.dynamic_zoom_var.set(self.dynamic_zoom)
        if hasattr(self, 'posture_validation_var'):
            self.posture_validation_var.set(self.validate_posture)
        if hasattr(self, 'toggle_mode_nearby_var'):
            self.toggle_mode_nearby_var.set(self.enable_toggle_mode_nearby)
        if hasattr(self, 'toggle_mode_conference_var'):
            self.toggle_mode_conference_var.set(self.enable_toggle_mode_conference)
        if hasattr(self, 'camera_mode_var'):
            self.camera_mode_var.set(self.preferences.get('camera_mode', 'normal'))
        if hasattr(self, 'pose_estimation_var'):
            self.pose_estimation_var.set(self.enable_pose_estimation)
        if hasattr(self, 'furniture_detection_var'):
            self.furniture_detection_var.set(self.enable_furniture_detection)
        if hasattr(self, 'teacher_mode_var'):
            self.teacher_mode_var.set(self.teacher_mode_active)
        if hasattr(self, 'voice_control_var'):
            self.voice_control_var.set(self.enable_voice_control)
        if hasattr(self, 'voice_lang_var'):
            self.voice_lang_var.set(self.preferences.get('voice_language', 'en'))

    # ======================
    # SETTINGS FUNCTIONS
    # ======================

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

        # Mode toggles
        toggle_nearby_frame = tk.Frame(control_frame, bg=self.colors["primary"])
        toggle_nearby_frame.pack(fill=tk.X, pady=5, padx=5)

        self.toggle_mode_nearby_var = tk.BooleanVar(value=self.enable_toggle_mode_nearby)
        tk.Checkbutton(
            toggle_nearby_frame,
            text="Enable Nearby Toggle Mode",
            variable=self.toggle_mode_nearby_var,
            command=self.toggle_mode_nearby_switch,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        ).pack(side=tk.LEFT, padx=10)

        toggle_conference_frame = tk.Frame(control_frame, bg=self.colors["primary"])
        toggle_conference_frame.pack(fill=tk.X, pady=5, padx=5)

        self.toggle_mode_conference_var = tk.BooleanVar(value=self.enable_toggle_mode_conference)
        tk.Checkbutton(
            toggle_conference_frame,
            text="Enable Conference Toggle Mode",
            variable=self.toggle_mode_conference_var,
            command=self.toggle_mode_conference_switch,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        ).pack(side=tk.LEFT, padx=10)

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

        # Camera Settings
        camera_frame = tk.LabelFrame(
            right_panel,
            text="4. Camera Settings",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        camera_frame.pack(fill=tk.X, pady=5, padx=5)

        camera_mode_frame = tk.Frame(camera_frame, bg=self.colors["primary"])
        camera_mode_frame.pack(fill=tk.X, pady=5, padx=5)

        self.camera_mode_var = tk.StringVar(value="normal")

        normal_btn = ttk.Radiobutton(
            camera_mode_frame,
            text="Normal Mode",
            variable=self.camera_mode_var,
            value="normal",
            command=self.set_camera_normal_mode,
            style="Toolbutton"
        )
        normal_btn.pack(side=tk.LEFT, padx=5, expand=True)

        high_btn = ttk.Radiobutton(
            camera_mode_frame,
            text="High Mode",
            variable=self.camera_mode_var,
            value="high",
            command=self.set_camera_high_mode,
            style="Toolbutton"
        )
        high_btn.pack(side=tk.LEFT, padx=5, expand=True)

        # Update camera mode description
        desc_label = tk.Label(
            camera_frame,
            text="Normal: Eye-level camera | High: Camera above head level",
            font=("Helvetica", 9),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            justify=tk.LEFT
        )
        desc_label.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Pointer settings
        pointer_frame = tk.LabelFrame(
            right_panel,
            text="5. Pointer Mode",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        pointer_frame.pack(fill=tk.X, pady=5, padx=5)

        # Pointer speed
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
            text="6. Zoom Settings",
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
        tk.Checkbutton(
            dynamic_zoom_frame,
            text="Enable Dynamic Zoom Adjustment",
            variable=self.dynamic_zoom_var,
            command=self.toggle_dynamic_zoom,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        ).pack(side=tk.LEFT, padx=10)

        # Posture settings
        posture_frame = tk.LabelFrame(
            right_panel,
            text="7. Posture Settings",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        posture_frame.pack(fill=tk.X, pady=5, padx=5)

        # Posture validation checkbox
        posture_validation_frame = tk.Frame(posture_frame, bg=self.colors["primary"])
        posture_validation_frame.pack(fill=tk.X, pady=5, padx=5)

        self.posture_validation_var = tk.BooleanVar(value=self.validate_posture)
        tk.Checkbutton(
            posture_validation_frame,
            text="Enable Posture Validation",
            variable=self.posture_validation_var,
            command=self.toggle_posture_validation,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        ).pack(side=tk.LEFT, padx=10)

        # Back button at bottom
        back_frame = tk.Frame(settings_frame, bg=self.colors["primary"])
        back_frame.pack(fill=tk.X, pady=10)

        back_btn = ttk.Button(
            back_frame,
            text="Back to Main",
            command=self.show_main_app
        )
        back_btn.pack(pady=10)

        # Pose Estimation Settings
        pose_frame = tk.LabelFrame(
            right_panel,
            text="8. Pose Estimation",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        pose_frame.pack(fill=tk.X, pady=5, padx=5)

        self.pose_estimation_var = tk.BooleanVar(value=self.enable_pose_estimation)
        pose_check = tk.Checkbutton(
            pose_frame,
            text="Enable Pose Estimation (Standing/Sitting Detection)",
            variable=self.pose_estimation_var,
            command=self.toggle_pose_estimation,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        pose_check.pack(side=tk.LEFT, padx=10)

        # Add Furniture Detection section
        furniture_frame = tk.LabelFrame(
            right_panel,
            text="9. Furniture Detection",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        furniture_frame.pack(fill=tk.X, pady=5, padx=5)

        self.furniture_detection_var = tk.BooleanVar(value=self.enable_furniture_detection)
        furniture_check = tk.Checkbutton(
            furniture_frame,
            text="Enable Classroom Mode (Detect chairs/tables)",
            variable=self.furniture_detection_var,
            command=self.toggle_furniture_detection,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        furniture_check.pack(side=tk.LEFT, padx=10)

        # Add teacher/student mode toggle
        mode_frame = tk.Frame(furniture_frame, bg=self.colors["primary"])
        mode_frame.pack(fill=tk.X, pady=5)

        self.teacher_mode_var = tk.BooleanVar(value=self.teacher_mode_active)
        teacher_check = tk.Checkbutton(
            mode_frame,
            text="Teacher Mode.",
            variable=self.teacher_mode_var,
            command=self.toggle_teacher_mode,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        teacher_check.pack(side=tk.LEFT, padx=10)

        # Voice Control Settings - Add this section to your settings
        voice_frame = tk.LabelFrame(
            left_panel,
            text="2. Voice Control",
            font=("Helvetica", 12, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        voice_frame.pack(fill=tk.X, pady=5, padx=5)

        # Voice control toggle
        voice_toggle_frame = tk.Frame(voice_frame, bg=self.colors["primary"])
        voice_toggle_frame.pack(fill=tk.X, pady=5, padx=5)

        self.voice_control_var = tk.BooleanVar(value=self.enable_voice_control)
        voice_check = tk.Checkbutton(
            voice_toggle_frame,
            text="Enable Voice Control",
            variable=self.voice_control_var,
            command=self.toggle_voice_control,
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"],
            selectcolor=self.colors["secondary"]
        )
        voice_check.pack(side=tk.LEFT, padx=10)

        # Language selection for voice control
        lang_frame = tk.Frame(voice_frame, bg=self.colors["primary"])
        lang_frame.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(
            lang_frame,
            text="Voice Languages:",
            font=("Helvetica", 11),
            bg=self.colors["primary"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=10)

        # Language selection
        self.voice_lang_var = tk.StringVar(value="en")
        lang_menu = ttk.Combobox(
            lang_frame,
            textvariable=self.voice_lang_var,
            values=["en"],
            state="readonly",
            width=10
        )
        lang_menu.pack(side=tk.LEFT, padx=5)
        lang_menu.bind("<<ComboboxSelected>>", self.change_voice_language)

        # Voice control test button
        test_voice_btn = ttk.Button(
            voice_frame,
            text="Test Voice Recognition",
            command=self.test_voice_recognition
        )
        test_voice_btn.pack(pady=5, padx=10)

        # Set UI elements with preference values
        if hasattr(self, 'delay_slider'):
            self.delay_slider.set(self.slide_delay)
        if hasattr(self, 'zoom_slider'):
            self.zoom_slider.set(self.base_zoom_factor)
        if hasattr(self, 'speed_slider'):
            self.speed_slider.set(self.pointer_speed_multiplier)
        if hasattr(self, 'dynamic_zoom_var'):
            self.dynamic_zoom_var.set(self.dynamic_zoom)
        if hasattr(self, 'posture_validation_var'):
            self.posture_validation_var.set(self.validate_posture)
        if hasattr(self, 'toggle_mode_nearby_var'):
            self.toggle_mode_nearby_var.set(self.enable_toggle_mode_nearby)
        if hasattr(self, 'toggle_mode_conference_var'):
            self.toggle_mode_conference_var.set(self.enable_toggle_mode_conference)
        if hasattr(self, 'camera_mode_var'):
            self.camera_mode_var.set(self.preferences.get('camera_mode', 'normal'))
        if hasattr(self, 'pose_estimation_var'):
            self.pose_estimation_var.set(self.enable_pose_estimation)
        if hasattr(self, 'furniture_detection_var'):
            self.furniture_detection_var.set(self.enable_furniture_detection)
        if hasattr(self, 'teacher_mode_var'):
            self.teacher_mode_var.set(self.teacher_mode_active)
        if hasattr(self, 'voice_control_var'):
            self.voice_control_var.set(self.enable_voice_control)
        if hasattr(self, 'voice_lang_var'):
            self.voice_lang_var.set(self.preferences.get('voice_language', 'en'))

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

    def create_camera_settings(self):
        self.clear_frame()
        self.current_screen_method = self.create_camera_settings

        frame = tk.Frame(self.root, bg=self.colors["primary"])
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="Camera Settings", font=("Helvetica", 20, "bold"),
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(pady=(10, 20))

        # Refresh button
        refresh_btn = ttk.Button(frame, text="Refresh Camera List",
                                 command=self.refresh_camera_list)
        refresh_btn.pack(pady=5)

        # Selection area
        select_frame = tk.Frame(frame, bg=self.colors["primary"])
        select_frame.pack(pady=10)

        tk.Label(select_frame, text="Select Camera:", font=self.colors["font"],
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(side=tk.LEFT, padx=5)

        self.camera_combo = ttk.Combobox(select_frame,
                                         values=[name for idx, name in self.available_cameras],
                                         state="readonly", width=40)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        # Set current selection
        if self.available_cameras:
            current_name = next((name for idx, name in self.available_cameras if idx == self.selected_camera_index),
                                self.available_cameras[0][1])
            self.camera_combo.set(current_name)

        # Apply button
        apply_btn = ttk.Button(select_frame, text="Apply",
                               command=self.apply_camera_selection)
        apply_btn.pack(side=tk.LEFT, padx=5)

        # Status label (for feedback)
        self.settings_status_label = tk.Label(frame, text="", bg=self.colors["primary"], fg=self.colors["text"])
        self.settings_status_label.pack(pady=5)

        # Back button
        back_btn = ttk.Button(frame, text="Back to Main",
                              command=lambda: [self.clear_settings_status(), self.show_main_app()])
        back_btn.pack(pady=20)

        # Store reference to the status label for use in other methods
        self.current_settings_status_label = self.settings_status_label

    def clear_settings_status(self):
        """Clear the settings status label and remove reference."""
        if hasattr(self, 'current_settings_status_label') and self.current_settings_status_label:
            self.current_settings_status_label.config(text="")
            self.current_settings_status_label = None

    def create_voice_settings(self):
        self.clear_frame()
        self.current_screen_method = self.create_voice_settings

        frame = tk.Frame(self.root, bg=self.colors["primary"])
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="Voice Settings", font=("Helvetica", 20, "bold"),
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(pady=(10, 20))

        # Refresh button
        refresh_btn = ttk.Button(frame, text="Refresh Microphone List",
                                 command=self.refresh_microphone_list)
        refresh_btn.pack(pady=5)

        # Selection area
        select_frame = tk.Frame(frame, bg=self.colors["primary"])
        select_frame.pack(pady=10)

        tk.Label(select_frame, text="Select Microphone:", font=self.colors["font"],
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(side=tk.LEFT, padx=5)

        self.microphone_combo = ttk.Combobox(select_frame,
                                             values=[name for idx, name in self.available_microphones],
                                             state="readonly", width=50)
        self.microphone_combo.pack(side=tk.LEFT, padx=5)

        # Set current selection
        if self.available_microphones and self.selected_microphone_index is not None:
            for idx, name in self.available_microphones:
                if idx == self.selected_microphone_index:
                    self.microphone_combo.set(name)
                    break

        # Apply button
        apply_btn = ttk.Button(select_frame, text="Apply",
                               command=self.apply_microphone_selection)
        apply_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.settings_status_label = tk.Label(frame, text="", bg=self.colors["primary"], fg=self.colors["text"])
        self.settings_status_label.pack(pady=5)

        # Back button
        back_btn = ttk.Button(frame, text="Back to Main",
                              command=lambda: [self.clear_settings_status(), self.show_main_app()])
        back_btn.pack(pady=20)

        self.current_settings_status_label = self.settings_status_label

    def change_theme(self, event=None):
        theme_name = self.theme_var.get()
        self.current_theme = theme_name
        self.colors = THEMES[theme_name]

        # Save to preferences
        self.preferences.set('theme', theme_name)

        # Reconfigure all styles
        self.configure_styles()

        # Only recreate the current screen if we're not applying preferences
        if not hasattr(self, '_applying_preferences') or not self._applying_preferences:
            # Determine current screen
            if hasattr(self, 'current_screen_method'):
                current_screen = self.current_screen_method
                current_screen()

        # Update status
        self.set_status(f"Theme changed to {theme_name}")

    # vvvvvvvvvvvvvvooooooooooooooooiiiiiiiiiiiiiiiccccccccccccceeeeeeeeeeeeeee section !!!!!!!!!!!!!!!!!!!!!
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################

    def toggle_voice_control(self):
        self.enable_voice_control = self.voice_control_var.get()

        if self.enable_voice_control:
            # Check if microphone is available before starting
            try:
                # Test microphone access
                with sr.Microphone() as source:
                    pass
                self.start_voice_recognition()
            except OSError as e:
                messagebox.showerror("Microphone Error",
                                     f"Cannot access microphone: {str(e)}\n\n"
                                     "Please check:\n"
                                     "1. Microphone is connected and working\n"
                                     "2. Microphone permissions are granted to this application\n"
                                     "3. No other application is using the microphone")
                self.enable_voice_control = False
                self.voice_control_var.set(False)
            except Exception as e:
                messagebox.showerror("Error", f"Error accessing microphone: {str(e)}")
                self.enable_voice_control = False
                self.voice_control_var.set(False)
        else:
            self.stop_voice_recognition()

    def start_voice_recognition(self):
        if self.voice_recognizer is None:
            try:
                self.voice_recognizer = AdaptiveVoiceRecognizer()
                if not self.voice_recognizer.microphone_available:
                    raise Exception("Microphone not available")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to initialize voice recognition: {str(e)}")
                self.enable_voice_control = False
                self.voice_control_var.set(False)
                return

        # Set language (if your recognizer supports it)
        languages = [self.voice_lang_var.get()]
        self.voice_recognizer.set_target_words(languages)

        self.voice_stop_event.clear()
        self.voice_listening_thread = threading.Thread(target=self.voice_listening_loop)
        self.voice_listening_thread.daemon = True
        self.voice_listening_thread.start()

    def stop_voice_recognition(self):
        self.voice_stop_event.set()
        if self.voice_listening_thread and self.voice_listening_thread.is_alive():
            self.voice_listening_thread.join(timeout=2.0)
        print("Voice recognition stopped")

    def voice_listening_loop(self):
        """Continuously listen for voice commands using the selected microphone."""
        # Add a small delay before starting to ensure everything is initialized
        time.sleep(0.5)

        while not self.voice_stop_event.is_set() and self.enable_voice_control:
            try:
                # Check if microphone is available
                if not self.voice_recognizer or not self.voice_recognizer.microphone_available:
                    print("Microphone not available, stopping voice recognition")
                    self.enable_voice_control = False
                    self.voice_control_var.set(False)
                    break

                # Prepare microphone arguments with selected device index (if any)
                mic_kwargs = {}
                if self.selected_microphone_index is not None:
                    mic_kwargs['device_index'] = self.selected_microphone_index
                    print(f"🎤 Using microphone device index: {self.selected_microphone_index}")
                else:
                    print("🎤 Using system default microphone")

                # Use the selected microphone
                with sr.Microphone(**mic_kwargs) as source:
                    # Adjust for ambient noise each iteration
                    self.voice_recognizer.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                    # Listen for audio with a short timeout
                    audio = self.voice_recognizer.recognizer.listen(
                        source,
                        timeout=1,
                        phrase_time_limit=3  # Increased to allow for number phrases
                    )

                # Try to recognize in the selected language
                recognized_text, detected_command, similarity = self.voice_recognizer.recognize_speech(audio)

                print(f"Recognized: '{recognized_text}', Detected: '{detected_command}', Similarity: {similarity:.2f}")

                # Check for forward/backward commands in the recognized text
                if recognized_text:
                    text_lower = recognized_text.lower()

                    # Define trigger words
                    forward_indicators = ["forward", "ahead", "next", "skip forward", "go forward", "move forward"]
                    backward_indicators = ["back", "backward", "previous", "go back", "move back", "skip back"]

                    # Extract number from the recognized text using the voice recognizer's method
                    number = self.voice_recognizer.extract_slide_number(recognized_text)
                    if number is None:
                        number = 1  # Default to 1 if no number is found

                    # Check if it's a forward command
                    is_forward = any(indicator in text_lower for indicator in forward_indicators)
                    is_backward = any(indicator in text_lower for indicator in backward_indicators)

                    if is_forward:
                        print(f"Moving FORWARD by {number} slides")
                        for _ in range(number):
                            pyautogui.press('right')
                            time.sleep(0.05)  # Delay between key presses

                    elif is_backward:
                        print(f"Moving BACKWARD by {number} slides")
                        for _ in range(number):
                            pyautogui.press('left')
                            time.sleep(0.2)  # Delay between key presses

                    # Handle other commands normally
                    elif detected_command:
                        if detected_command == "next":
                            pyautogui.press('right')
                            print(f"Voice command: NEXT (similarity: {similarity:.2f})")
                        elif detected_command == "previous":
                            pyautogui.press('left')
                            print(f"Voice command: PREVIOUS (similarity: {similarity:.2f})")
                        elif detected_command == "first":
                            pyautogui.hotkey('ctrl', 'shift', 'home')
                            print(f"Voice command: FIRST (similarity: {similarity:.2f})")
                        elif detected_command == "last":
                            pyautogui.hotkey('ctrl', 'shift', 'end')
                            print(f"Voice command: LAST (similarity: {similarity:.2f})")
                        elif detected_command == "exit":
                            pyautogui.press('esc')
                            print(f"Voice command: EXIT (similarity: {similarity:.2f})")

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                pass
            except sr.UnknownValueError:
                # Could not understand audio, continue listening
                pass
            except OSError as e:
                # Microphone hardware error (e.g., device unplugged, wrong index)
                if "Invalid error code" in str(e) or "Invalid number of channels" in str(e):
                    error_msg = f"Microphone access error: {e}"
                else:
                    error_msg = f"Error in voice recognition: {e}"
                print(error_msg)
                self.set_status(error_msg)
                self.enable_voice_control = False
                self.voice_control_var.set(False)
                break
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                time.sleep(1)  # Wait before retrying

    def change_voice_language(self, event=None):
        language = self.voice_lang_var.get()
        self.preferences.set('voice_language', language)

        # If voice recognition is running, restart it with the new language
        if self.enable_voice_control:
            self.stop_voice_recognition()
            self.start_voice_recognition()

    def test_voice_recognition(self):
        if not self.voice_recognizer:
            messagebox.showinfo("Info", "Please enable voice control first.")
            return

        # Simple test to check if voice recognition is working
        try:
            with sr.Microphone() as source:
                self.voice_recognizer.recognizer.adjust_for_ambient_noise(source)
                messagebox.showinfo("Test", "Please say 'next' or 'previous' now...")
                audio = self.voice_recognizer.recognizer.listen(source, timeout=3)

            # Try to recognize
            recognized_text = self.voice_recognizer.recognizer.recognize_google(
                audio,
                language=self.voice_recognizer.lang_codes.get(self.voice_lang_var.get(), "en-US")
            ).lower()

            messagebox.showinfo("Result", f"I heard: {recognized_text}")
        except sr.UnknownValueError:
            messagebox.showinfo("Result", "Could not understand audio")
        except sr.WaitTimeoutError:
            messagebox.showinfo("Result", "No speech detected")
        except OSError as e:
            if "Invalid error code" in str(e):
                messagebox.showerror("Microphone Error",
                                     "Cannot access microphone. Please check:\n"
                                     "1. Microphone is connected\n"
                                     "2. Microphone permissions are granted\n"
                                     "3. No other application is using the microphone")
            else:
                messagebox.showerror("Error", f"Microphone error: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error testing voice: {str(e)}")


    ####################################################
    ####################################################
    ####################################################
    ####################################################

    def process_control_mode_high(self, frame):
        try:
            # Process voice commands if voice control is enabled
            if self.enable_voice_control and not self.pointer_mode:
                cv2.putText(frame, "VOICE CONTROL ACTIVE", (frame.shape[1] - 250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # Initialize hand_results to None
            hand_results = None

            # Detect furniture if enabled
            chairs, tables = [], []
            if self.enable_furniture_detection:
                chairs, tables = self.detect_furniture(frame)
                for chair in chairs:
                    x1, y1, x2, y2 = map(int, chair)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "Chair", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                for table in tables:
                    x1, y1, x2, y2 = map(int, table)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, "Table", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Process pose detection only if enabled
            if self.enable_pose_estimation:
                pose_results = self.pose.process(rgb_frame)
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark

                    # INITIALISER avec valeurs par défaut
                    is_sitting = False
                    valid_posture = True
                    near_furniture = False

                    # ⚠️ CORRECTION : ONLY check posture if the feature is ENABLED
                    if self.enable_pose_estimation:
                        is_sitting = self.determine_posture(landmarks) == "SITTING"

                    if self.validate_posture:
                        valid_posture = self.check_posture(landmarks, w, h)

                    if self.enable_furniture_detection:
                        near_furniture = self.is_person_near_furniture(landmarks, chairs, tables, w, h)

                    # ⚠️ CORRECTION : ONLY apply teacher mode restrictions if enabled
                    if self.teacher_mode_active and (is_sitting or near_furniture):
                        # Student detected - disable control ONLY if teacher mode is active
                        if self.enable_pose_estimation or self.enable_furniture_detection:
                            cv2.putText(frame, "STUDENT DETECTED - CONTROL DISABLED",
                                        (w // 2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            return frame

                    # ⚠️ CORRECTION : ONLY validate posture if the feature is enabled
                    if (self.validate_posture and not valid_posture) or (self.enable_pose_estimation and is_sitting):
                        if self.validate_posture and not valid_posture:
                            cv2.putText(frame, "INVALID POSTURE - STAND STRAIGHT",
                                        (w // 2 - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if self.enable_pose_estimation and is_sitting:
                            cv2.putText(frame, "SITTING DETECTED - STAND UP TO CONTROL",
                                        (w // 2 - 250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Reset modes only if restrictions apply
                        if self.pointer_mode:
                            self.pointer_mode = False
                            pyautogui.hotkey('ctrl', 'l')
                        self.both_hands_up_start = 0
                        return frame

            current_time = time.time()

            # Control only if posture is valid and standing
            if not valid_posture or is_sitting:
                if not valid_posture:
                    cv2.putText(frame, "INVALID POSTURE - STAND STRAIGHT",
                                (w // 2 - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if is_sitting:
                    cv2.putText(frame, "SITTING DETECTED - STAND UP TO CONTROL",
                                (w // 2 - 250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.pointer_mode:
                    self.pointer_mode = False
                    pyautogui.hotkey('ctrl', 'l')
                self.both_hands_up_start = 0
                return frame

                # Distance calculation and toggle logic
                pixel_distance = abs(left_wrist.x - right_wrist.x) * w
                cm_per_pixel = 40 / (0.2 * w)
                distance_cm = pixel_distance * cm_per_pixel

                # Check for toggle modes
                hands_nearby = (20 <= distance_cm <= 70 and
                                left_wrist.visibility > 0.7 and
                                right_wrist.visibility > 0.7)

                # Conference mode dynamic threshold
                left_hand_dist = ((left_wrist.x - body_center_x) ** 2 + (left_wrist.y - body_center_y) ** 2) ** 0.5
                right_hand_dist = ((right_wrist.x - body_center_x) ** 2 + (right_wrist.y - body_center_y) ** 2) ** 0.5
                max_hand_dist = max(left_hand_dist, right_hand_dist)
                conference_threshold = 10 + 20 * max_hand_dist
                hands_conference = (distance_cm <= conference_threshold and
                                    left_wrist.visibility > 0.7 and
                                    right_wrist.visibility > 0.7)

                # Draw distance feedback
                left_x = int(left_wrist.x * w)
                right_x = int(right_wrist.x * w)
                cv2.line(frame, (left_x, 100), (right_x, 100), (0, 255, 255), 2)
                color = (0, 255, 0) if (hands_nearby or hands_conference) else (0, 0, 255)
                cv2.putText(frame, f"{distance_cm:.0f}cm", ((left_x + right_x) // 2, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Display dynamic threshold values
                if self.enable_toggle_mode_conference:
                    cv2.putText(frame, f"Conf Thresh: {conference_threshold:.1f}cm", (w - 250, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frame, f"Hand Dist: {max_hand_dist:.2f}", (w - 250, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Toggle mode activation logic (only if posture is valid and standing)
                if (self.enable_toggle_mode_nearby and hands_nearby) or \
                        (self.enable_toggle_mode_conference and hands_conference):
                    if self.both_hands_up_start == 0:
                        self.both_hands_up_start = current_time
                        cv2.putText(frame, "HOLD FOR 3s TO TOGGLE POINTER", (50, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    else:
                        # Dynamic hold time for conference mode
                        if self.enable_toggle_mode_conference and hands_conference:
                            dynamic_hold_time = max(1.0, self.both_hands_up_threshold * (1 - max_hand_dist * 2))
                            remaining_time = dynamic_hold_time - (current_time - self.both_hands_up_start)
                        else:  # Fixed time for nearby mode
                            dynamic_hold_time = self.both_hands_up_threshold
                            remaining_time = dynamic_hold_time - (current_time - self.both_hands_up_start)

                        if remaining_time > 0:
                            remaining = int(remaining_time) + 1
                            cv2.putText(frame, f"{remaining}", (w // 2, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                        if current_time - self.both_hands_up_start >= dynamic_hold_time:
                            self.pointer_mode = not self.pointer_mode
                            self.both_hands_up_start = 0
                            pyautogui.hotkey('ctrl', 'l')

                            if self.pointer_mode:
                                screen_width, screen_height = pyautogui.size()
                                pyautogui.moveTo(screen_width // 2, screen_height // 2)
                                self.last_center_time = current_time  # Initialize centering

                            status_msg = "POINTER MODE ON" if self.pointer_mode else "SLIDE MODE ON"
                            cv2.putText(frame, status_msg, (w // 2 - 100, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.both_hands_up_start = 0

                # Pointer Mode (joystick-centered)
                if self.pointer_mode:
                    screen_width, screen_height = pyautogui.size()
                    screen_center_x = screen_width // 2
                    screen_center_y = screen_height // 2

                    # CONFERENCE MODE: Hand-as-joystick
                    if self.enable_toggle_mode_conference:
                        # Use midpoint between wrists as control point
                        control_point_x = (left_wrist.x + right_wrist.x) / 2
                        control_point_y = (left_wrist.y + right_wrist.y) / 2

                        # Calculate relative position to body center (normalized -1 to 1)
                        rel_x = (control_point_x - body_center_x) * 2
                        rel_y = (control_point_y - body_center_y) * 2

                        # Apply deadzone to prevent drift
                        deadzone = 0.1
                        if abs(rel_x) < deadzone: rel_x = 0
                        if abs(rel_y) < deadzone: rel_y = 0

                        # Calculate movement with dynamic sensitivity
                        hand_dist_from_center = math.sqrt(rel_x ** 2 + rel_y ** 2)
                        dynamic_sensitivity = self.pointer_speed_multiplier * (1 + hand_dist_from_center * 3)

                        # Calculate cursor position
                        cursor_x = screen_center_x + int(rel_x * dynamic_sensitivity * 500)
                        cursor_y = screen_center_y + int(rel_y * dynamic_sensitivity * 500)

                        # Visual feedback
                        point_x = int(control_point_x * w)
                        point_y = int(control_point_y * h)
                        body_px = int(body_center_x * w)
                        body_py = int(body_center_y * h)
                        cv2.circle(frame, (point_x, point_y), 20, (0, 255, 255), -1)
                        cv2.line(frame, (body_px, body_py), (point_x, point_y), (0, 255, 0), 2)

                    # NEARBY MODE: Finger-as-joystick
                    elif self.enable_toggle_mode_nearby and hand_results and hand_results.multi_hand_landmarks:
                        hand_landmarks = hand_results.multi_hand_landmarks[0]
                        index_finger = hand_landmarks.landmark[8]
                        control_point_x = index_finger.x
                        control_point_y = index_finger.y

                        # Calculate relative to body center
                        rel_x = (control_point_x - body_center_x) * 2
                        rel_y = (control_point_y - body_center_y) * 2

                        # Apply deadzone
                        deadzone = 0.1
                        if abs(rel_x) < deadzone: rel_x = 0
                        if abs(rel_y) < deadzone: rel_y = 0

                        dynamic_sensitivity = self.pointer_speed_multiplier
                        cursor_x = screen_center_x + int(rel_x * dynamic_sensitivity * 500)
                        cursor_y = screen_center_y + int(rel_y * dynamic_sensitivity * 500)

                        # Visual feedback
                        point_x = int(control_point_x * w)
                        point_y = int(control_point_y * h)
                        cv2.circle(frame, (point_x, point_y), 10, (0, 255, 255), -1)

                    # Common cursor movement for both modes
                    cursor_x = max(0, min(screen_width, cursor_x))
                    cursor_y = max(0, min(screen_height, cursor_y))
                    pyautogui.moveTo(cursor_x, cursor_y, _pause=False)

                    # Auto-center the cursor periodically (every 2 seconds)
                    if current_time - self.last_center_time > 2.0:
                        pyautogui.moveTo(screen_center_x, screen_center_y, _pause=False)
                        self.last_center_time = current_time

                    # Display mode status
                    mode_name = "CONFERENCE" if self.enable_toggle_mode_conference else "NEARBY"
                    cv2.putText(frame, f"JOYSTICK MODE ({mode_name})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Slide Control Mode
                else:
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
                            cv2.putText(frame, "PREV SLIDE", (w // 2 - 70, h - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif left_hand_raised:
                            pyautogui.press('right')
                            self.last_gesture_time = current_time
                            cv2.putText(frame, "NEXT SLIDE", (w // 2 - 70, h - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            print(f"Error in process_control_mode_high: {e}")

        return frame

    def process_control_mode_normal(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # Voice control status
            if self.enable_voice_control and not self.pointer_mode:
                try:
                    cv2.putText(frame, "VOICE CONTROL ACTIVE", (w - 250, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                except:
                    pass

            # Furniture detection (only if enabled)
            chairs, tables = [], []
            if self.enable_furniture_detection:
                chairs, tables = self.detect_furniture(frame)
                for chair in chairs:
                    x1, y1, x2, y2 = map(int, chair)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "Chair", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                for table in tables:
                    x1, y1, x2, y2 = map(int, table)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, "Table", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Process pose detection
            pose_results = self.pose.process(rgb_frame)

            # Process hand detection (only if needed)
            hand_results = None
            if self.enable_toggle_mode_nearby or (self.pointer_mode and self.enable_toggle_mode_nearby):
                hand_results = self.hands.process(rgb_frame)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                # Get key body points
                left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
                right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]

                # Body center calculations
                body_center_x = (left_shoulder.x + right_shoulder.x) / 2
                body_center_y = (left_shoulder.y + right_shoulder.y) / 2
                eye_level = (left_eye.y + right_eye.y) / 2

                # ⚠️ CORRECTION : Initialize with default values
                is_sitting = False
                valid_posture = True
                near_furniture = False

                # ⚠️ CORRECTION : Only check if features are enabled
                if self.enable_pose_estimation:
                    is_sitting = self.determine_posture(landmarks) == "SITTING"

                if self.validate_posture:
                    valid_posture = self.check_posture(landmarks, w, h)

                if self.enable_furniture_detection:
                    near_furniture = self.is_person_near_furniture(landmarks, chairs, tables, w, h)

                # ⚠️ CORRECTION : Teacher mode only applies if relevant features are enabled
                teacher_restriction_active = False
                if self.teacher_mode_active:
                    if self.enable_pose_estimation and is_sitting:
                        teacher_restriction_active = True
                    if self.enable_furniture_detection and near_furniture:
                        teacher_restriction_active = True

                if teacher_restriction_active:
                    cv2.putText(frame, "STUDENT DETECTED - CONTROL DISABLED",
                                (w // 2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return frame

                current_time = time.time()

                # ⚠️ CORRECTION : Only validate if the features are enabled
                posture_restriction = False
                if self.validate_posture and not valid_posture:
                    cv2.putText(frame, "INVALID POSTURE - STAND STRAIGHT",
                                (w // 2 - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    posture_restriction = True

                if self.enable_pose_estimation and is_sitting:
                    cv2.putText(frame, "SITTING DETECTED - STAND UP TO CONTROL",
                                (w // 2 - 250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    posture_restriction = True

                if posture_restriction:
                    if self.pointer_mode:
                        self.pointer_mode = False
                        pyautogui.hotkey('ctrl', 'l')
                    self.both_hands_up_start = 0
                    return frame

                # Calculate distance between hands
                pixel_distance = abs(left_wrist.x - right_wrist.x) * w
                cm_per_pixel = 40 / (0.2 * w)
                distance_cm = pixel_distance * cm_per_pixel

                # Check for toggle modes
                hands_nearby = (20 <= distance_cm <= 70 and
                                left_wrist.visibility > 0.7 and
                                right_wrist.visibility > 0.7)

                # Conference mode dynamic threshold
                left_hand_dist = ((left_wrist.x - body_center_x) ** 2 + (left_wrist.y - body_center_y) ** 2) ** 0.5
                right_hand_dist = ((right_wrist.x - body_center_x) ** 2 + (right_wrist.y - body_center_y) ** 2) ** 0.5
                max_hand_dist = max(left_hand_dist, right_hand_dist)
                conference_threshold = 10 + 20 * max_hand_dist
                hands_conference = (distance_cm <= conference_threshold and
                                    left_wrist.visibility > 0.7 and
                                    right_wrist.visibility > 0.7)

                # Draw distance feedback
                left_x = int(left_wrist.x * w)
                right_x = int(right_wrist.x * w)
                cv2.line(frame, (left_x, 100), (right_x, 100), (0, 255, 255), 2)
                color = (0, 255, 0) if (hands_nearby or hands_conference) else (0, 0, 255)
                cv2.putText(frame, f"{distance_cm:.0f}cm", ((left_x + right_x) // 2, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Display dynamic threshold values
                if self.enable_toggle_mode_conference:
                    cv2.putText(frame, f"Conf Thresh: {conference_threshold:.1f}cm", (w - 250, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frame, f"Hand Dist: {max_hand_dist:.2f}", (w - 250, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Toggle mode activation logic
                if (self.enable_toggle_mode_nearby and hands_nearby) or \
                        (self.enable_toggle_mode_conference and hands_conference):
                    if self.both_hands_up_start == 0:
                        self.both_hands_up_start = current_time
                        cv2.putText(frame, "HOLD FOR 3s TO TOGGLE POINTER", (50, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    else:
                        # Dynamic hold time for conference mode
                        if self.enable_toggle_mode_conference and hands_conference:
                            dynamic_hold_time = max(1.0, self.both_hands_up_threshold * (1 - max_hand_dist * 2))
                            remaining_time = dynamic_hold_time - (current_time - self.both_hands_up_start)
                        else:  # Fixed time for nearby mode
                            dynamic_hold_time = self.both_hands_up_threshold
                            remaining_time = dynamic_hold_time - (current_time - self.both_hands_up_start)

                        if remaining_time > 0:
                            remaining = int(remaining_time) + 1
                            cv2.putText(frame, f"{remaining}", (w // 2, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                        if current_time - self.both_hands_up_start >= dynamic_hold_time:
                            self.pointer_mode = not self.pointer_mode
                            self.both_hands_up_start = 0
                            pyautogui.hotkey('ctrl', 'l')

                            if self.pointer_mode:
                                screen_width, screen_height = pyautogui.size()
                                pyautogui.moveTo(screen_width // 2, screen_height // 2)
                                self.last_center_time = current_time

                            status_msg = "POINTER MODE ON" if self.pointer_mode else "SLIDE MODE ON"
                            cv2.putText(frame, status_msg, (w // 2 - 100, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.both_hands_up_start = 0

                # Pointer Mode (joystick-centered)
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

                        # Calculate distance from body center for dynamic sensitivity
                        hand_dist_from_center = ((control_point_x - body_center_x) ** 2 +
                                                 (control_point_y - body_center_y) ** 2) ** 0.5
                        dynamic_sensitivity = self.pointer_speed_multiplier * (1 + hand_dist_from_center * 2)

                        # Visual feedback
                        point_x = int(control_point_x * w)
                        point_y = int(control_point_y * h)
                        body_px = int(body_center_x * w)
                        body_py = int(body_center_y * h)
                        cv2.circle(frame, (point_x, point_y), 20, (0, 255, 255), -1)
                        cv2.line(frame, (body_px, body_py), (point_x, point_y), (0, 255, 0), 2)

                        # Calculate relative movement with dynamic sensitivity
                        rel_x = (control_point_x - body_center_x) * 2
                        rel_y = (control_point_y - body_center_y) * 2

                        move_x = rel_x * dynamic_sensitivity * 500
                        move_y = rel_y * dynamic_sensitivity * 500

                        cursor_x = screen_center_x + int(move_x)
                        cursor_y = screen_center_y + int(move_y)

                        # Display dynamic sensitivity
                        cv2.putText(frame, f"Sensitivity: {dynamic_sensitivity:.1f}x", (w - 250, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # NEARBY MODE: Finger-as-joystick
                    elif self.enable_toggle_mode_nearby and hand_results and hand_results.multi_hand_landmarks:
                        hand_landmarks = hand_results.multi_hand_landmarks[0]
                        index_finger = hand_landmarks.landmark[8]
                        control_point_x = index_finger.x
                        control_point_y = index_finger.y
                        dynamic_sensitivity = self.pointer_speed_multiplier  # Fixed sensitivity for nearby mode

                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Visual feedback
                        point_x = int(control_point_x * w)
                        point_y = int(control_point_y * h)
                        cv2.circle(frame, (point_x, point_y), 10, (0, 255, 255), -1)

                        # Calculate relative movement (standard)
                        rel_x = (control_point_x - 0.5) * 2
                        rel_y = (control_point_y - 0.5) * 2

                        move_x = rel_x * dynamic_sensitivity * 500
                        move_y = rel_y * dynamic_sensitivity * 500

                        cursor_x = screen_center_x + int(move_x)
                        cursor_y = screen_center_y + int(move_y)

                    # Common cursor movement for both modes
                    cursor_x = max(0, min(screen_width, cursor_x))
                    cursor_y = max(0, min(screen_height, cursor_y))
                    pyautogui.moveTo(cursor_x, cursor_y, _pause=False)

                    # Display mode status
                    mode_name = "CONFERENCE" if self.enable_toggle_mode_conference else "NEARBY"
                    cv2.putText(frame, f"JOYSTICK MODE ({mode_name}) - Speed: {dynamic_sensitivity:.1f}x",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Slide Control Mode
                else:
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
                            cv2.putText(frame, "PREV SLIDE", (w // 2 - 70, h - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif left_hand_raised:
                            pyautogui.press('right')
                            self.last_gesture_time = current_time
                            cv2.putText(frame, "NEXT SLIDE", (w // 2 - 70, h - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            print(f"Error in process_control_mode_normal: {e}")

        return frame

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

    def process_pose_estimation(self, frame):
        """Process frame with pose estimation and return results"""
        if not self.enable_pose_estimation or not self.pose_estimation_model:
            return frame, None

        try:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image
            results = self.pose_estimation_model.process(rgb_frame)

            # Only process landmarks without drawing them
            if results.pose_landmarks:
                # Determine if standing or sitting
                posture = self.determine_posture(results.pose_landmarks.landmark)

                # Display posture status in bottom left corner with background for better visibility
                text = f"{posture}"
                font_scale = 1.2  # Larger font size
                thickness = 2
                text_color = (0, 255, 0) if posture == "STANDING" else (
                0, 0, 255)  # Green for standing, red for sitting

                # Get text size to create background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # Position in bottom left corner with some padding
                text_x = 20
                text_y = frame.shape[0] - 30  # 30 pixels from bottom

                # Draw background rectangle
                cv2.rectangle(frame,
                              (text_x - 10, text_y - text_height - 10),
                              (text_x + text_width + 10, text_y + 10),
                              (0, 0, 0), -1)  # Black background

                # Draw text
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            text_color, thickness)

                return frame, posture
            return frame, None
        except Exception as e:
            print(f"Error in pose estimation: {e}")
            return frame, None

    def determine_posture(self, landmarks):
        """Determine if the person is standing or sitting based on landmarks"""
        try:
            # Get key landmarks
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]

            # Calculate vertical positions
            shoulder_avg = (left_shoulder.y + right_shoulder.y) / 2
            hip_avg = (left_hip.y + right_hip.y) / 2
            knee_avg = (left_knee.y + right_knee.y) / 2

            # Calculate distances
            shoulder_hip_dist = abs(shoulder_avg - hip_avg)
            hip_knee_dist = abs(hip_avg - knee_avg)

            # Determine posture
            if hip_knee_dist < shoulder_hip_dist * 0.6:  # If knees are close to hips
                return "SITTING"
            else:
                return "STANDING"
        except Exception as e:
            print(f"Error determining posture: {e}")
            return "UNKNOWN"

    def determine_if_standing(self, landmarks):
        """Check if a person is standing based on pose landmarks."""
        try:
            # Key landmarks (hips, knees, ankles)
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Standing if hips > knees > ankles (y increases downward)
            hip_y = (left_hip.y + right_hip.y) / 2
            knee_y = (left_knee.y + right_knee.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2

            return (hip_y < knee_y < ankle_y)
        except:
            return False

    def initialize_pose_estimation(self):
        """Initialize the MediaPipe pose estimation model using bundled solution"""
        try:
            self.pose_estimation_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=Light, 1=Full, 2=Heavy
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

            # No need for separate model file with this approach
            self.pose_landmarker = None
            print("Pose estimation initialized using bundled MediaPipe model")
        except Exception as e:
            print(f"Could not initialize pose estimation: {e}")
            self.pose_estimation_model = None
            self.set_status(f"Pose estimation error: {str(e)}")

    def load_furniture_model(self):
        """Load YOLO model for furniture detection"""
        try:
            self.furniture_model = YOLO('yolov8n.pt')  # Load pretrained model
            self.set_status("Furniture detection model loaded")
        except Exception as e:
            self.set_status(f"Failed to load furniture model: {str(e)}")
            self.furniture_model = None

    def detect_furniture(self, frame):
        """Detect chairs and tables in the frame"""
        chairs = []
        tables = []

        if not self.enable_furniture_detection or self.furniture_model is None:
            return chairs, tables

        try:
            # Run object detection
            results = self.furniture_model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    label = self.furniture_model.names[class_id]

                    # Check for chairs (class 56 in COCO dataset)
                    if class_id == 56:  # Chair
                        chairs.append(box.xyxy[0].tolist())
                    # Check for tables (class 60 in COCO dataset)
                    elif class_id == 60 or label.lower() == 'dining table':  # Table
                        tables.append(box.xyxy[0].tolist())

            return chairs, tables
        except Exception as e:
            print(f"Error in furniture detection: {e}")
            return [], []

    def is_person_near_furniture(self, person_landmarks, chairs, tables, frame_width, frame_height):
        """Check if person is near furniture"""
        if not chairs and not tables:
            return False

        # Get person's hip position (center point)
        left_hip = person_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = person_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        person_x = (left_hip.x + right_hip.x) / 2
        person_y = (left_hip.y + right_hip.y) / 2

        # Check distance to chairs
        for chair in chairs:
            chair_center_x = (chair[0] + chair[2]) / 2 / frame_width
            chair_center_y = (chair[1] + chair[3]) / 2 / frame_height

            distance = math.sqrt((person_x - chair_center_x) ** 2 + (person_y - chair_center_y) ** 2)
            if distance < self.chair_distance_threshold:
                return True

        # Check distance to tables
        for table in tables:
            table_center_x = (table[0] + table[2]) / 2 / frame_width
            table_center_y = (table[1] + table[3]) / 2 / frame_height

            distance = math.sqrt((person_x - table_center_x) ** 2 + (person_y - table_center_y) ** 2)
            if distance < self.table_distance_threshold:
                return True

        return False

    def change_language(self, event=None):
        lang = self.lang_var.get()
        # You would implement language translation here
        self.set_status(f"Language changed to {lang}")

    def show_main_app(self):
        self.current_screen_method = self.show_main_app
        self.create_main_app()

    def show_settings(self):
        self.current_screen_method = self.show_settings
        self.create_settings()

    def show_about(self):
        self.current_screen_method = self.show_about
        self.create_about()

    def show_camera_error(self):
        """Show camera error message exactly once"""
        if hasattr(self, '_camera_error_shown') and self._camera_error_shown:
            return

        self._camera_error_shown = True

        error_window = tk.Toplevel(self.root)
        error_window.title("Camera Error")
        error_window.resizable(False, False)

        # Set window icon
        try:
            error_window.iconbitmap(self.icon_path)
        except:
            pass

        # Window styling
        error_window.configure(bg=self.colors["primary"])
        window_width = 400
        window_height = 200
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        error_window.geometry(
            f"{window_width}x{window_height}+"
            f"{(screen_width // 2) - (window_width // 2)}+"
            f"{(screen_height // 2) - (window_height // 2)}"
        )

        # Error message content - simplified to your requested format
        error_content = [
            "Camera Error",  # Line 0 (will be red and bold)
            "",  # Line 1 (empty space)
            "Another application is already using the camera",  # Line 2
            "",  # Line 3 (empty space)
        ]

        # Create and pack each line with specific formatting
        for i, line in enumerate(error_content):
            # First line is red and bold, others use normal text color
            fg_color = self.colors["error"] if i == 0 else self.colors["text"]
            font_weight = "bold" if i == 0 else "normal"

            label = tk.Label(
                error_window,
                text=line,
                font=(self.colors["font"][0], self.colors["font"][1], font_weight),
                bg=self.colors["primary"],
                fg=fg_color,
                justify=tk.CENTER
            )
            label.pack(pady=0)

        # OK button (exits app)
        ok_button = ttk.Button(
            error_window,
            text="OK",
            command=self.quit_application,
            style="Accent.TButton"
        )
        ok_button.pack(pady=10)

        # Ensure single instance behavior
        error_window.protocol("WM_DELETE_WINDOW", self.quit_application)
        error_window.grab_set()

    def set_status(self, message):
        if hasattr(self, 'status_var') and self.status_var is not None:
            try:
                self.status_var.set(message)
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.update()
            except Exception as e:
                print(f"Error updating status: {e}")

    def setup_camera(self):
        """Initialize camera with the currently selected index."""
        try:
            self.cap = cv2.VideoCapture(self.selected_camera_index)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            self.cap.set(3, self.width)
            self.cap.set(4, self.height)
            self._camera_error_shown = False
            self._camera_failure_count = 0
            print(f"Camera initialized: index {self.selected_camera_index}")
        except Exception as e:
            print(f"Camera error: {e}")
            if not self._camera_error_shown:
                self.show_camera_error()
            self.cap = None

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

    def set_nearby_mode(self):
        """Configure settings for nearby mode (small rooms, close distance)"""
        self.base_zoom_factor = 1.0
        self.dynamic_zoom = False
        self.validate_posture = False
        self.pointer_speed_multiplier = 2.0
        self.enable_toggle_mode_nearby = True
        self.enable_toggle_mode_conference = False
        self.enable_voice_control = False
        self.enable_furniture_detection = False
        self.teacher_mode_active = False
        self.enable_pose_estimation = False

        # Save to preferences
        self.preferences.set('base_zoom_factor', self.base_zoom_factor)
        self.preferences.set('dynamic_zoom', self.dynamic_zoom)
        self.preferences.set('validate_posture', self.validate_posture)
        self.preferences.set('pointer_speed', self.pointer_speed_multiplier)
        self.preferences.set('enable_toggle_mode_nearby', self.enable_toggle_mode_nearby)
        self.preferences.set('enable_toggle_mode_conference', self.enable_toggle_mode_conference)
        self.preferences.set('mode', 'nearby')
        self.preferences.set('teacher_mode', False)
        self.preferences.set('enable_furniture_detection', False)
        self.preferences.set('enable_voice_control', False)


        # Update UI elements if they exist
        if hasattr(self, 'zoom_slider'):
            self.zoom_slider.set(self.base_zoom_factor)
            self.dynamic_zoom_var.set(False)
            self.posture_validation_var.set(False)
            self.speed_slider.set(self.pointer_speed_multiplier)
            self.toggle_mode_nearby_var.set(True)
            self.toggle_mode_conference_var.set(False)

        self.set_status("Nearby mode activated: Optimized for small rooms with close distance (20-70cm gestures)")

    def set_conference_mode(self):
        """Configure settings for conference mode (large rooms, far distance)"""
        self.base_zoom_factor = 1.0
        self.dynamic_zoom = False
        self.validate_posture = False
        self.pointer_speed_multiplier = 7.0
        self.enable_toggle_mode_nearby = False
        self.enable_toggle_mode_conference = True
        self.enable_voice_control = False
        self.enable_furniture_detection = False
        self.teacher_mode_active = False
        self.enable_pose_estimation = True

        # Save to preferences
        self.preferences.set('base_zoom_factor', self.base_zoom_factor)
        self.preferences.set('dynamic_zoom', self.dynamic_zoom)
        self.preferences.set('validate_posture', self.validate_posture)
        self.preferences.set('pointer_speed', self.pointer_speed_multiplier)
        self.preferences.set('enable_toggle_mode_nearby', self.enable_toggle_mode_nearby)
        self.preferences.set('enable_toggle_mode_conference', self.enable_toggle_mode_conference)
        self.preferences.set('mode', 'conference')
        self.preferences.set('teacher_mode', False)
        self.preferences.set('enable_furniture_detection', False)
        self.preferences.set('enable_voice_control', False)


        # Update UI elements if they exist
        if hasattr(self, 'zoom_slider'):
            self.zoom_slider.set(self.base_zoom_factor)
            self.dynamic_zoom_var.set(False)
            self.posture_validation_var.set(False)
            self.speed_slider.set(self.pointer_speed_multiplier)
            self.toggle_mode_nearby_var.set(False)
            self.toggle_mode_conference_var.set(True)

        self.set_status("Conference mode activated: Optimized for large rooms with far distance (2-10cm gestures)")

    def set_custom_mode(self):
        self.zoom_slider.config(state=tk.NORMAL)
        self.dynamic_zoom_cb.config(state=tk.NORMAL)
        self.posture_cb.config(state=tk.NORMAL)
        self.speed_slider.config(state=tk.NORMAL)
        self.toggle_nearby_cb.config(state=tk.NORMAL)
        self.toggle_conference_cb.config(state=tk.NORMAL)
        self.delay_slider.config(state=tk.NORMAL)
        """Allow all settings to be customized manually"""
        self.set_status("Custom mode activated: All settings can be adjusted individually")

    def set_camera_normal_mode(self):
        """Set camera processing to normal mode"""
        self.camera_mode_var.set("normal")
        self.process_control_mode = self._original_process_control_mode
        self.preferences.set('camera_mode', 'normal')
        self.set_status("Camera mode set to: Normal (basic tracking)")

    def set_camera_high_mode(self):
        """Set camera processing to high mode"""
        self.camera_mode_var.set("high")
        self.process_control_mode = self._high_process_control_mode
        self.preferences.set('camera_mode', 'high')
        self.set_status("Camera mode set to: High (enhanced precision with auto-centering)")

    def setup_hotkeys(self):
        keyboard.add_hotkey('f12', self.toggle_interface)
        keyboard.add_hotkey('shift+f12', self.quit_application)

    def update_video(self):
        """Main video processing loop (runs in a separate thread)."""
        while self.running:
            try:
                if self.cap is None:
                    time.sleep(1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    self._camera_failure_count += 1
                    if self._camera_failure_count > self._camera_failure_threshold:
                        if not self._camera_error_shown:
                            self.show_camera_error()
                        # Attempt to reinitialize the camera
                        self.setup_camera()
                        self._camera_failure_count = 0
                    time.sleep(0.1)
                    continue
                else:
                    self._camera_failure_count = 0  # successful read resets counter

                # Mirror the frame horizontally
                frame = cv2.flip(frame, 1)

                # Process voice status display
                if self.enable_voice_control and not self.pointer_mode:
                    try:
                        cv2.putText(frame, "VOICE CONTROL ACTIVE", (frame.shape[1] - 250, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    except:
                        pass

                # Process pose estimation if enabled
                if self.enable_pose_estimation:
                    frame, posture = self.process_pose_estimation(frame)
                    if posture:
                        self.update_posture_status(posture)

                # Apply current control mode if active
                if self.control_active:
                    # Use the appropriate processing function (normal or high mode)
                    frame = self.process_control_mode(frame)

                # Update display if camera view is shown and the label exists
                if self.show_camera and hasattr(self, 'video_label') and self.video_label.winfo_exists():
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

    def update_slide_delay(self, value):
        """Update the delay between slide changes"""
        self.slide_delay = float(value)
        self.preferences.set('slide_delay', self.slide_delay)
        self.set_status(f"Slide delay set to: {self.slide_delay:.1f} seconds")

    def update_pointer_speed(self, value):
        """Update the pointer movement speed multiplier"""
        speed = float(value)
        self.pointer_speed_multiplier = speed
        self.preferences.set('pointer_speed', speed)

        if speed >= 7:
            warning = " (High speed - Not recommended for far distances)"
            color = self.colors["warning"]
        else:
            warning = ""
            color = self.colors["text"]

        self.set_status(f"Pointer speed set to: {speed:.1f}x{warning}")

        if hasattr(self, 'speed_slider'):
            self.speed_slider.config(fg=color if speed >= 7 else self.colors["text"])

    def update_zoom_level(self, value):
        """Update the base zoom level"""
        self.base_zoom_factor = float(value)
        self.preferences.set('base_zoom_factor', self.base_zoom_factor)
        self.set_status(f"Base zoom level set to: {self.base_zoom_factor:.1f}x")

    def update_posture_status(self, posture):
        if hasattr(self, 'posture_status_var'):
            self.posture_status_var.get()

    def update_camera_list(self):
        """Get friendly camera names using pygrabber."""
        self.available_cameras = []  # list of (index, name) tuples
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for idx, name in enumerate(devices):
                self.available_cameras.append((idx, name))
        except Exception as e:
            print(f"Could not enumerate cameras with pygrabber: {e}")
            # Fallback to OpenCV indexing
            self.available_cameras = []
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.available_cameras.append((i, f"Camera {i}"))
                    cap.release()

        if not self.available_cameras:
            self.available_cameras = [(0, "Default Camera")]

        # Ensure selected index is valid
        if self.selected_camera_index >= len(self.available_cameras):
            self.selected_camera_index = 0

    def update_microphone_list(self):
        """List only input microphone devices, filtering out unwanted ones."""
        self.available_microphones = []  # list of (index, name) tuples
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            exclude_names = [
                "stereo mix",
                "what u hear",
                "wave out",
                "aux",
                "headphones",
                "speakers",
            ]
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0 and i != 0:  # skip mapper
                    name = dev_info['name']
                    name_lower = name.lower()
                    if any(excl in name_lower for excl in exclude_names):
                        continue
                    self.available_microphones.append((i, name))
            p.terminate()
        except Exception as e:
            print(f"Could not enumerate microphones with pyaudio: {e}")
            import speech_recognition as sr
            names = sr.Microphone.list_microphone_names()
            self.available_microphones = [(i, name) for i, name in enumerate(names) if i != 0]

        # Ensure at least one entry
        if not self.available_microphones:
            self.available_microphones = [(0, "Default Microphone")]  # fallback, though 0 is mapper

        # Validate saved index: if not in list, set to first available
        valid_indices = [idx for idx, _ in self.available_microphones]
        if self.selected_microphone_index not in valid_indices:
            print(
                f"Saved microphone index {self.selected_microphone_index} not valid, resetting to {valid_indices[0] if valid_indices else 0}")
            self.selected_microphone_index = valid_indices[0] if valid_indices else 0
            self.preferences.set('microphone_index', self.selected_microphone_index)

    def toggle_dynamic_zoom(self):
        """Toggle dynamic zoom adjustment"""
        self.dynamic_zoom = self.dynamic_zoom_var.get()
        self.preferences.set('dynamic_zoom', self.dynamic_zoom)
        if self.dynamic_zoom:
            self.shoulder_distance_ref = None
            self.set_status("Dynamic zoom enabled - Zoom will adjust based on distance")
        else:
            self.set_status("Dynamic zoom disabled - Using fixed zoom level")

    def toggle_posture_validation(self):
        """Toggle posture validation"""
        self.validate_posture = self.posture_validation_var.get()
        self.preferences.set('validate_posture', self.validate_posture)
        if self.validate_posture:
            self.set_status("Posture validation enabled - System will only respond when standing straight")
        else:
            self.set_status("Posture validation disabled - System will respond to any posture")
            if hasattr(self, 'posture_status'):
                self.posture_status_var.set("Posture: Validation disabled")
                self.posture_status.config(fg=self.colors["text"])

    def toggle_mode_nearby_switch(self):
        """Toggle nearby mode gesture detection"""
        self.enable_toggle_mode_nearby = self.toggle_mode_nearby_var.get()
        self.preferences.set('enable_toggle_mode_nearby', self.enable_toggle_mode_nearby)
        if self.enable_toggle_mode_nearby:
            self.enable_toggle_mode_conference = False
            self.toggle_mode_conference_var.set(False)
            self.preferences.set('enable_toggle_mode_conference', False)
            self.set_status("Nearby toggle mode enabled  - Hold hands apart for 3s to switch modes")
        else:
            self.set_status("Nearby toggle mode disabled")

    def toggle_mode_conference_switch(self):
        """Toggle conference mode gesture detection"""
        self.enable_toggle_mode_conference = self.toggle_mode_conference_var.get()
        self.preferences.set('enable_toggle_mode_conference', self.enable_toggle_mode_conference)
        if self.enable_toggle_mode_conference:
            self.enable_toggle_mode_nearby = False
            self.toggle_mode_nearby_var.set(False)
            self.preferences.set('enable_toggle_mode_nearby', False)
            self.set_status("Conference toggle mode enabled  - Hold hands apart for 3s to switch modes")
        else:
            self.set_status("Conference toggle mode disabled")

    def toggle_pose_estimation(self):
        """Toggle pose estimation feature"""
        self.enable_pose_estimation = self.pose_estimation_var.get()
        self.preferences.set('enable_pose_estimation', self.enable_pose_estimation)
        if self.enable_pose_estimation:
            self.set_status("Pose estimation enabled - System will detect standing/sitting posture")
            if not self.pose_estimation_model:
                self.initialize_pose_estimation()
        else:
            self.set_status("Pose estimation disabled")
            if hasattr(self, 'posture_status_var'):
                self.posture_status_var.set("Posture: Detection off")
                self.posture_status.config(fg=self.colors["text"])

    def toggle_furniture_detection(self):
        """Toggle furniture detection feature"""
        self.enable_furniture_detection = self.furniture_detection_var.get()
        self.preferences.set('enable_furniture_detection', self.enable_furniture_detection)
        if self.enable_furniture_detection:
            if self.furniture_model is None:
                self.load_furniture_model()
            self.set_status("Furniture detection enabled - Will detect chairs and tables")
        else:
            self.set_status("Furniture detection disabled")

    def toggle_teacher_mode(self):
        """Toggle teacher/student mode"""
        self.teacher_mode_active = self.teacher_mode_var.get()
        self.preferences.set('teacher_mode_active', self.teacher_mode_active)
        status = "ON (Only teacher can control)" if self.teacher_mode_active else "OFF (Anyone can control)"
        self.set_status(f"Teacher mode {status}")

    def toggle_voice_control(self):
        self.enable_voice_control = self.voice_control_var.get()
        self.preferences.set('enable_voice_control', self.enable_voice_control)

        if self.enable_voice_control:
            # Check if microphone is available before starting
            try:
                # Test microphone access
                with sr.Microphone() as source:
                    pass
                self.start_voice_recognition()
            except OSError as e:
                messagebox.showerror("Microphone Error",
                                     f"Cannot access microphone: {str(e)}\n\n"
                                     "Please check:\n"
                                     "1. Microphone is connected and working\n"
                                     "2. Microphone permissions are granted to this application\n"
                                     "3. No other application is using the microphone")
                self.enable_voice_control = False
                self.voice_control_var.set(False)
            except Exception as e:
                messagebox.showerror("Error", f"Error accessing microphone: {str(e)}")
                self.enable_voice_control = False
                self.voice_control_var.set(False)
        else:
            self.stop_voice_recognition()

    def toggle_interface(self):
        if self.root.state() == 'normal':
            self.root.withdraw()
        else:
            self.root.deiconify()

    def toggle_high_mode(self):
        """Safely toggle high camera mode"""
        try:
            if hasattr(self, 'high_mode_var') and self.high_mode_var is not None:
                self.high_mode = self.high_mode_var.get()

                # Update posture validation state
                if hasattr(self, 'posture_validation_var'):
                    if self.high_mode:
                        self.posture_validation_var.set(False)
                        self.validate_posture = False
                        if hasattr(self, 'posture_cb'):
                            self.posture_cb.config(state=tk.DISABLED)
                    else:
                        if hasattr(self, 'posture_cb'):
                            self.posture_cb.config(state=tk.NORMAL)

                print(f"High mode {'enabled' if self.high_mode else 'disabled'}")
            else:
                print("Warning: high_mode_var not initialized")
        except Exception as e:
            print(f"Error in toggle_high_mode: {e}")

    def switch_camera(self, index):
        """Change the active camera to the given index (thread‑safe)."""
        print(f"🔄 Switching camera to index {index}...")
        self.set_status("⏳ Switching camera...")  # update main status bar
        # Update settings status label if visible
        if hasattr(self, 'current_settings_status_label') and self.current_settings_status_label:
            self.current_settings_status_label.config(text="⏳ Switching camera...", fg=self.colors["text"])

        new_cap = cv2.VideoCapture(index)
        if new_cap.isOpened():
            new_cap.set(3, self.width)
            new_cap.set(4, self.height)
            # Give the new camera a moment to initialise
            time.sleep(0.3)

            old_cap = self.cap
            self.cap = new_cap
            if old_cap:
                old_cap.release()

            self.selected_camera_index = index
            self.preferences.set('camera_index', index)
            self._camera_failure_count = 0
            self._camera_error_shown = False

            camera_name = next((name for idx, name in self.available_cameras if idx == index), f"Camera {index}")
            success_msg = f"✅ Camera switched to {camera_name}"
            print(success_msg)
            self.set_status(success_msg)
            if hasattr(self, 'current_settings_status_label') and self.current_settings_status_label:
                self.current_settings_status_label.config(text=success_msg, fg="green")
        else:
            error_msg = f"❌ Failed to open camera {index}, keeping current camera"
            print(error_msg)
            self.set_status(error_msg)
            if hasattr(self, 'current_settings_status_label') and self.current_settings_status_label:
                self.current_settings_status_label.config(text=error_msg, fg="red")

    def switch_microphone(self, device_index):
        """Change the active microphone device (device_index is the actual PyAudio index)."""
        print(f"🔄 Switching microphone to device index {device_index}...")
        self.set_status("⏳ Switching microphone...")
        if hasattr(self, 'current_settings_status_label') and self.current_settings_status_label:
            self.current_settings_status_label.config(text="⏳ Switching microphone...", fg=self.colors["text"])

        try:
            name = next((n for idx, n in self.available_microphones if idx == device_index), f"Device {device_index}")

            self.selected_microphone_index = device_index
            self.preferences.set('microphone_index', device_index)

            success_msg = f"✅ Microphone switched to {name}"
            print(success_msg)
            self.set_status(success_msg)
            if hasattr(self, 'current_settings_status_label') and self.current_settings_status_label:
                self.current_settings_status_label.config(text=success_msg, fg="green")

            # If voice control is active, restart it with the new device
            if self.enable_voice_control:
                self.stop_voice_recognition()
                self.start_voice_recognition()
        except Exception as e:
            error_msg = f"❌ Error switching microphone: {e}"
            print(error_msg)
            self.set_status(error_msg)
            if hasattr(self, 'current_settings_status_label') and self.current_settings_status_label:
                self.current_settings_status_label.config(text=error_msg, fg="red")

    def apply_basic_preferences(self):
        """Apply only the basic preferences that are safe to apply before UI is fully initialized"""
        # Theme
        self.current_theme = self.preferences.get('theme', 'Default')
        self.colors = THEMES[self.current_theme]

        # Basic settings
        self.slide_delay = self.preferences.get('slide_delay', 1.0)
        self.base_zoom_factor = self.preferences.get('base_zoom_factor', 1.5)
        self.dynamic_zoom = self.preferences.get('dynamic_zoom', True)
        self.validate_posture = self.preferences.get('validate_posture', True)
        self.pointer_speed_multiplier = self.preferences.get('pointer_speed', 2.5)
        self.enable_toggle_mode_nearby = self.preferences.get('enable_toggle_mode_nearby', True)
        self.enable_toggle_mode_conference = self.preferences.get('enable_toggle_mode_conference', False)
        self.enable_pose_estimation = self.preferences.get('enable_pose_estimation', False)
        self.enable_furniture_detection = self.preferences.get('enable_furniture_detection', False)
        self.teacher_mode_active = self.preferences.get('teacher_mode_active', True)
        self.enable_voice_control = self.preferences.get('enable_voice_control', False)
        # In apply_basic_preferences:
        self.selected_camera_index = self.preferences.get('camera_index', 0)
        self.selected_microphone_index = self.preferences.get('microphone_index', 0)  # default to first

        # Reconfigure styles with the loaded theme
        self.configure_styles()

    def apply_camera_selection(self):
        selected_name = self.camera_combo.get()
        for idx, name in self.available_cameras:
            if name == selected_name:
                self.set_status("⏳ Applying camera change...")
                self.switch_camera(idx)
                break

    def apply_microphone_selection(self):
        selected_name = self.microphone_combo.get()
        for idx, name in self.available_microphones:
            if name == selected_name:
                self.switch_microphone(idx)
                break

    def refresh_microphone_list(self):
        self.update_microphone_list()
        self.microphone_combo['values'] = [name for idx, name in self.available_microphones]
        # Set the combobox to the (now validated) selected index
        for idx, name in self.available_microphones:
            if idx == self.selected_microphone_index:
                self.microphone_combo.set(name)
                break

    def refresh_camera_list(self):
        self.update_camera_list()
        self.camera_combo['values'] = [name for idx, name in self.available_cameras]
        if self.available_cameras and self.selected_camera_index is not None:
            if 0 <= self.selected_camera_index < len(self.available_cameras):
                self.camera_combo.set(self.available_cameras[self.selected_camera_index][1])

    def start_video_thread(self):
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def start_control(self):
        self.control_active = True
        self.show_camera = True
        self.pointer_mode = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        if self.enable_toggle_mode_nearby:
            instructions = (
                "NEARBY MODE ACTIVE\n\n"
                "• Left hand up: Previous slide\n"
                "• Right hand up: Next slide\n"
                "• Hold hands near camera for 3s: Toggle pointer mode"
            )
        else:  # Conference mode
            instructions = (
                "CONFERENCE MODE ACTIVE\n\n"
                "• Stand straight facing camera\n"
                "• Left hand up: Previous slide\n"
                "• Right hand up: Next slide\n"
                "• Hold hands near camera for 3s: Toggle pointer mode"
            )

        self.set_status(instructions)

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

    def clear_frame(self):
        for widget in self.root.winfo_children():
            if widget not in [self.root.children['!menu']]:
                widget.destroy()

    def quit_application(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
        sys.exit()

    def on_closing(self):
        self.stop_voice_recognition()
        if self.voice_recognizer:
            self.voice_recognizer.save_user_profile()
        self.quit_application()


if __name__ == "__main__":
    if os.name == 'nt':
        windll.shell32.SetCurrentProcessExplicitAppUserModelID('com.tagemouati.pwcv')

    root = tk.Tk()
    app = PoseGestureControl(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()












"""This is an impressive and complex application! You've combined computer vision (MediaPipe, OpenCV), object detection (YOLO), voice recognition, and GUI management (Tkinter) into a single tool for hands-free presentation control.

Here is a breakdown of the functions in your `PoseGestureControl` class, highlighting their roles and the specific logic and mathematics they use. To keep this digestible, I've grouped the straightforward UI functions and expanded heavily on the core logic and math functions.

---

### 1. Initialization and UI Construction

These functions handle the setup of the Tkinter window, variables, and UI elements.

* **`__init__`**: Initializes the application state, loads user preferences, sets default variables (zoom, speed, thresholds), and triggers the setup of the camera and ML models.
* **`configure_styles`**, **`setup_icons`**, **`change_theme`**: Handle the visual appearance, color dictionaries, and OS-level icon rendering.
* **`create_menu`**, **`create_welcome_screen`**, **`create_main_app`**, **`create_settings`**, **`create_about`**: Construct the Tkinter widgets (buttons, frames, sliders) for different screens.
* **`show_main_app`**, **`show_settings`**, **`show_about`**, **`show_camera_error`**: Navigation functions that clear the current screen and call the respective `create_...` functions.

### 2. General Mathematics & Utilities

* **`map_value(self, value, in_min, in_max, out_min, out_max)`**
* **Role**: Remaps a number from one range to another, applying a speed multiplier based on the distance from the center.
* **Math**: It uses a linear interpolation formula modified by a multiplier.



To center the movement and apply the speed multiplier ():






### 3. Voice Control Logic

* **`toggle_voice_control`**, **`start_voice_recognition`**, **`stop_voice_recognition`**, **`change_voice_language`**, **`test_voice_recognition`**: Manage the lifecycle and settings of the speech recognition engine.
* **`voice_listening_loop`**
* **Role**: Runs continuously in a background thread to listen for microphone input.
* **Logic**: Captures audio in 0.5s to 3s chunks. It checks the transcript against lists of "forward" (e.g., "next", "ahead") and "backward" (e.g., "previous", "go back") trigger words. If triggered, it parses any numbers spoken to determine *how many* times to trigger the `pyautogui.press('right')` or `'left'` commands.



### 4. Posture and Body Mathematics

* **`calculate_shoulder_distance`**
* **Role**: Measures how wide the user is on-screen (used to determine distance from the camera).
* **Math**: 1D Euclidean distance normalized to the frame width ().




* **`check_posture` & `validate_posture**`
* **Role**: Ensures the user is standing straight and facing the camera before allowing control.
* **Logic & Math**:
1. **Vertical alignment**: Checks if  (remembering that Y increases downwards in images).
2. **Horizontal tilt (Slopes)**: Ensures shoulders and hips are relatively level.


3. **Proportionality**: Ensures the legs are straight by checking if the thigh length roughly equals the calf length.






* **`determine_posture`**
* **Role**: Classifies whether a user is "STANDING" or "SITTING".
* **Logic**: It calculates the vertical distance between shoulders and hips (), and hips to knees (). If , the system assumes the knees are bent upwards (sitting).



### 5. Furniture Detection (YOLO)

* **`load_furniture_model`**, **`detect_furniture`**: Loads the YOLOv8 weights and scans frames for objects classified as 56 (chair) or 60 (dining table).
* **`is_person_near_furniture`**
* **Role**: Disables control if a "student" is sitting at a desk.
* **Math**: Calculates the center point of the person's hips  and the center of the bounding box of the furniture . It uses the 2D Euclidean distance:



If this distance is below `chair_distance_threshold` or `table_distance_threshold`, it flags the person as near furniture.



### 6. Core Computer Vision Processing

* **`process_control_mode_normal` & `process_control_mode_high**`
* **Role**: These are the master functions that process every video frame. They tie together posture, furniture detection, and gesture recognition to trigger mouse/keyboard events.
* **Logic 1: Real-world distance approximation**: To estimate how far apart the user's hands are in real centimeters (), it uses a constant mapping based on the camera's field of view:



* **Logic 2: Conference Mode Dynamic Threshold**: Because hands look closer together when the person is further away, the threshold to trigger "Pointer Mode" scales dynamically based on how far the user is reaching outward.



* **Logic 3: Pointer Mode (Joystick Math)**: Translates hand positions into mouse cursor coordinates. It calculates a relative vector () from the body center to the hand, applies a deadzone (so minor twitches don't move the mouse), and calculates dynamic sensitivity based on how far the hand is stretched.



* **Logic 4: Slide Mode (Dynamic Zoom)**: Instead of physically moving the mouse, this mode waits for hand raises above the eye line to trigger left/right arrow keys. It calculates a dynamic zoom ratio based on the reference shoulder distance vs. current shoulder distance:



It then crops the frame centered on the body using this  value to keep the presenter uniformly sized on screen regardless of distance.



### 7. Application Loop and State Updates

* **`update_video`**: The core loop running in a separate thread. It captures a frame from the webcam, flips it (mirror effect), routes it through the selected processing mode (Normal or High), and sends it to the UI.
* **`update_display`**: Converts the OpenCV BGR frame array into a Tkinter-compatible PIL Image.
* **`set_nearby_mode`**, **`set_conference_mode`**, **`set_custom_mode`**, **`set_camera_normal_mode`**, **`set_camera_high_mode`**: Helper functions that bundle specific boolean flags and thresholds (like base zoom factors and speed multipliers) to fit different physical environments.
* **`setup_camera`**, **`setup_pose_detection`**: Connect to hardware (webcam 0) and initialize the MediaPipe static and tracking confidences.
* **`start_video_thread`**, **`start_control`**, **`stop_control`**, **`quit_application`**, **`on_closing`**: Manage the threading lifecycles, ensuring the camera releases memory and background threads close safely when the user exits the app.

Would you like to dive deeper into how to optimize any specific mathematical formula here, or are you looking to add a new gesture logic to the `process_control_mode`?"""