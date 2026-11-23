import sys
import os
from cx_Freeze import setup, Executable

# Get current directory and the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to Python's path so cx_Freeze can find "attempt_14" as a package
sys.path.insert(0, parent_dir)

# 1. Include necessary files
include_files = [
    (os.path.join(current_dir, "yolov8n.pt"), "yolov8n.pt"),
    (os.path.join(current_dir, "app.ico"), "app.ico"),
]

# Add user profile if it exists
pkl_path = os.path.join(current_dir, "user_pronunciation_profile.pkl")
if os.path.exists(pkl_path):
    include_files.append((pkl_path, "user_pronunciation_profile.pkl"))

# 2. FORCE include hidden dependencies AND your local package
build_exe_options = {
    "packages": [
        "os", "sys", "tkinter", "threading", "time", "json", "pickle", "pathlib",
        "cv2", "numpy", "mediapipe", "ultralytics",
        "pygame", "speech_recognition", "pyaudio", "audioop",
        "torch", "torchvision", "matplotlib",
        "attempt_14"  # <-- This packages your other .py files (cookies, voice recognizer, etc.)
    ],
    "excludes": [
        "scipy", "IPython", "jupyter", "notebook"
    ],
    "include_files": include_files,
    "silent": True
}

# 3. HIDE THE CONSOLE WINDOW
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Keeps the black CMD box hidden!

executable = Executable(
    script=os.path.join(current_dir, "App_v17_correctedBugs.py"),
    base=base,
    target_name="PWCV.exe",
    icon=os.path.join(current_dir, "app.ico")
)

setup(
    name="PWCV",
    version="4.5.0",
    description="PWCV App",
    options={"build_exe": build_exe_options},
    executables=[executable]
)