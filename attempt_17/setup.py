import sys
import os
from cx_Freeze import setup, Executable

# CRITICAL: Increase recursion limit before cx_Freeze starts
sys.setrecursionlimit(5000)

# Get current directory and the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to Python's path so cx_Freeze can find "attempt_17" as a package
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

# 2. Build options - simplified but complete
build_exe_options = {
    "packages": [
        "tkinter",
        "threading",
        "queue",
        "json",
        "pickle",
        "pathlib",
        "cv2",
        "numpy",
        "mediapipe",
        "ultralytics",
        "pyautogui",
        "keyboard",
        "speech_recognition",
        "pyaudio",
        "PIL",
        "torch",
        "torchvision",
        "attempt_17"  # Your local package
    ],
    "includes": [
        "tkinter.ttk",
        "http.client",
        "http.server",
        "urllib.request",
        "urllib.parse",
        "modulefinder",
        "inspect",
        "pkgutil",
        "importlib.metadata",
        "yaml",
        "tqdm",
    ],
    "excludes": [
        "scipy",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
        "tkinter.test",
        "distutils",
        "setuptools",
        "pip",
        "lib2to3",
        "test",
        "pytest",
    ],
    "include_files": include_files,
    "include_msvcr": True,  # Include Visual C++ redistributable
    "zip_exclude_packages": [],
    "optimize": 0,
    "silent": False  # Set to True if you want less output
}

# 3. HIDE THE CONSOLE WINDOW
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Keeps the black CMD box hidden!

executable = Executable(
    script=os.path.join(current_dir, "App_v20_nearbyModeFix.py"),  # Your current script
    base=base,
    target_name="PWCV.exe",
    icon=os.path.join(current_dir, "app.ico")
)

# 4. Run setup
print("\n" + "="*60)
print("STARTING CX_FREEZE BUILD...")
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Script: App_v21_nearbyChangeOfLogic.py")
print("="*60 + "\n")

setup(
    name="PWCV",
    version="4.5.0",
    description="PWCV App",
    options={"build_exe": build_exe_options},
    executables=[executable]
)

print("\n✅ Build completed!")
print(f"📂 Output folder: {os.path.join(current_dir, 'build')}")