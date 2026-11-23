


# Maybe working but lot of technical issues..




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
import asyncio
from bleak import BleakScanner
import os


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
        self.slide_delay = 1.0
        self.width, self.height = 640, 480
        self.pose_threshold = 0.7
        self.gesture_threshold = 0.5

        # ======================
        # APPLICATION STATE
        # ======================
        self.interface_visible = True
        self.running = True
        self.control_active = False
        self.show_camera = False
        self.last_gesture_time = 0
        self.bluetooth_connected = False
        self.target_ble_mac = None

        # ======================
        # INITIALIZE COMPONENTS
        # ======================
        self.setup_hotkeys()
        self.setup_camera()
        self.setup_pose_detection()
        self.create_widgets()
        self.start_video_thread()
        self.start_ble_monitoring()

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

    def start_ble_monitoring(self):
        self.ble_thread = threading.Thread(target=self.run_ble_loop, daemon=True)
        self.ble_thread.start()

    def run_ble_loop(self):
        asyncio.run(self.ble_loop())

    async def ble_loop(self):
        while self.running:
            try:
                devices = await BleakScanner.discover(timeout=4.0)
                found = False
                for d in devices:
                    if self.target_ble_mac and d.address == self.target_ble_mac:
                        found = True
                        print(f"✅ Found selected device: {d.name} ({d.address})")
                        break
                self.bluetooth_connected = found
            except Exception as e:
                print(f"BLE scan error: {e}")
            await asyncio.sleep(5)

    # ======================
    # GUI METHODS
    # ======================
    def create_widgets(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(pady=10)
        self.video_label.pack_forget()

        self.status_var = tk.StringVar()
        self.set_status("Ready to start control")
        self.status_label = tk.Label(self.main_frame, textvariable=self.status_var, font=('Helvetica', 12),
                                     wraplength=400)
        self.status_label.pack(pady=5)

        self.create_ble_interface()
        self.create_delay_control()
        self.create_control_buttons()

    def create_ble_interface(self):
        ble_frame = tk.Frame(self.main_frame)
        ble_frame.pack(pady=5)

        tk.Label(ble_frame, text="Bluetooth Devices:").pack(side=tk.LEFT)

        self.device_list = ttk.Combobox(ble_frame, width=30)
        self.device_list.pack(side=tk.LEFT, padx=5)

        scan_button = ttk.Button(ble_frame, text="Scan", command=self.scan_ble_devices)
        scan_button.pack(side=tk.LEFT)

        select_button = ttk.Button(ble_frame, text="Select", command=self.set_target_ble_device)
        select_button.pack(side=tk.LEFT)

        open_bt_btn = ttk.Button(ble_frame, text="Open Bluetooth Settings", command=self.open_bluetooth_settings)
        open_bt_btn.pack(side=tk.LEFT)

    def scan_ble_devices(self):
        def scan():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            devices = loop.run_until_complete(BleakScanner.discover(timeout=5.0))
            device_names = [f"{d.name} ({d.address})" for d in devices if d.name]
            self.device_map = {f"{d.name} ({d.address})": d.address for d in devices if d.name}
            self.device_list["values"] = device_names
            if device_names:
                self.device_list.set(device_names[0])
            else:
                self.device_list.set("No devices found")

        threading.Thread(target=scan, daemon=True).start()

    def set_target_ble_device(self):
        selection = self.device_list.get()
        if selection in self.device_map:
            self.target_ble_mac = self.device_map[selection]
            self.set_status(f"Tracking device: {selection}")
        else:
            self.set_status("Invalid device selected.")

    def create_delay_control(self):
        delay_frame = tk.Frame(self.main_frame)
        delay_frame.pack(pady=5)

        tk.Label(delay_frame, text="Slide Delay:").pack(side=tk.LEFT)
        self.delay_slider = tk.Scale(delay_frame, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_slide_delay)
        self.delay_slider.set(self.slide_delay)
        self.delay_slider.pack(side=tk.LEFT, padx=5)

    def create_control_buttons(self):
        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start Control", command=self.start_control)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop Control", command=self.stop_control, state=tk.DISABLED)
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
        if not self.bluetooth_connected:
            cv2.putText(frame, "Waiting for trusted device...", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            center_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)

            zoom_factor = 1.5
            crop_w = int(self.width / zoom_factor)
            crop_h = int(self.height / zoom_factor)

            x1 = max(0, min(center_x - crop_w // 2, w - crop_w))
            y1 = max(0, min(center_y - crop_h // 2, h - crop_h))

            cropped_frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            frame = cv2.resize(cropped_frame, (self.width, self.height))

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
        self.set_status("Control active - Raise your hand above eye level to navigate")

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

    def open_bluetooth_settings(self):
        try:
            os.system("start ms-settings:bluetooth")
        except Exception as e:
            self.set_status(f"Error opening Bluetooth settings: {e}")


# ======================
# APPLICATION ENTRY POINT
# ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = PoseGestureControl(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
