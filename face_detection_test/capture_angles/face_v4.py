# ======================
# PERSON REGISTRATION INTERFACE
# ======================
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import os
import json
from datetime import datetime


class PersonRegistrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Person Registration Interface")
        self.root.geometry("800x600")

        # Camera settings
        self.width, self.height = 640, 480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

        # Registration settings
        self.registration_data = {
            "person_id": f"person_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created": datetime.now().isoformat(),
            "images": {
                "front": [],
                "left_side": [],
                "right_side": [],
                "back": []
            }
        }

        self.current_angle = "front"
        self.captured_count = 0
        self.total_needed = 5
        self.capture_delay = 1.0  # seconds between captures
        self.last_capture_time = 0
        self.is_capturing = False

        # Output directory
        self.output_dir = "registered_persons"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Create interface
        self.create_widgets()

        # Start video thread
        self.running = True
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Person Registration", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Instructions
        self.instructions_var = tk.StringVar()
        self.instructions_var.set("Please stand facing the camera for front view photos")
        instructions_label = ttk.Label(main_frame, textvariable=self.instructions_var,
                                       wraplength=600, justify=tk.CENTER)
        instructions_label.grid(row=2, column=0, columnspan=2, pady=10)

        # Progress
        self.progress_var = tk.StringVar()
        self.progress_var.set("Front view: 0/5 photos")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.grid(row=3, column=0, columnspan=2, pady=5)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)

        # Start capture button
        self.start_btn = ttk.Button(button_frame, text="Start Capture", command=self.start_capture)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        # Next angle button
        self.next_btn = ttk.Button(button_frame, text="Next Angle", command=self.next_angle, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # Save button
        self.save_btn = ttk.Button(button_frame, text="Save Registration", command=self.save_registration,
                                   state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def start_capture(self):
        self.is_capturing = True
        self.start_btn.config(state=tk.DISABLED)
        self.update_instructions()

    def next_angle(self):
        if self.current_angle == "front":
            self.current_angle = "left_side"
            self.instructions_var.set("Please turn to your left side for left side photos")
        elif self.current_angle == "left_side":
            self.current_angle = "right_side"
            self.instructions_var.set("Please turn to your right side for right side photos")
        elif self.current_angle == "right_side":
            self.current_angle = "back"
            self.instructions_var.set("Please turn your back to the camera for back photos")
        else:
            self.save_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.DISABLED)
            self.instructions_var.set("Registration complete! Click 'Save Registration' to finish.")
            return

        self.captured_count = 0
        self.update_progress()
        self.is_capturing = True
        self.next_btn.config(state=tk.DISABLED)

    def update_instructions(self):
        if self.current_angle == "front":
            self.instructions_var.set("Please stand facing the camera for front view photos")
        elif self.current_angle == "left_side":
            self.instructions_var.set("Please turn to your left side for left side photos")
        elif self.current_angle == "right_side":
            self.instructions_var.set("Please turn to your right side for right side photos")
        elif self.current_angle == "back":
            self.instructions_var.set("Please turn your back to the camera for back photos")

    def update_progress(self):
        angle_name = self.current_angle.replace("_", " ").title()
        self.progress_var.set(f"{angle_name}: {self.captured_count}/{self.total_needed} photos")

    def capture_image(self, frame):
        current_time = time.time()
        if current_time - self.last_capture_time < self.capture_delay:
            return False

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.registration_data['person_id']}_{self.current_angle}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        # Save image
        cv2.imwrite(filepath, frame)

        # Add to registration data
        self.registration_data['images'][self.current_angle].append({
            'filename': filename,
            'filepath': filepath,
            'timestamp': datetime.now().isoformat()
        })

        self.captured_count += 1
        self.last_capture_time = current_time
        self.update_progress()

        # Check if we've captured enough images for this angle
        if self.captured_count >= self.total_needed:
            self.is_capturing = False
            self.next_btn.config(state=tk.NORMAL)
            if self.current_angle != "back":
                self.instructions_var.set(
                    f"{self.current_angle.replace('_', ' ').title()} photos completed. Click 'Next Angle' to continue.")
            else:
                self.instructions_var.set("All photos completed! Click 'Next Angle' to finish.")

        return True

    def save_registration(self):
        # Save registration data to JSON file
        json_path = os.path.join(self.output_dir, f"{self.registration_data['person_id']}.json")
        with open(json_path, 'w') as f:
            json.dump(self.registration_data, f, indent=2)

        messagebox.showinfo("Registration Complete",
                            f"Person registration completed successfully!\n"
                            f"ID: {self.registration_data['person_id']}\n"
                            f"Images saved to: {self.output_dir}")

        self.root.destroy()

    def update_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                # Draw overlay instructions on the frame
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (self.width, 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                angle_text = self.current_angle.replace("_", " ").title()
                cv2.putText(frame, f"Angle: {angle_text}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(frame, f"Captured: {self.captured_count}/{self.total_needed}",
                            (self.width - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw a circle in the center to help with positioning
                center = (self.width // 2, self.height // 2)
                cv2.circle(frame, center, 10, (0, 255, 0), 2)
                cv2.circle(frame, center, 100, (0, 255, 0), 2)

                # Capture images if in capturing mode
                if self.is_capturing:
                    self.capture_image(frame.copy())
                    # Flash effect when capturing
                    if time.time() - self.last_capture_time < 0.1:
                        cv2.rectangle(frame, (0, 0), (self.width, self.height), (0, 255, 0), 30)

                # Display the frame
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            time.sleep(0.03)

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# Run the registration app
if __name__ == "__main__":
    root = tk.Tk()
    app = PersonRegistrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()