"""
Standalone test for eye‑closure detection using MediaPipe Face Mesh
with iris refinement for higher accuracy.

Improvements:
- Use refine_landmarks=True in FaceMesh for more precise eye landmarks.
- Added CLICK_COOLDOWN parameter to control delay between detections.
- All other UI and logic preserved exactly as in the original.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ------------------------------
# CONFIGURATION – adjust as needed
# ------------------------------
EAR_THRESHOLD = 0.20       # Default threshold (may be overridden by calibration)
HOLD_TIME = 0.3           # Seconds to hold eye closed for a simulated click
CLICK_COOLDOWN = 1.5       # Minimum seconds between two consecutive clicks
SHOW_LANDMARK_IDS = True   # Display the global landmark IDs next to each point

SMOOTH_WINDOW = 5          # Number of frames used for median smoothing of EAR

# Use widely used 6‑point indices for MediaPipe Face Mesh (works well for EAR)
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# ------------------------------
# Initialize MediaPipe Face Mesh with iris refinement
# ------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,          # <-- KEY CHANGE: enables more accurate eye landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ------------------------------
# EAR calculation function
# ------------------------------
EPS = 1e-6

def eye_aspect_ratio_px(landmarks, eye_indices, image_w, image_h):
    """
    Calculate EAR for one eye using 6‑point model, in pixel coordinates.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([int(lm.x * image_w), int(lm.y * image_h)], dtype=np.float32))

    p1, p2, p3, p4, p5, p6 = pts

    # vertical distances (two pairs)
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    # horizontal distance (corners)
    horizontal = np.linalg.norm(p1 - p4)

    ear = (vertical1 + vertical2) / (2.0 * (horizontal + EPS))
    return ear

# ------------------------------
# Helper: median smoothing buffer
# ------------------------------
class SmoothedValue:
    def __init__(self, maxlen=5):
        self.buf = deque(maxlen=maxlen)

    def add(self, value):
        self.buf.append(value)

    def median(self):
        if not self.buf:
            return None
        return float(np.median(np.array(self.buf, dtype=np.float32)))

    def mean(self):
        if not self.buf:
            return None
        return float(np.mean(np.array(self.buf, dtype=np.float32)))

    def clear(self):
        self.buf.clear()

# ------------------------------
# Main loop
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

left_smooth = SmoothedValue(maxlen=SMOOTH_WINDOW)
right_smooth = SmoothedValue(maxlen=SMOOTH_WINDOW)
avg_smooth = SmoothedValue(maxlen=SMOOTH_WINDOW)

# Variables for hold‑to‑click simulation
eye_closure_start = None
closure_eye = None
last_click_time = 0

# Calibration state
calibrating = False
calib_frames = []
CALIB_FRAMES_REQUIRED = 30
calibrated_threshold = None

print("Press 'q' to quit. Press 'c' to calibrate (capture open-eye EAR).")
print(f"Default EAR_THRESHOLD = {EAR_THRESHOLD} (can be updated by calibration)")
print(f"Click cooldown: {CLICK_COOLDOWN} seconds")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Mirror the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Compute EARs in pixel coordinates
        left_ear = eye_aspect_ratio_px(landmarks, LEFT_EYE_INDICES, w, h)
        right_ear = eye_aspect_ratio_px(landmarks, RIGHT_EYE_INDICES, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Add to smoothing buffers
        left_smooth.add(left_ear)
        right_smooth.add(right_ear)
        avg_smooth.add(avg_ear)

        left_ear_sm = left_smooth.median()
        right_ear_sm = right_smooth.median()
        avg_ear_sm = avg_smooth.median()

        # Draw the six landmark points for each eye
        for idx in LEFT_EYE_INDICES:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue for left eye
            if SHOW_LANDMARK_IDS:
                cv2.putText(frame, str(idx), (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for idx in RIGHT_EYE_INDICES:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green for right eye
            if SHOW_LANDMARK_IDS:
                cv2.putText(frame, str(idx), (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Decide effective threshold
        effective_threshold = calibrated_threshold if calibrated_threshold is not None else EAR_THRESHOLD

        # Determine closed status using smoothed average EAR
        left_closed = (left_ear_sm is not None) and (left_ear_sm < effective_threshold)
        right_closed = (right_ear_sm is not None) and (right_ear_sm < effective_threshold)

        current_time = time.time()

        # Display values (identical to original)
        cv2.putText(frame, f"L-EAR: {left_ear:.3f} sm:{left_ear_sm:.3f}" if left_ear_sm is not None else f"L-EAR: {left_ear:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"R-EAR: {right_ear:.3f} sm:{right_ear_sm:.3f}" if right_ear_sm is not None else f"R-EAR: {right_ear:.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"AVG-EAR: {avg_ear:.3f} sm:{avg_ear_sm:.3f}" if avg_ear_sm is not None else f"AVG-EAR: {avg_ear:.3f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        cv2.putText(frame, f"Threshold: {effective_threshold:.3f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        cv2.putText(frame, f"L-closed: {left_closed}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if left_closed else (0, 0, 255), 2)
        cv2.putText(frame, f"R-closed: {right_closed}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if right_closed else (0, 0, 255), 2)

        # Calibration capture
        if calibrating:
            calib_frames.append(avg_ear)
            cv2.putText(frame, f"Calibrating... {len(calib_frames)}/{CALIB_FRAMES_REQUIRED}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if len(calib_frames) >= CALIB_FRAMES_REQUIRED:
                open_avg = float(np.mean(calib_frames))
                calibrated_threshold = max(0.12, open_avg * 0.70)
                calibrating = False
                calib_frames = []
                print(f"Calibration done. open_avg={open_avg:.3f} -> calibrated_threshold={calibrated_threshold:.3f}")

        # Simulate a click if one eye is held closed for HOLD_TIME
        if (left_closed ^ right_closed):  # exactly one eye closed
            closed_eye = 'left' if left_closed else 'right'
            if eye_closure_start is None:
                eye_closure_start = current_time
                closure_eye = closed_eye
                cv2.putText(frame, f"Hold {closed_eye} eye for {HOLD_TIME}s to click", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            elif closure_eye == closed_eye:
                elapsed = current_time - eye_closure_start
                if elapsed >= HOLD_TIME:
                    # Use configurable cooldown
                    if current_time - last_click_time > CLICK_COOLDOWN:
                        print(f"*** SIMULATED CLICK! (held {closed_eye} eye for {HOLD_TIME}s) ***")
                        last_click_time = current_time
                    eye_closure_start = None
                    closure_eye = None
                else:
                    remaining = HOLD_TIME - elapsed
                    cv2.putText(frame, f"Hold for {remaining:.1f}s more", (10, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                # switched eye – reset
                eye_closure_start = None
                closure_eye = None
        else:
            eye_closure_start = None
            closure_eye = None

    else:
        # No face detected – clear buffers
        left_smooth.clear()
        right_smooth.clear()
        avg_smooth.clear()
        if calibrating:
            calib_frames = []

    cv2.imshow("Eye Closure Test - Press 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibrating = True
        calib_frames = []
        calibrated_threshold = None
        print("Calibration started: please keep your eyes open and face visible for ~1-2s...")

cap.release()
cv2.destroyAllWindows()