from cvzone.HandTrackingModule import HandDetector
import cv2
import pyautogui
import numpy as np
import keyboard  # For global hotkey detection
import sys

# Parameters
width, height = 200, 200
gestureThreshold = width // 2

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
buttonPressed = False
counter = 0
delay = 30
show_camera = False  # Start hidden in background
should_exit = False  # Flag for exit condition

print("""
PowerPoint Gesture Control Running in Background
- Press F12 to show/hide the camera view
- Press Shift+F12 to exit completely
- Hand gestures work even when hidden
""")

def toggle_camera():
    global show_camera
    show_camera = not show_camera
    print("Camera view:", "shown" if show_camera else "hidden")

def exit_app():
    global should_exit
    print("Exiting application...")
    should_exit = True

# Register hotkeys
keyboard.add_hotkey('f12', toggle_camera)
keyboard.add_hotkey('shift+f12', exit_app)

while not should_exit:
    # Get image frame
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # Hand detection (works even when hidden)
    hands, img = detectorHand.findHands(img)
    if hands and not buttonPressed:
        hand = hands[0]
        cx, cy = hand["center"]
        fingers = detectorHand.fingersUp(hand)

        if fingers == [1, 1, 1, 1, 1]:
            if cx < gestureThreshold:
                pyautogui.press('left')
                print("Previous Slide")
                buttonPressed = True
            else:
                pyautogui.press('right')
                print("Next Slide")
                buttonPressed = True

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    # Display logic
    if show_camera:
        # Add UI overlay
        cv2.putText(img, "F12: Hide  |  Shift+F12: Exit", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("PowerPoint Gesture Control", img)
        # Check for 'q' when window is focused
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Ensure window is hidden
        if cv2.getWindowProperty("PowerPoint Gesture Control", cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow("PowerPoint Gesture Control")

# Cleanup
cap.release()
cv2.destroyAllWindows()
keyboard.unhook_all()
sys.exit()