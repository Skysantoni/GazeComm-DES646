from imutils import face_utils
from utils import * # Assuming utils.py has mouth_aspect_ratio, eye_aspect_ratio, direction
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import time
import pyperclip
import platform
import threading
import pyttsx3

# --- Helper function for non-blocking TTS ---
def speak_text(engine, text):
    """Run TTS in a separate thread to avoid blocking the main loop"""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
# --- END HELPER ---

# Determine the 'copy' hotkey based on the OS
OS_NAME = platform.system().lower()
if OS_NAME == 'darwin':  # macOS
    COPY_KEY = 'command'
else:  # Windows, Linux
    COPY_KEY = 'ctrl'

# Thresholds and consecutive frame length for triggering the mouse action.
MOUTH_AR_THRESH = 0.3
MOUTH_AR_CONSECUTIVE_FRAMES = 4
EYE_AR_THRESH = 0.24
EYE_AR_CONSECUTIVE_FRAMES = 5
WINK_AR_DIFF_THRESH = 0.05  # Increased threshold for clearer wink detection
WINK_CONSECUTIVE_FRAMES = 4

# Initialize the frame counters for each action as well as
# booleans used to indicate if action is performed or not
MOUTH_COUNTER = 0
EYE_COUNTER = 0
LEFT_WINK_COUNTER = 0
RIGHT_WINK_COUNTER = 0
INPUT_MODE = False
DRAG_MODE = False
ANCHOR_POINT = (0, 0)

#  On-screen notification variables 
notification_text = ""
notification_timestamp = 0.0
NOTIFICATION_DURATION = 1.5 # Show text for 1.5 seconds


# Define colors
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

# Initialize Dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# Initialize TTS Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150) # Speed of speech

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Video capture
vid = cv2.VideoCapture(1) # Try 0 or 1
cam_w = 640
cam_h = 480

while True:
    # Grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    _, frame = vid.read(0)
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # Determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    mouth = shape[mStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]

    # Because I flipped the frame, left is right, right is left.
    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    # Calculate aspect ratios
    mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    nose_point = (nose[3, 0], nose[3, 1])

    # Compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)

 

    # Check for WINK (Left or Right)
    if diff_ear > WINK_AR_DIFF_THRESH:
        
        # LEFT WINK - For Left Click
        if leftEAR < rightEAR:
            if leftEAR < EYE_AR_THRESH:
                LEFT_WINK_COUNTER += 1
                
                if LEFT_WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    print("LEFT CLICK")
                    notification_text = "LEFT CLICK" 
                    notification_timestamp = time.time() 
                    pag.click(button='left')
                    LEFT_WINK_COUNTER = 0 # Reset counter
            else:
                LEFT_WINK_COUNTER = 0

        # RIGHT WINK - For Right Click
        elif rightEAR < leftEAR:
            if rightEAR < EYE_AR_THRESH:
                RIGHT_WINK_COUNTER += 1
                
                if RIGHT_WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    print("RIGHT CLICK")
                    notification_text = "RIGHT CLICK" 
                    notification_timestamp = time.time() 
                    pag.click(button='right')
                    RIGHT_WINK_COUNTER = 0 # Reset counter
            else:
                RIGHT_WINK_COUNTER = 0
        
        # Reset other counters
        EYE_COUNTER = 0

    # Check for BLINK (Both Eyes) - For Drag Mode
    else:
        if ear <= 0.2:
            EYE_COUNTER += 1

            if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                DRAG_MODE = not DRAG_MODE
                
                if DRAG_MODE:
                    print("DRAG MODE ON")
                    notification_text = "DRAG MODE ON" # 
                    notification_timestamp = time.time()
                    pag.mouseDown(button='left')
                else:
                    print("DRAG MODE OFF - Copying and Speaking")
                    notification_text = "Copying and Speaking" 
                    notification_timestamp = time.time() 
                    pag.mouseUp(button='left')
                    
                    # Give OS time to register mouse up
                    time.sleep(0.1) 
                    pag.hotkey(COPY_KEY, 'c') 
                    time.sleep(0.1) 
                    
                    try:
                        selected_text = pyperclip.paste()
                        print("--- SELECTED TEXT ---")
                        print(selected_text)
                        print("---------------------")

                        if selected_text:
                            # Run in a thread to avoid freezing the CV loop
                            tts_thread = threading.Thread(
                                target=speak_text, 
                                args=(tts_engine, selected_text)
                            )
                            tts_thread.start()

                    except Exception as e:
                        print(f"Could not get text from clipboard or start TTS: {e}")
                
                EYE_COUNTER = 0 # Reset counter

        else:
            # Reset all eye counters if eyes are open and not winking
            EYE_COUNTER = 0
            LEFT_WINK_COUNTER = 0
            RIGHT_WINK_COUNTER = 0

    # Check for MOUTH OPEN - For Input Mode
    if mar > MOUTH_AR_THRESH:
        MOUTH_COUNTER += 1

        if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
            INPUT_MODE = not INPUT_MODE
            print(f"INPUT MODE: {INPUT_MODE}")
            MOUTH_COUNTER = 0
            ANCHOR_POINT = nose_point

    else:
        MOUTH_COUNTER = 0

    # --- MODE ACTIONS ---

    if INPUT_MODE:
        cv2.putText(frame, "INPUT MODE: ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 60, 35 # Sensitivity zone
        
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

        dir = direction(nose_point, ANCHOR_POINT, w, h)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        
        drag_speed = 15 # Pixels to move per frame
        if dir == 'right':
            pag.moveRel(drag_speed, 0)
        elif dir == 'left':
            pag.moveRel(-drag_speed, 0)
        elif dir == 'up':
            pag.moveRel(0, -drag_speed)
        elif dir == 'down':
            pag.moveRel(0, drag_speed)

    if DRAG_MODE: # <-- REMOVED
        cv2.putText(frame, 'DRAG MODE: ON', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2) # <-- REMOVED



    # Draw temporary notification text 
    if time.time() - notification_timestamp < NOTIFICATION_DURATION:
        # Set font properties for the notification
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        
        # Get text size to center it
        text_size = cv2.getTextSize(notification_text, font, font_scale, font_thickness)[0]
        text_x = (cam_w - text_size[0]) // 2
        text_y = 60 # 60 pixels from the top
    
        
        # Draw the (Red) notification text
        cv2.putText(frame, notification_text, (text_x, text_y), 
                    font, font_scale, RED_COLOR, font_thickness)
 

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `Esc` key was pressed, break from the loop
    if key == 27:
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vid.release()