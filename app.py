import cv2
import mediapipe as mp
import pyautogui
import streamlit as st
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Streamlit app
st.title("Virtual Mouse using Hand Gestures")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Streamlit placeholder for video feed
frame_placeholder = st.empty()

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables for smoothing cursor movement
smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Function to detect hand landmarks and control the mouse
def virtual_mouse(frame):
    global plocX, plocY

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_tip.x * frame.shape[1])
            index_finger_y = int(index_finger_tip.y * frame.shape[0])

            # Get the coordinates of the thumb tip (landmark 4)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])

            # Move the mouse cursor
            clocX = plocX + (index_finger_x - plocX) / smoothening
            clocY = plocY + (index_finger_y - plocY) / smoothening
            pyautogui.moveTo(screen_width - clocX * screen_width / frame.shape[1], clocY * screen_height / frame.shape[0])
            plocX, plocY = clocX, clocY

            # Check for left click (thumb and index finger close)
            if abs(index_finger_x - thumb_x) < 20 and abs(index_finger_y - thumb_y) < 20:
                pyautogui.click()

            # Check for right click (middle finger and index finger close)
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_x = int(middle_finger_tip.x * frame.shape[1])
            middle_finger_y = int(middle_finger_tip.y * frame.shape[0])
            if abs(index_finger_x - middle_finger_x) < 20 and abs(index_finger_y - middle_finger_y) < 20:
                pyautogui.rightClick()

    return frame

# Main loop to capture video and process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video.")
        break

    # Process the frame for virtual mouse
    processed_frame = virtual_mouse(frame)

    # Display the processed frame in the Streamlit app
    frame_placeholder.image(processed_frame, channels="BGR")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
