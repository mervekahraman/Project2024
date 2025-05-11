
"""
SignSpeak: Real-Time Turkish Sign Language Recognition Web Application

This Streamlit application implements real-time Turkish Sign Language recognition using
a pre-trained LSTM model and MediaPipe for pose estimation. The app supports both
webcam input and video file uploads for sign language recognition.

Key Features:
- Real-time webcam processing
- Video file upload support
- Continuous sign language prediction
- Smooth prediction using rolling average
- User-friendly interface with Streamlit

Author: Merve Kahraman & İpek Doğa Yalçın & Seher Zeynep Sonkaya
Date: December 2024
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mediapipe as mp
from collections import deque

# Define constants
max_seq_length = 30 # Maximum number of frames to consider for prediction
threshold = 0.5  # Minimum confidence threshold for making predictions

# Class labels
class_labels = {
    0: "Ben", #30
    1: "Sevmek", #179
    2: "Anne", #14
    3: "Beklemek", #29
    4: "Ev" #64
}

from tensorflow.keras.models import load_model
# Load the pre-trained model
model = load_model("/Users/mervekahraman/Desktop/Bitirme_projesi/keypoint_model (1).h5")

# MediaPipe setup
mp_holistic = mp.solutions.holistic


def extract_keypoints_from_frame(frame):
    """
    Extract pose and hand keypoints from a single frame using MediaPipe Holistic.

    Args:
        frame: Input video frame

    Returns:
        np.array: Concatenated array of pose and hand keypoints
        [pose (132) + left_hand (63) + right_hand (63) = 258 features]
    """
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Convert BGR to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # Extract pose keypoints (x, y, z, visibility)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        # Extract left hand keypoints (x, y, z)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        # Extract right hand keypoints (x, y, z)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, lh, rh])


# Streamlit UI Setup
st.title("SignSpeak: Real-Time Turkish Sign Language Recognition")

# Input Selection Sidebar
option = st.sidebar.selectbox("Choose Input", ["Webcam", "Upload Video"])

# Webcam Processing Section
if option == "Webcam":
    st.write("Webcam live feed will appear below:")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()  # Placeholder for the video frames
        prediction_box = st.empty()  # Placeholder for predictions
        sequence = []  # Store keypoints from frames
        predictions = deque(maxlen=5)  # Rolling buffer for smoothing predictions

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # Extract keypoints and make predictions
            keypoints = extract_keypoints_from_frame(frame)
            sequence.append(keypoints)
            if len(sequence) > max_seq_length:
                sequence.pop(0)

            if len(sequence) == max_seq_length:
                input_sequence = pad_sequences([sequence], maxlen=max_seq_length, dtype='float32', padding='post',
                                               truncating='post')
                prediction = model.predict(input_sequence)[0]

                if np.max(prediction) > threshold:
                    predictions.append(np.argmax(prediction))

                if len(predictions) == predictions.maxlen:
                    most_common_prediction = max(set(predictions), key=predictions.count)
                    label = class_labels.get(most_common_prediction, "Unknown")
                else:
                    label = "Processing..."
            else:
                label = "Waiting for sufficient data..."

            # Update the prediction box
            prediction_box.markdown(f"### Prediction: `{label}`")

            # Display the webcam feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        cap.release()
    else:
        st.warning("Click the checkbox to start the webcam.")

# Video Upload functionality
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_video:
        video_path = f"./temp_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(video_path)

        st.write("Processing video...")
        cap = cv2.VideoCapture(video_path)
        sequence = []
        predictions = deque(maxlen=5)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract keypoints and make predictions
            keypoints = extract_keypoints_from_frame(frame)
            sequence.append(keypoints)
            if len(sequence) > max_seq_length:
                sequence.pop(0)

            if len(sequence) == max_seq_length:
                input_sequence = pad_sequences([sequence], maxlen=max_seq_length, dtype='float32', padding='post',
                                               truncating='post')
                prediction = model.predict(input_sequence)[0]

                if np.max(prediction) > threshold:
                    predictions.append(np.argmax(prediction))

                if len(predictions) == predictions.maxlen:
                    most_common_prediction = max(set(predictions), key=predictions.count)
                    label = class_labels.get(most_common_prediction, "Unknown")
                    st.write(f"Prediction: {label}")

        cap.release()
