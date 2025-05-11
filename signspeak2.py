"""
SignSpeak: Real-Time Turkish Sign Language Recognition Web Application

This Streamlit application implements real-time Turkish Sign Language recognition using
a pre-trained LSTM model and MediaPipe for pose estimation. The app supports both
webcam input and video file uploads for sign language recognition.

Key Features:
- Real-time webcam processing
- Video file upload support
- Natural language processing with transformers
- Dynamic sentence generation
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
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
from typing import List, Dict
import time
import speech_recognition as sr

# Initialize the Turkish BERT model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = AutoModelForMaskedLM.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = model.to('cpu')  # Explicitly move to CPU
    fill_mask_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1)  # Use CPU
except Exception as e:
    st.error(f"Error loading NLP model: {str(e)}")
    st.info("Please run: pip install transformers torch")
    fill_mask_pipeline = None

# Define constants
max_seq_length = 30
threshold = 0.5

# Class labels with their linguistic properties
class_labels = {
    0: {
        "word": "Ben",
        "pos": "PRON",
        "base": "ben",
        "person": 1,
        "number": "SING"
    },
    1: {
        "word": "Sevmek",
        "pos": "VERB",
        "base": "sev",
        "tense": "PRES"
    },
    2: {
        "word": "Anne",
        "pos": "NOUN",
        "base": "anne",
        "case": "NOM"
    },
    3: {
        "word": "Beklemek",
        "pos": "VERB",
        "base": "bekle",
        "tense": "PRES"
    },
    4: {
        "word": "Ev",
        "pos": "NOUN",
        "base": "ev",
        "case": "LOC"
    }
}

class TurkishNLPProcessor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.fill_mask = fill_mask_pipeline  # Use the pipeline we created above
        
        # Word type mapping
        self.word_types = {
            'Ben': 'SUBJECT',
            'Anne': 'OBJECT',
            'Ev': 'LOCATION',
            'Sevmek': 'VERB',
            'Beklemek': 'VERB'
        }
        
        # Turkish suffixes and word forms
        self.word_forms = {
            'Ben': {'base': 'ben'},
            'Anne': {'base': 'anne', 'accusative': 'annemi', 'nominative': 'annem'},
            'Ev': {'base': 'ev', 'locative': 'evde', 'accusative': 'evi'},
            'Sevmek': {'base': 'sev', 'present': 'seviyorum'},
            'Beklemek': {'base': 'bekle', 'present': 'bekliyorum'}
        }

    def _get_word_type_sequence(self, gesture_sequence: List[int]) -> List[str]:
        """Convert gesture sequence to word type sequence"""
        return [self.word_types[class_labels[idx]["word"]] for idx in gesture_sequence]

    def _rearrange_words(self, words: List[str], word_types: List[str]) -> List[str]:
        """Rearrange words according to Turkish word order (Subject-Object-Location-Verb)"""
        word_order = {'SUBJECT': 0, 'OBJECT': 1, 'LOCATION': 2, 'VERB': 3}
        word_pairs = list(zip(words, word_types))
        return [w for w, _ in sorted(word_pairs, key=lambda x: word_order.get(x[1], 99))]

    def generate_sentence(self, gesture_sequence: List[int]) -> str:
        """Generate a natural Turkish sentence from gesture sequence"""
        if not gesture_sequence:
            return ""

        # Get base words and their types
        words = [class_labels[idx]["word"] for idx in gesture_sequence]
        word_types = self._get_word_type_sequence(gesture_sequence)
        
        # Create a dictionary to track what we have
        components = {t: [] for t in ['SUBJECT', 'OBJECT', 'LOCATION', 'VERB']}
        for word, type_ in zip(words, word_types):
            components[type_].append(word)

        # Process words based on their role in the sentence
        processed_words = []
        
        # Add subject if present
        if components['SUBJECT']:
            processed_words.append('Ben')
        
        # Add object if present and there's a verb
        if components['OBJECT'] and components['VERB']:
            processed_words.append(self.word_forms['Anne']['accusative'])
        elif components['OBJECT']:
            processed_words.append(self.word_forms['Anne']['nominative'])
        
        # Add location
        if components['LOCATION']:
            processed_words.append(self.word_forms['Ev']['locative'])
        
        # Add verb (always at the end in Turkish)
        if components['VERB']:
            if 'Sevmek' in components['VERB']:
                processed_words.append(self.word_forms['Sevmek']['present'])
            if 'Beklemek' in components['VERB']:
                processed_words.append(self.word_forms['Beklemek']['present'])
        
        # Join words and capitalize
        sentence = ' '.join(processed_words)
        return sentence.capitalize() + '.'

# Initialize NLP processor
nlp_processor = TurkishNLPProcessor(model, tokenizer)

from tensorflow.keras.models import load_model
# Load the pre-trained model
model = load_model("/Users/mervekahraman/Desktop/Bitirme_projesi/keypoint_model (1).h5")

# MediaPipe setup
mp_holistic = mp.solutions.holistic

def extract_keypoints_from_frame(frame):
    """Extract pose and hand keypoints from a single frame using MediaPipe Holistic."""
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
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
        # Initialize webcam with specific parameters
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("Failed to open webcam. Please check your camera permissions.")
        else:
            stframe = st.empty()
            prediction_box = st.empty()
            sentence_box = st.empty()
            timer_box = st.empty()
            sequence = []
            predictions = deque(maxlen=10)
            gesture_sequence = []
            last_prediction = None
            current_sentence = ""
            
            # Timing variables
            collection_time = 15 * 30  # 15 seconds at 30 fps
            display_time = 5 * 30  # 5 seconds at 30 fps
            frame_counter = 0
            phase_start_time = time.time()
            is_collecting = True
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from webcam.")
                        break

                    # Flip the frame horizontally
                    frame = cv2.flip(frame, 1)
                    
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Update timing
                    current_time = time.time()
                    elapsed_time = current_time - phase_start_time
                    
                    if is_collecting:
                        # Collection phase (15 seconds)
                        remaining_time = 15 - int(elapsed_time)
                        if remaining_time > 0:
                            timer_box.markdown(f"### Time remaining for gestures: {remaining_time} seconds")
                            
                            # Process frame and make predictions
                            keypoints = extract_keypoints_from_frame(frame)
                            sequence.append(keypoints)
                            if len(sequence) > max_seq_length:
                                sequence.pop(0)

                            if len(sequence) == max_seq_length:
                                input_sequence = pad_sequences([sequence], maxlen=max_seq_length, dtype='float32', padding='post',
                                                            truncating='post')
                                prediction = model.predict(input_sequence)[0]

                                if np.max(prediction) > threshold:
                                    current_prediction = np.argmax(prediction)
                                    predictions.append(current_prediction)

                                    if len(predictions) == predictions.maxlen:
                                        most_common_prediction = max(set(predictions), key=predictions.count)
                                        
                                        if not gesture_sequence or most_common_prediction != gesture_sequence[-1]:
                                            gesture_sequence.append(most_common_prediction)
                                            last_prediction = most_common_prediction
                                            
                                            # Show current word
                                            current_word = class_labels[most_common_prediction]["word"]
                                            prediction_box.markdown(f"### Current Sign: {current_word}")
                                            
                                            # Update sentence
                                            current_sentence = nlp_processor.generate_sentence(gesture_sequence)
                                            sentence_box.markdown(f"### Building Sentence: {current_sentence}")
                        else:
                            # Switch to display phase
                            is_collecting = False
                            phase_start_time = current_time
                            if current_sentence:
                                sentence_box.markdown(f"### Final Sentence: {current_sentence}")
                            else:
                                sentence_box.markdown("### No sentence formed")
                    else:
                        # Display phase (5 seconds)
                        remaining_time = 5 - int(elapsed_time)
                        if remaining_time > 0:
                            timer_box.markdown(f"### New collection starts in: {remaining_time} seconds")
                        else:
                            # Reset for next round
                            is_collecting = True
                            phase_start_time = current_time
                            gesture_sequence = []
                            last_prediction = None
                            predictions.clear()
                            current_sentence = ""
                            sentence_box.empty()
                            prediction_box.empty()
                            timer_box.markdown("### Starting new collection phase...")

                    # Display the frame
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            except Exception as e:
                st.error(f"Error during webcam processing: {str(e)}")
            finally:
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



st.sidebar.markdown("---")
st.sidebar.markdown("### Turkish Speech-to-Text (Beta)")
speech_button = st.sidebar.button("Start Listening (Turkish)")

if speech_button:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="tr-TR")
        st.success(f"Transcribed Speech: `{text}`")
    except sr.UnknownValueError:
        st.error("Speech not understood. Please try again.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google STT; {e}")
