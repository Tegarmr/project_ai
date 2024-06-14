import os
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Load the trained models
image_model = load_model('modelCNN_ASL.h5')
video_model = load_model('smnist.h5')

# Define the labels for the ASL alphabet
image_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
video_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Initialize MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define a function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.reshape(image, (1, 64, 64, 3))
    return image

# Streamlit app
st.set_page_config(page_title="ASL Detection", layout="wide", page_icon="🖐")
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    .stApp {background-color: #ffffff;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖐 American Sign Language Detection")

# Image input section
st.header("Upload an Image of a Hand Sign")
st.write("This section allows you to upload an image of a hand sign to predict the corresponding ASL alphabet letter.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Classifying...')

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = image_model.predict(processed_image)
    predicted_label = image_labels[np.argmax(prediction)]

    st.success(f'The predicted letter is: {predicted_label}', icon="✅")

# Video input section
st.header("Real-time ASL Detection")
st.write("Use your webcam to detect ASL alphabet letters in real-time.")

frame_placeholder = st.empty()
user_input = st.text_input("Enter a video source URL (leave empty to use the webcam):")
url = 0 if not user_input else user_input

cap = cv2.VideoCapture(url)
_, frame = cap.read()
h, w, c = frame.shape

col1, col2 = st.columns(2)
start_btn_pressed = col1.button("Start", key="start")
stop_btn_pressed = col2.button("Stop", key="stop")

if start_btn_pressed:
    st.write("Press 'Stop' button to end the video stream.")
    while True and not stop_btn_pressed:
        ret, frame = cap.read()
        if not ret:
            break

        analysisframe = frame
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks

        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmarks in handLMsanalysis.landmark:
                    x, y = int(landmarks.x * w), int(landmarks.y * h)
                    x_min, x_max = min(x_min, x), max(x_max, x)
                    y_min, y_max = min(y_min, y), max(y_max, y)
                
                margin = 20
                x_min, x_max = max(0, x_min - margin), min(w, x_max + margin)
                y_min, y_max = max(0, y_min - margin), min(h, y_max + margin)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                mp_drawing.draw_landmarks(frame, handLMsanalysis, mphands.HAND_CONNECTIONS)

                try:
                    analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
                    analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                    analysisframe = cv2.resize(analysisframe, (28, 28))
                    flat_image = analysisframe.flatten()
                    datan = pd.DataFrame(flat_image).T
                    pixeldata = datan.values / 255
                    pixeldata = pixeldata.reshape(-1, 28, 28, 1)

                    # Prediction
                    prediction = video_model.predict(pixeldata)
                    predarray = np.array(prediction[0])

                    letter_prediction_dict = {video_labels[i]: predarray[i] for i in range(len(video_labels))}
                    letter, probabality = max(letter_prediction_dict.items(), key=lambda item: item[1])

                    letter = f"{letter} prob:{probabality:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    position = (x_max, y_min)
                    font_scale = round(h / 400)
                    font_color = (255, 255, 255)
                    font_thickness = round(h / 200)

                    cv2.putText(frame, letter, position, font, font_scale, font_color, font_thickness)

                except cv2.error:
                    pass

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
