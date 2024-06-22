import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('sign_language_model')

# Constants
LABELS = ['A', 'M', 'N', 'S', 'T', 'blank']
INPUT_SHAPE = (48, 48, 3)
IMAGE_SIZE = (INPUT_SHAPE[0], INPUT_SHAPE[1])
NUM_CLASSES = len(LABELS)

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array * 1./255  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)        # Preprocess using EfficientNet's method
    return img_array

# Function to classify the sign language gesture
def classify_gesture(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = LABELS[predicted_class_index]
    return predicted_class

# Streamlit application setup
st.title("Sign Language Classifier")

# Setting CSS style to align text in the middle
st.markdown(
    """
    <style>
    .h1 {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Start video capture
st.write("Starting video stream...")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break

    # Convert the frame to RGB and resize to the target size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((48, 48))

    # Classify the frame
    predicted_class = classify_gesture(img)

    # Display the frame with the predicted class
    frame_rgb = cv2.putText(frame_rgb, f'Predicted class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    FRAME_WINDOW.image(frame_rgb)

cap.release()

# Clear button to reset the interface
if st.button("Clear"):
    st.empty()
