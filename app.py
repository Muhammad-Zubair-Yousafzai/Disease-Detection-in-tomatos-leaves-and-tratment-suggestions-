import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('Tomato.h5')

# Define class names
class_names = ['Healthy', 'Gall Detected']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(model, image):
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Streamlit app
st.title("Tomato Leaf Disease Detection")

st.write("Upload an image of a leaf ")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    predicted_class, confidence = predict(model, processed_image)

    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Confidence: {confidence}%")

