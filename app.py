import streamlit as st
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import colorsys
from keras.models import load_model

st.set_page_config(page_title='Machine Learning App with Random Forest')

st.title("Disease Detection in Tomato leaves")
st.text("Upload an image of tomato leaf")

MODEL = load_model("Tomato.h5")
CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

uploaded_file = st.file_uploader("Choose an image ...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Tomato Leaf Image')
    image = image.resize((256,256))
    img_array = np.array(image)
    
    # Get predictions
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    st.text('Prediction')
    st.info(predicted_class)
    st.text('Confidence')
    st.info(confidence)
    
    st.text('Medicine for a quick treatment')
    # Add medicine suggestions based on predicted_class
    
    # Convert RGB image to HSV
    hsv_image = colorsys.rgb_to_hsv(img_array[:, :, 0]/255, img_array[:, :, 1]/255, img_array[:, :, 2]/255)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    
    # Plot in 3D HSV space (Hue)
    fig_hue = plt.figure()
    ax_hue = fig_hue.add_subplot(111, projection='3d')
    ax_hue.scatter(np.arange(h.shape[1]), np.arange(h.shape[0]), h.flatten(), c='r', marker='o')
    ax_hue.set_xlabel('X')
    ax_hue.set_ylabel('Y')
    ax_hue.set_zlabel('Hue')
    st.pyplot(fig_hue)
    
    # Plot in 3D HSV space (Saturation)
    fig_sat = plt.figure()
    ax_sat = fig_sat.add_subplot(111, projection='3d')
    ax_sat.scatter(np.arange(s.shape[1]), np.arange(s.shape[0]), s.flatten(), c='g', marker='o')
    ax_sat.set_xlabel('X')
    ax_sat.set_ylabel('Y')
    ax_sat.set_zlabel('Saturation')
    st.pyplot(fig_sat)
    
    # Plot in 3D HSV space (Value)
    fig_val = plt.figure()
    ax_val = fig_val.add_subplot(111, projection='3d')
    ax_val.scatter(np.arange(v.shape[1]), np.arange(v.shape[0]), v.flatten(), c='b', marker='o')
    ax_val.set_xlabel('X')
    ax_val.set_ylabel('Y')
    ax_val.set_zlabel('Value')
    st.pyplot(fig_val)
