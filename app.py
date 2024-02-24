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
    img_batch = np.expand_dims(image, 0)
    
    # Get predictions
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    st.text('Prediction')
    st.info(predicted_class)
    st.text('Confidence')
    st.info(confidence)
    
    st.text('Medicine for a quick treatment')
    if predicted_class == 'Tomato_Bacterial_spot':
        st.info('A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit.')
    elif predicted_class == 'Tomato_Early_blight':
        st.info('Cure the plant quickly otherwise the diease can be spread, Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable')
    # Add other conditions for disease and medicine
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Function to convert RGB image to HSV
    def rgb_to_hsv(rgb):
        return colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

    # Convert RGB image to HSV
    hsv_image = np.apply_along_axis(rgb_to_hsv, -1, img_array)
    
    # Extract HSV components
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]
    
    # Plot in 3D HSV space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(h.flatten(), s.flatten(), v.flatten(), c='r', marker='o')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')
    st.pyplot(fig)

