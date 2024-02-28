import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from keras.models import load_model

# Custom CSS styles
CUSTOM_CSS = """
<style>
h1 {
    color: #FF5733;
    text-align: center;
}
h2 {
    color: #FF5733;
}
.file-upload-btn {
    background-color: #FF5733;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
}
.file-upload-btn:hover {
    background-color: #D35400;
}
</style>
"""

# Inject custom CSS into Streamlit
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Load the pre-trained model and define class names
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

# Function to display medicine recommendations
def display_medicine(predicted_class):
    recommendations = {
        'Tomato_Bacterial_spot': 'A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury, or hot compost the affected plants and DO NOT eat symptomatic fruit.',
        'Tomato_Early_blight': 'Cure the plant quickly otherwise the disease can be spread, Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable',
        'Tomato_Late_blight': 'Spraying fungicides is the most effective way to prevent late blight. For conventional gardeners and commercial producers, protectant fungicides such as chlorothalonil (e.g., Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate) can be used.',
        'Tomato_Leaf_Mold': 'Baking soda solution: Mix 1 tablespoon baking soda and ½ teaspoon liquid soap such as Castile soap (not detergent) in 1 gallon of water. Spray liberally, getting top and bottom leaf surfaces and any affected areas.',
        'Tomato_Septoria_leaf_spot': 'Fungicides with active ingredients such as chlorothalonil, copper, or mancozeb will help reduce disease, but they must be applied before disease occurs as they can only provide preventative protection. They will not cure the plant. If the disease has spread then remove the plants',
        'Tomato_Spider_mites_Two_spotted_spider_mite': 'Aiming a hard stream of water at infested plants to knock spider mites off the plants. Other options include insecticidal soaps, horticultural oils, or neem oil.',
        'Tomato__Target_Spot': 'Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials.',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers.',
        'Tomato__Tomato_mosaic_virus': 'Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. Monitor the rest of your plants closely, especially those that were located near infected plants. Disinfect gardening tools after every use.',
        'Tomato_healthy': 'Your plant is healthy, there is no need to apply medicines, please take care of your plants, if any disease occurs, then cure it fast and remove the infected leaves.'
    }
    return recommendations.get(predicted_class, "No recommendation available for this class.")


# Function to generate heatmap
def generate_heatmap(image, disease_mask):
    # Apply colormap to disease mask
    cmap = LinearSegmentedColormap.from_list('custom', [(0, 'green'), (1, 'red')])
    disease_heatmap = cmap(disease_mask)

    # Overlay heatmap on image
    overlaid_image = Image.fromarray((image * 255).astype(np.uint8))
    overlaid_image.putalpha(128)  # Set opacity to 50%
    overlaid_image = overlaid_image.convert("RGB")

    plt.imshow(overlaid_image)
    plt.imshow(disease_heatmap, alpha=0.5)
    plt.axis('off')
    return plt.gcf()
# Function to generate 3D surface plot
def generate_surface_plot(disease):
    # Generate example data (replace with actual data)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)
    
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(f'3D Surface Plot of {disease}')

    return fig

# Function to predict disease and generate heatmap
def predict_disease_and_generate_heatmap(image):
    # Resize image
    image_resized = image.resize((256, 256))

    # Convert image to numpy array
    img_array = np.array(image_resized)

    # Get predictions
    predictions = MODEL.predict(np.expand_dims(img_array, 0))
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return predicted_class, confidence, display_medicine(predicted_class)


# Function to display detection page
def detection_page():
    st.header("Detect")
    uploaded_file = st.file_uploader("Upload an image of tomato leaf", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        predicted_class, confidence, medicine = predict_disease_and_generate_heatmap(image)
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence}")
        st.write("Medicine for quick treatment:")
        st.info(medicine)


# Function to display heatmap page
def heatmap_page():
    st.header("Heatmap")
    uploaded_file = st.file_uploader("Upload an image of tomato leaf", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        predicted_class, confidence, _ = predict_disease_and_generate_heatmap(image)
        st.image(image, caption='Tomato Leaf Image', use_column_width=True)
        st.write(f"Heatmap for {predicted_class}")


# Function to display 3D surface plot page
def surface_plot_page():
    st.header("3D Surface Plot")
    disease = st.text_input("Enter the detected disease:", "Tomato_Bacterial_spot")
    
    # Button to generate surface plot
    if st.button("Generate 3D Surface Plot"):
        # Generate and display surface plot
        surface_plot_fig = generate_surface_plot(disease)
        st.pyplot(surface_plot_fig)



# Main code
st.sidebar.title("Navigation")
tabs = ["Detect", "Heatmap", "3D Surface Plot"]
choice = st.sidebar.radio("Go to", tabs)

if choice == "Detect":
    detection_page()
elif choice == "Heatmap":
    heatmap_page()
elif choice == "3D Surface Plot":
    surface_plot_page()
