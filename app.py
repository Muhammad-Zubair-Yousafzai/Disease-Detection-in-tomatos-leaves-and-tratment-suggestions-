import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from keras.models import load_model

st.set_page_config(page_title='Machine Learning App with Random Forest')

st.title("Disease Detection in Tomato leaves")
st.text("Upload an image of tomato leaf")

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



# Function to generate accuracy plot based on predictions
def generate_accuracy_plot(predictions, ground_truth):
    # Assuming predictions and ground_truth are arrays/lists of binary values (0 or 1)
    accuracy = np.mean(predictions == ground_truth)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy'], [accuracy], color='blue')
    plt.title('Accuracy Based on Predictions')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis limits between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plt.gcf()

# Sample predictions and ground truth labels (replace with actual data)
predictions = np.random.randint(2, size=100)  # Generate random binary predictions
ground_truth = np.random.randint(2, size=100)  # Generate random binary ground truth labels

# Generate accuracy plot
fig = generate_accuracy_plot(predictions, ground_truth)

# Display accuracy plot
st.pyplot(fig)



# Function to display medicine recommendations
def display_medicine(predicted_class):
    if predicted_class == 'Tomato_Bacterial_spot':
        st.info('A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit.')
    elif predicted_class == 'Tomato_Early_blight':
        st.info('Cure the plant quickly otherwise the disease can be spread, Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable')
    elif predicted_class == 'Tomato_Late_blight':
        st.info('Spraying fungicides is the most effective way to prevent late blight. For conventional gardeners and commercial producers, protectant fungicides such as chlorothalonil (e.g., Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate) can be used.')
    elif predicted_class == 'Tomato_Leaf_Mold':
        st.info('Baking soda solution: Mix 1 tablespoon baking soda and ½ teaspoon liquid soap such as Castile soap (not detergent) in 1 gallon of water. Spray liberally, getting top and bottom leaf surfaces and any affected areas.')
    elif predicted_class == 'Tomato_Septoria_leaf_spot':
        st.info('Fungicides with active ingredients such as chlorothalonil, copper, or mancozeb will help reduce disease, but they must be applied before disease occurs as they can only provide preventative protection. They will not cure the plant. If the disease has spread then remove the plants')
    elif predicted_class == 'Tomato_Spider_mites_Two_spotted_spider_mite':
        st.info('Aiming a hard stream of water at infested plants to knock spider mites off the plants. Other options include insecticidal soaps, horticultural oils, or neem oil.')
    elif predicted_class == 'Tomato__Target_Spot':
        st.info('Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials.')
    elif predicted_class == 'Tomato__Tomato_YellowLeaf__Curl_Virus':
        st.info('Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers.')
    elif predicted_class == 'Tomato__Tomato_mosaic_virus':
        st.info('Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. Monitor the rest of your plants closely, especially those that were located near infected plants. Disinfect gardening tools after every use.')
    elif predicted_class == 'Tomato_healthy':
        st.info('Your plant is healthy, there is no need to apply medicines, please take care of your plants, if any disease occurs, then cure it fast and remove the infected leaves.')

# Function to generate heatmap
def generate_heatmap(image, disease_mask, width=8, height=6):
    # Apply colormap to disease mask
    cmap = LinearSegmentedColormap.from_list('custom', [(0, 'green'), (1, 'red')])
    disease_heatmap = cmap(disease_mask)
    
    # Overlay heatmap on image
    overlaid_image = Image.fromarray((image * 255).astype(np.uint8))
    overlaid_image.putalpha(100)  # Set opacity to 50%
    overlaid_image = overlaid_image.convert("RGB")
    
    # Set figure size
    plt.figure(figsize=(width, height))
    
    plt.imshow(overlaid_image)
    plt.imshow(disease_heatmap, alpha=0.5)
    plt.axis('off')
    return plt.gcf()

# Function to generate accuracy plot
def generate_accuracy_plot(epochs, accuracy):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    return plt.gcf()

# Function to predict disease and generate heatmap
def predict_disease_and_generate_heatmap(image):
    # Display uploaded image
    st.image(image, caption='Tomato Leaf Image')

    # Resize image
    image_resized = image.resize((256, 256))

    # Convert image to numpy array
    img_array = np.array(image_resized)

    # Get predictions
    predictions = MODEL.predict(np.expand_dims(img_array, 0))
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Display prediction and confidence
    st.text('Prediction:')
    st.info(predicted_class)
    st.text('Confidence:')
    st.info(confidence)

    # Display medicine recommendation
    st.text('Medicine for quick treatment:')
    display_medicine(predicted_class)

    # Generate heatmap
    disease_mask = np.random.rand(img_array.shape[0], img_array.shape[1])  # Example random mask, replace with actual mask
    fig1 = generate_heatmap(img_array, disease_mask)
    
    # Generate accuracy plot (example data, replace with actual data)
    epochs = np.arange(1, 11)
    accuracy = np.random.rand(10)
    fig2 = generate_accuracy_plot(epochs, accuracy)

    # Display plots side by side
    st.pyplot(fig1, fig2)


# Main code
uploaded_file = st.file_uploader("Choose an image ...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    predict_disease_and_generate_heatmap(image)
