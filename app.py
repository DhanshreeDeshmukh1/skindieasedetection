
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import json
import gdown  
from PIL import Image
import os

# Define file paths  
MODEL_URL = 'https://drive.google.com/uc?export=download&id=10w2aVwYSkkHpoDwzduZRHC9NRzwioPsL'  
CLASS_INDICES_URL = 'https://drive.google.com/uc?export=download&id=1m_Uc0SuSDsJIS2zIFjCucQLQ6Af3hyjR' 

MODEL_PATH = 'inception_finetuned_complete.keras'
CLASS_INDICES_PATH = 'inception30fined_tuneclass_indices.json'

# Function to download files from Google Drive
def download_file_from_drive(url, destination):
    if not os.path.exists(destination):
        gdown.download(url, destination, quiet=False)

# Download the model and class indices if not already present
download_file_from_drive(MODEL_URL, MODEL_PATH)
download_file_from_drive(CLASS_INDICES_URL, CLASS_INDICES_PATH)

# Load the model
model = load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

# Map class indices to class labels
class_labels = {v: k for k, v in class_indices.items()}

# Remedies for skin diseases
remedies = {
    "Psoriasis": [
        "Application of emollients over the rashes is helpful",
        "Do not take steroid tablets as they can worsen psoriasis",
        "Quit smoking and alcoholism as they trigger psoriasis",
        "Reduce sun exposure",
        "Reduction of body weight helps in reducing flare-ups",
        "Use natural sunlight in controlled amounts."
    ],
    "Eczema": [
        "Proper moisturizer application will prevent eczema",
        "Do not apply any irritant over skin rashes",
        "Clean the eczematous rashes with saline",
        "Maintain cleanliness",
        "Wear soft, breathable fabrics like cotton.",
        "Use mild, fragrance-free soaps and detergents."
    ],
    "Tinea Ringworm": [
        "Wear loose garments",
        "Do not wear damp clothes",
        "Do not wear sweaty clothes",
        "If anybody at home has ringworm, wash their clothes separately",
        "Do not apply any steroid-containing creams over ringworm",
        "Do not apply oil or soaps or any irritants over the ringworm rashes"
    ],
    "Melanoma": [
        "Immediate treatment is mandatory",
        "Proper sunscreen application is essential",
        "Do not do home remedies. It needs surgical management and chemotherapsuy",
        "Whenever a mole increases in size drastically over a short period of time, conlt your dermatologist as early as possible"
        "Check your skin regularly for changes in moles.",
        "Avoid tanning beds and UV lamps."
    ],
    "Melanocytic Nevi": [
        "Application of emollients over the rashes is helpful",
        "Do not take steroid tablets as they can worsen psoriasis",
        "Quit smoking and alcoholism as they trigger psoriasis",
        "Reduce sun exposure",
        "Reduction of body weight helps in reducing flare-ups",
        "Keep an eye on family history of skin conditions."
    ]
}

# Function to preprocess images
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Streamlit App
st.title("🩺 Skin Disease Detection System")

# Add custom styling with black text
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: black;  /* Black Text */
    }
    .medium-font {
        font-size:24px !important;
        color: black;  /* Black Text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Option to choose input type: File Upload or Camera
input_type = st.radio("Choose Input Type", ('Upload Image', 'Use Camera'))

if input_type == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        
        # Preprocess the image
        image = np.array(image)
        preprocessed_image = preprocess_image(image)
        
        # Make prediction
        pred = model.predict(preprocessed_image)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred) * 100
        
        # Display prediction with increased font size
        predicted_label = class_labels[pred_class]
        st.markdown(f"<p class='big-font'> Prediction: {predicted_label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='big-font'> Confidence: {confidence:.2f}</p>", unsafe_allow_html=True)

        # Display remedies
        if predicted_label in remedies:
            st.write("### Suggested Remedies: ")
            for remedy in remedies[predicted_label]:
                st.write(f"- {remedy}")

elif input_type == 'Use Camera':
    st.write("📸 Activate your camera and take a picture.")
    
    # Start the camera
    camera = st.camera_input("Take a picture")
    if camera is not None:
        # Read the camera image
        image = Image.open(camera)
        st.image(image, caption="📷 Captured Image", use_column_width=True)
        
        # Preprocess the image
        image = np.array(image)
        preprocessed_image = preprocess_image(image)
        
        # Make prediction
        pred = model.predict(preprocessed_image)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred) * 100
        

        # Display prediction with increased font size
        predicted_label = class_labels[pred_class]
        st.markdown(f"<p class='big-font'> Prediction: {predicted_label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='big-font'> Confidence: {confidence:.2f}</p>", unsafe_allow_html=True)

        # Display remedies
        if predicted_label in remedies:
            st.write("###  Suggested Remedies: ")
            for remedy in remedies[predicted_label]:
                st.write(f"- {remedy}")
