
# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.xception import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# import cv2
# import numpy as np
# import json
# import gdown  
# from PIL import Image
# import os

# # Define file paths  
# MODEL_URL = 'https://drive.google.com/uc?export=download&id=10w2aVwYSkkHpoDwzduZRHC9NRzwioPsL'  
# CLASS_INDICES_URL = 'https://drive.google.com/uc?export=download&id=1m_Uc0SuSDsJIS2zIFjCucQLQ6Af3hyjR' 

# MODEL_PATH = 'inception_finetuned_complete.keras'
# CLASS_INDICES_PATH = 'inception30fined_tuneclass_indices.json'

# # Function to download files from Google Drive
# def download_file_from_drive(url, destination):
#     if not os.path.exists(destination):
#         gdown.download(url, destination, quiet=False)

# # Download the model and class indices if not already present
# download_file_from_drive(MODEL_URL, MODEL_PATH)
# download_file_from_drive(CLASS_INDICES_URL, CLASS_INDICES_PATH)

# # Load the model
# model = load_model(MODEL_PATH)

# # Load class indices
# with open(CLASS_INDICES_PATH, 'r') as f:
#     class_indices = json.load(f)

# # Map class indices to class labels
# class_labels = {v: k for k, v in class_indices.items()}

# # Remedies for skin diseases
# remedies = {
#     "Psoriasis": [
#         "Moisturize: Apply a moisturizer daily, especially after washing your hands or bathing.",
#         "Take warm baths with colloidal oatmeal or Epsom salts. Avoid long or hot baths.",
#         "Avoid triggers like alcohol, smoking, and too much sun.",
#         "Reduce stress with yoga or meditation.",
#         "Use natural sunlight in controlled amounts.",
#         "Try fragrance-free products labeled 'for sensitive skin'."
#     ],
#     "Eczema": [
#         "Apply a thick moisturizer immediately after bathing.",
#         "Use mild, fragrance-free soaps and detergents.",
#         "Avoid scratching the affected area.",
#         "Wear soft, breathable fabrics like cotton.",
#         "Use a humidifier to keep air moist.",
#         "Apply cool compresses to soothe itching."
#     ],
#     "Ringworm": [
#         "Use antifungal creams or powders as prescribed.",
#         "Keep the affected area clean and dry.",
#         "Avoid sharing towels, clothes, or personal items.",
#         "Wash clothing and bedding in hot water.",
#         "Avoid excessive sweating in affected areas.",
#         "Wear loose-fitting clothing."
#     ],
#     "Melanoma": [
#         "Avoid direct sun exposure between 10 AM and 4 PM.",
#         "Use sunscreen with SPF 30 or higher.",
#         "Wear protective clothing, hats, and sunglasses.",
#         "Check your skin regularly for changes in moles.",
#         "See a dermatologist if you notice any irregular spots.",
#         "Avoid tanning beds and UV lamps."
#     ],
#     "Melanocytic Nevi": [
#         "Monitor moles for any changes in size, color, or shape.",
#         "Use sunscreen to protect against UV damage.",
#         "Avoid picking at or scratching moles.",
#         "Consult a dermatologist if a mole becomes painful or itchy.",
#         "Wear sun-protective clothing.",
#         "Keep an eye on family history of skin conditions."
#     ]
# }

# # Function to preprocess images
# def preprocess_image(image):
#     image = cv2.resize(image, (224, 224))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     return image

# # Streamlit App
# st.title("Skin Disease Detection")

# # Option to choose input type: File Upload or Camera
# input_type = st.radio("Choose Input Type", ('Upload Image', 'Use Camera'))

# if input_type == 'Upload Image':
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         # Display uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         # Preprocess the image
#         image = np.array(image)
#         preprocessed_image = preprocess_image(image)
        
#         # Make prediction
#         pred = model.predict(preprocessed_image)
#         pred_class = np.argmax(pred, axis=1)[0]
#         confidence = np.max(pred) * 100
        
#         # Display prediction
#         predicted_label = class_labels[pred_class]
#         st.write(f"Prediction: {predicted_label}")
#         st.write(f"Confidence: {confidence:.2f}%")

#         # Display remedies
#         if predicted_label in remedies:
#             st.write("### Suggested Remedies: ")
#             for remedy in remedies[predicted_label]:
#                 st.write(f"- {remedy}")

# elif input_type == 'Use Camera':
#     st.write("Activate your camera and take a picture.")
    
#     # Start the camera
#     camera = st.camera_input("Take a picture")
#     if camera is not None:
#         # Read the camera image
#         image = Image.open(camera)
#         st.image(image, caption="Captured Image", use_column_width=True)
        
#         # Preprocess the image
#         image = np.array(image)
#         preprocessed_image = preprocess_image(image)
        
#         # Make prediction
#         pred = model.predict(preprocessed_image)
#         pred_class = np.argmax(pred, axis=1)[0]
#         confidence = np.max(pred) * 100
        
#         # Display prediction
#         predicted_label = class_labels[pred_class]
#         st.write(f"Prediction: {predicted_label}")
#         st.write(f"Confidence: {confidence:.2f}%")
        
#         # Display remedies
#         if predicted_label in remedies:
#             st.write("### Suggested Remedies: ")
#             for remedy in remedies[predicted_label]:
#                 st.write(f"- {remedy}")

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
        "Moisturize: Apply a moisturizer daily, especially after washing your hands or bathing.",
        "Take warm baths with colloidal oatmeal or Epsom salts. Avoid long or hot baths.",
        "Avoid triggers like alcohol, smoking, and too much sun.",
        "Reduce stress with yoga or meditation.",
        "Use natural sunlight in controlled amounts.",
        "Try fragrance-free products labeled 'for sensitive skin'."
    ],
    "Eczema": [
        "Apply a thick moisturizer immediately after bathing.",
        "Use mild, fragrance-free soaps and detergents.",
        "Avoid scratching the affected area.",
        "Wear soft, breathable fabrics like cotton.",
        "Use a humidifier to keep air moist.",
        "Apply cool compresses to soothe itching."
    ],
    "Ringworm": [
        "Use antifungal creams or powders as prescribed.",
        "Keep the affected area clean and dry.",
        "Avoid sharing towels, clothes, or personal items.",
        "Wash clothing and bedding in hot water.",
        "Avoid excessive sweating in affected areas.",
        "Wear loose-fitting clothing."
    ],
    "Melanoma": [
        "Avoid direct sun exposure between 10 AM and 4 PM.",
        "Use sunscreen with SPF 30 or higher.",
        "Wear protective clothing, hats, and sunglasses.",
        "Check your skin regularly for changes in moles.",
        "See a dermatologist if you notice any irregular spots.",
        "Avoid tanning beds and UV lamps."
    ],
    "Melanocytic Nevi": [
        "Monitor moles for any changes in size, color, or shape.",
        "Use sunscreen to protect against UV damage.",
        "Avoid picking at or scratching moles.",
        "Consult a dermatologist if a mole becomes painful or itchy.",
        "Wear sun-protective clothing.",
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
