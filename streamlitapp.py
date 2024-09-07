import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Title of the Streamlit app
st.title("Breast Cancer Classification")

# Load the saved model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('model_name.h5')  # Replace 'model_name.h5' with your actual model file
    return model

model = load_trained_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match the input size of the model
    img = img_to_array(img)         # Convert image to array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0               # Rescale pixel values to [0, 1]
    return img

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for the model
    img = preprocess_image(image)

    # Make prediction
    prediction = model.predict(img)

    # Define labels (you should replace these with your actual class names)
    labels = ['Unknown', 'Benign', 'Malignant', 'Normal']  # Modify based on your dataset

    # Display the prediction
    st.write("Prediction:", labels[np.argmax(prediction)])
    st.write("Confidence:", np.max(prediction))

    # Optionally display probabilities for each class
    for idx, label in enumerate(labels):
        st.write(f"{label}: {prediction[0][idx]:.2%}")
