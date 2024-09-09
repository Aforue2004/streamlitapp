import streamlit as st  
from tensorflow.keras.models import load_model  
from tensorflow.keras.preprocessing.image import img_to_array  
import cv2  
import numpy as np  

# Title of the Streamlit app  
st.title("Breast Cancer Classification")  

# Load the saved model  
def load_trained_model():  
    model = load_model('testmodel2.h5')  # Replace 'testmodel.h5' with your actual model file  
    return model  

model = load_trained_model()  

# Function to preprocess the uploaded image  
def preprocess_image(image):  
    # Convert the PIL image to a NumPy array  
    img_array = np.array(image)  
    # Resize image to match the input size of the model (224x224)  
    img_resized = cv2.resize(img_array, (224, 224))  
    img_resized = img_to_array(img_resized)    # Convert image to array  
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension  
    img_resized = img_resized / 255.0               # Rescale pixel values to [0, 1]  
    return img_resized  

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
    labels = ['Unknown', 'benign', 'malignant', 'normal']  # Corrected syntax   

    # Display the prediction  
    predicted_class = labels[np.argmax(prediction)]  
    confidence = np.max(prediction)  
    
    st.write("Prediction:", predicted_class)  
    st.write("Confidence:", f"{confidence:.2%}")  

    # Optionally display probabilities for each class  
    for idx, label in enumerate(labels):  
        st.write(f"{label}: {prediction[0][idx]:.2%}")
