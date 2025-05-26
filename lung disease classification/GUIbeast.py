import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained Xception model
xception = load_model('xception_beast.h5')

# Define class names (replace with your own class names)
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

# Function to preprocess the image
def preprocess_image(img, target_size=(150, 150)):
    img = img.resize(target_size)  # Resize the image to match the model's input size
    img_array = image.img_to_array(img)  # Convert the image to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Streamlit app
st.title("Xception Model Tester")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)  # Open the image using PIL
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make a prediction using the Xception model
    prediction = xception.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index
    confidence = np.max(prediction)  # Get the confidence score

    # Display the result
    st.write(f"Predicted Class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")