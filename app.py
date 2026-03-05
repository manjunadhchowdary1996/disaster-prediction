import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Page settings
st.set_page_config(page_title="Disaster Prediction", page_icon="🌍")

st.title("🌍 Disaster Prediction using CNN")
st.write("Upload an image to detect the disaster type.")

# Load trained model
@st.cache_resource
def load_my_model():
    model = load_model("sample_model.h5")
    return model

model = load_my_model()

# Class labels (must match your training dataset order)
classes = ["Cyclone", "Earthquake", "Flood", "Wildfire"]

# Upload image
uploaded_file = st.file_uploader("Upload Disaster Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Convert image to RGB (fix RGBA issue)
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Resize image
    img = img.resize((128,128))

    # Convert image to array
    img_array = np.array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Expand dimensions for model
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")
    st.success(f"Disaster Type: **{predicted_class}**")

    st.write(f"Confidence: **{confidence:.2f}%**")

    # Show probabilities
    st.subheader("Prediction Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{classes[i]}: {prob*100:.2f}%")