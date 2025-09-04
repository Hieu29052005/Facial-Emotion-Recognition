import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("model.h5")

st.title("Image Classifier")
st.write("Upload an image and get predictions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    preds = model.predict(img_array)[0]

    st.write("Predictions:")
    for idx, prob in enumerate(preds):
        st.write(f"Class {idx}: {prob:.4f}")
