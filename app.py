import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_model-Brain-tumor.h5')

model = load_model()


class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    img = img.reshape((1, 256, 256, 3))
    img = img / 255.0
    return img

def predict_tumor(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return predicted_class_label, confidence

st.title('Brain Tumor MRI Classifier')

uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predicted_class, confidence = predict_tumor(image)

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

   