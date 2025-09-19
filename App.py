import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("resistor_model.h5")  # upload your trained model file
    return model

model = load_model()

# Classes (must match training)
classes = ["black","brown","red","orange","yellow","green","blue","violet","grey","white"]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎨 Resistor Band Color Classifier")
st.write("Upload a resistor image and the model will predict the band color.")

uploaded = st.file_uploader("📤 Upload resistor image", type=["jpg","png","jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224,224))
    x = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    # Predict
    pred = model.predict(x)
    pred_class = classes[np.argmax(pred)]
    confidence = np.max(pred)

    st.subheader(f"✅ Predicted band color: **{pred_class}**")
    st.write(f"Confidence: {confidence:.2f}")
