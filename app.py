import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np

# ----------------------------- Model Setup -----------------------------
# Optional patch for custom DepthwiseConv2D
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)

# Load pre-trained model
model = load_model(
    r"C:\Users\rani ghangare\OneDrive\Documents\Python Skill4Future Session\garbage[1]\garbage\keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D}
)

# Load class labels and clean them (remove number prefixes)
with open(r"C:\Users\rani ghangare\OneDrive\Documents\Python Skill4Future Session\garbage[1]\garbage\labels.txt", "r") as f:
    class_names = [label.strip().split(" ", 1)[-1] for label in f.readlines()]

# Mapping of waste types to R-methods
r_method_map = {
    "plastic": "Recycle ♻️",
    "paper": "Reuse 📄",
    "glass": "Recycle 🧪",
    "metal": "Recycle 🛠️",
    "organic": "Reduce 🌿",
    "e-waste": "Recycle ⚡",
    "textile": "Reuse 👕",
    "cardboard": "Reuse 📦",
    "hazardous": "Reduce 🚫",
    "other": "Reduce/Reuse ♻️"
}

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="🌱 Smart Waste Classifier & R-Method Recommender", page_icon="♻️", layout="centered")

# Sidebar information
with st.sidebar:
    st.title("🧭 About This Tool")
    st.markdown("""
    Upload an image of waste to identify its type using an AI model and receive a suitable action recommendation:
    
    - ♻️ Recycle
    - 👕 Reuse
    - 🌿 Reduce

    This tool helps promote **environmentally responsible disposal**.
    """)

# Title and subtitle (updated)
st.markdown("<h1 style='text-align: center;'>🌱 Smart Waste Classifier & R-Method Recommender</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>An AI-based tool for waste type prediction and sustainable action guidance</h4>", unsafe_allow_html=True)
st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image of the waste item", type=["jpg", "jpeg", "png"])

# Predict button
if st.button("🔍 Analyze Image"):
    if uploaded_file is not None:
        try:
            # Open and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Preprocess image for model
            image = image.convert("RGB")
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Make prediction
            with st.spinner("Analyzing..."):
                prediction = model.predict(data)
                index = np.argmax(prediction)
                predicted_label = class_names[index].strip()
                confidence = prediction[0][index]

                # Get R-method (safe fallback)
                label_key = predicted_label.lower().strip()
                r_method = r_method_map.get(label_key, "Dispose Responsibly ♻️")

            # Display result
            st.success("✅ Prediction Successful")
            col1, col2 = st.columns(2)
            col1.metric("Waste Type", predicted_label.title())
            col2.metric("Confidence", f"{confidence * 100:.2f}%")

            st.markdown(f"### 🧭 Recommended Action: <span style='color:green; background:#111;padding:5px;border-radius:4px;'> {r_method} </span>", unsafe_allow_html=True)
            st.info("Please dispose of this waste responsibly according to local regulations.")

        except Exception as e:
            st.error(f"❌ Error: Unable to process the image.\n\nDetails: {e}")
    else:
        st.warning("⚠️ Please upload an image first.")

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Developed with ❤️ by <strong>Twinkle Ghangare</strong> | Supported by EDUNET FOUNDATION</p>",
    unsafe_allow_html=True
)
