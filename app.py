import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat, ImageEnhance
import cv2
import os
import csv
import base64
from datetime import datetime
import time

# -------------------- STREAMLIT PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title=" Plant Leaf Classifier",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.title("🌿 Plant Leaf Condition Classifier")
st.write("Upload a plant leaf image, and the model will classify its condition.")


# --- Constants ---
MODEL_PATH = "vit_model.keras"
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
               'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy',
               'Cherry___Powdery_mildew', 'Cherry___healthy',
               'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
               'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
               'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
               'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

MITIGATION_STRATEGIES = {
    'Apple___Apple_scab': "Apply fungicides like captan. Practice proper pruning to increase airflow.",
    'Apple___Black_rot': "Remove infected fruit and twigs. Use fungicides early in the season.",
    'Apple___Cedar_apple_rust': "Remove nearby cedar trees if possible. Use resistant apple cultivars.",
    'Apple___healthy': "No action needed. Maintain good cultural practices.",
    'Background_without_leaves': "Image might not contain a leaf. Please retake the photo.",
    'Blueberry___healthy': "No action needed. Continue monitoring.",
    'Cherry___Powdery_mildew': "Use sulfur-based fungicides. Ensure proper air circulation.",
    'Cherry___healthy': "No action needed.",
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': "Use resistant hybrids. Rotate crops and apply fungicides.",
    'Corn___Common_rust': "Apply fungicides if severe. Use resistant hybrids.",
    'Corn___Northern_Leaf_Blight': "Apply fungicides and rotate crops.",
    'Corn___healthy': "Healthy. Maintain good cultural practices.",
    'Grape___Black_rot': "Remove mummified berries and infected shoots. Apply fungicides during bloom.",
    'Grape___Esca_(Black_Measles)': "Prune infected vines. Avoid water stress and wounds.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Apply copper-based fungicides. Prune to improve ventilation.",
    'Grape___healthy': "No action needed.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees. Control psyllid vector with insecticides.",
    'Peach___Bacterial_spot': "Use resistant varieties. Apply copper-based sprays.",
    'Peach___healthy': "No action needed.",
    'Pepper,_bell___Bacterial_spot': "Avoid overhead irrigation. Use copper sprays and resistant varieties.",
    'Pepper,_bell___healthy': "Healthy. No action needed.",
    'Potato___Early_blight': "Use fungicides like chlorothalonil. Practice crop rotation.",
    'Potato___Late_blight': "Remove infected plants. Apply systemic fungicides.",
    'Potato___healthy': "No action required.",
    'Raspberry___healthy': "No action needed.",
    'Soybean___healthy': "Healthy crop. Continue routine care.",
    'Squash___Powdery_mildew': "Apply sulfur or potassium bicarbonate. Increase air circulation.",
    'Strawberry___Leaf_scorch': "Use resistant cultivars. Remove infected leaves and use fungicides.",
    'Strawberry___healthy': "No action needed.",
    'Tomato___Bacterial_spot': "Use copper sprays. Avoid working when leaves are wet.",
    'Tomato___Early_blight': "Use crop rotation and fungicides like mancozeb.",
    'Tomato___Late_blight': "Apply fungicides and remove affected plants immediately.",
    'Tomato___Leaf_Mold': "Increase ventilation. Use fungicides like chlorothalonil.",
    'Tomato___Septoria_leaf_spot': "Remove lower infected leaves. Use fungicides.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use miticides or insecticidal soap. Keep humidity high.",
    'Tomato___Target_Spot': "Apply fungicides. Remove affected leaves.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Use virus-free seedlings. Control whitefly population.",
    'Tomato___Tomato_mosaic_virus': "Sanitize tools. Avoid tobacco near plants.",
    'Tomato___healthy': "Your tomato plant is healthy."
}


# -------------------- CUSTOM PATCH EXTRACTION LAYER --------------------
@tf.keras.utils.register_keras_serializable()
class PatchExtract(tf.keras.layers.Layer):
    def __init__(self, patch_size, stride, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [tf.shape(patches)[0], -1, patch_dims])
        return patches

    def get_config(self):
        config = super(PatchExtract, self).get_config()
        config.update({
            "patch_size": self.patch_size,
            "stride": self.stride,
        })
        return config

# Define custom objects dictionary
CUSTOM_OBJECTS = {'PatchExtract': PatchExtract}

# -------------------- CONFIGURATION --------------------
MODEL_PATH = "vit_model.keras"
SAVE_DIR = "collected_data"
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Update based on your ViT model input size

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_keras_model(model_path):
    """Cache the model so it's only loaded once."""
    try:
        model = tf.keras.models.load_model(
            model_path, custom_objects=CUSTOM_OBJECTS, compile=False
        )
        return model
    except Exception as e:
        st.error(f" Error loading model: {e}")
        return None

# Load the cached model
MODEL_PATH = "vit_model.keras"
model = load_keras_model(MODEL_PATH)

if model:
    st.success(" Model loaded successfully!")
else:
    st.stop()  # Stop execution if model fails to load

# -------------------- SIDEBAR UI TWEAKS --------------------
st.sidebar.markdown(
    "<style>"
    "div[data-testid='stSidebar'] {width: 300px !important; padding: 15px;}"
    "@media (max-width: 768px) {div[data-testid='stSidebar'] {width: 220px !important;}}"
    "</style>",
    unsafe_allow_html=True
)

# -------------------- IMAGE QUALITY CHECK FUNCTIONS --------------------
def is_blurry(image_pil, threshold=100.0):
    """Check if the image is blurry using Laplacian variance."""
    img_gray = np.array(image_pil.convert("L"))
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def is_poorly_lit(image_pil, low=40, high=220):
    """Check brightness of the image."""
    brightness = ImageStat.Stat(image_pil).mean[0]
    return brightness < low or brightness > high, brightness

def is_low_contrast(image_pil, threshold=30.0):
    """Check if the image has low contrast."""
    contrast_value = np.std(np.array(ImageEnhance.Contrast(image_pil).enhance(1.0)))
    return contrast_value < threshold, contrast_value

# -------------------- IMAGE PREPROCESSING --------------------
def preprocess_image(image_pil, height, width):
    """Resize, normalize, and prepare image for the model."""
    image_pil = image_pil.resize((width, height))
    img_array = np.array(image_pil)

    # Convert grayscale or RGBA images to RGB
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    return np.expand_dims(img_array.astype(np.float32) / 255.0, axis=0)

# -------------------- INTERACTIVITY: REAL-TIME UPDATES --------------------
st.sidebar.subheader(" Live Data Updates")

if st.sidebar.button(" Refresh App"):
    st.rerun()


# -------------------- IMAGE STORAGE & LOGGING --------------------
def save_image(image: Image.Image, predicted_class: str) -> str:
    """Save classified images with timestamp."""
    class_folder = os.path.join(SAVE_DIR, predicted_class)
    os.makedirs(class_folder, exist_ok=True)
    filename = f"{predicted_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(class_folder, filename)
    image.save(filepath)
    return filepath

def log_prediction(filename: str, predicted_class: str, confidence: float):
    """Log predictions to CSV file."""
    log_file = os.path.join(SAVE_DIR, "log.csv")
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['filename', 'predicted_class', 'confidence', 'timestamp'])
        writer.writerow([filename, predicted_class, f"{confidence:.2f}", datetime.now().isoformat()])

# -------------------- STREAMLIT APP --------------------
def main():
    
    # Select input method
    input_method = st.radio("Choose input method:", ("Upload Image", "Use Camera"))
    image = None

    if input_method == "Use Camera":
        camera_image = st.camera_input("📸 Take a photo")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")
    else:
        uploaded_file = st.file_uploader("📁 Upload a leaf image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    if image:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        blurry, blur_score = is_blurry(image)
        lighting_issue, brightness = is_poorly_lit(image)
        low_contrast, contrast = is_low_contrast(image)

        with col2:
            st.write("###  Prediction")
            if st.button("Classify Image", use_container_width=True):
                with st.spinner("Processing..."):
                    if blurry:
                        st.warning(f" Blurry image detected (Sharpness: {blur_score:.2f})")
                    if lighting_issue:
                        st.warning(f" Poor lighting (Brightness: {brightness:.2f})")
                    if low_contrast:
                        st.warning(f" Low contrast (Contrast: {contrast:.2f})")

                    if blurry or lighting_issue or low_contrast:
                        st.stop()

                    processed = preprocess_image(image, IMG_HEIGHT, IMG_WIDTH)
                    prediction = model.predict(processed)
                    class_index = np.argmax(prediction)
                    confidence = prediction[0][class_index]
                    predicted_class = CLASS_NAMES[class_index]

                    st.success(f" Predicted: **{predicted_class}**")
                    st.info(f" Confidence: **{confidence * 100:.2f}%**")

                    # Display mitigation strategy
                    strategy = MITIGATION_STRATEGIES.get(predicted_class, "No specific strategy found.")
                    st.write("### 🌱 Mitigation Strategy")
                    st.write(strategy)

                    saved_path = save_image(image, predicted_class)
                    log_prediction(saved_path, predicted_class, confidence)

        # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("This app uses a Vision Transformer (ViT) model to classify plant leaf conditions.")

    st.sidebar.write("### Project Notes")
    st.sidebar.write("""
    - **Technology Used**: Streamlit for the app, TensorFlow for the model, and OpenCV for image processing.
    - **Model Type**: Vision Transformer (ViT) trained on a dataset of plant leaf images.
    - **Key Features**:
      - Classify plant leaf conditions.
      - Suggest mitigation or treatment strategies.
      - Ensure good image quality (sharpness, lighting, contrast) for accurate predictions.
    """)

    st.write("")
    st.write("")
    

    # Adding brief notes about the project in the main body
    st.write("### Project Overview")
    st.write("""
    This project uses a Vision Transformer (ViT) model to classify plant leaf conditions into 
    various categories such as healthy leaves and diseased leaves. 
    The model is trained on a variety of plant species and their common diseases. 
    After classification, the app provides actionable mitigation strategies to help treat or manage the disease.
    """)

    st.write("")
    st.write("")
    

import base64

def render_footer():
    # Read and encode the image
    with open("my_photo.jpg", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    image_html = f"<img src='data:image/jpeg;base64,{encoded}' style='width:100px; height:100px; border-radius:50%; margin-right: 20px;'>"

    footer_html = f"""
    <hr style="border: 1px solid #ccc; margin-top: 40px;">
    <div style='display: flex; align-items: center; justify-content: center;'>
        {image_html}
        <div style='text-align: left; font-size: 14px; color: gray;'>
            <p><strong>Contact Address:</strong> Computer Science Dept. - UENR, Sunyani - Ghana</p>
            <p><strong>Email:</strong> <a href='mailto:yourname@example.com'>isaac.baiden.stu@uenr.edu.gh</a></p>
            <p><strong>Skills:</strong> Machine Learning | Deep Learning | Computer Vision | Streamlit | TensorFlow | Python</p>
            <p>© 2025 Crop Disease Classifier App</p>
        </div>
    </div>
    """

    st.markdown(footer_html, unsafe_allow_html=True)


# Add this call at the end of main()
if __name__ == "__main__":
    main()
    render_footer()
