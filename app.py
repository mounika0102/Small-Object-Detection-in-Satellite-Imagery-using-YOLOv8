import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import glob

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="YOLOv8 Image Prediction",
    page_icon="📷",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f8f9fa;
            color: #343a40;
        }
        .stApp {
            background: linear-gradient(to bottom right, #f8f9fa, #dfe6e9);
        }
        .header {
            color: #2c3e50;
            text-align: center;
            margin-top: 20px;
        }
        .prediction-box {
            border: 2px dashed #6c5ce7;
            border-radius: 15px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .footer {
            color: #636e72;
            font-size: 14px;
            text-align: center;
            margin-top: 20px;
        }
        .upload-section {
            border: 2px solid #6c5ce7;
            border-radius: 10px;
            padding: 10px;
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='header'>Small Object Detection using YOLOv8</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Upload and Predict")
model_path = st.sidebar.text_input("Model Path", "./runs/detect/weights/best.pt")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Ensure the "predictions" directory exists
os.makedirs("predictions", exist_ok=True)

# Main Section
if uploaded_file:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.markdown("<h3 class='header'>Uploaded Image</h3>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 model
    if st.button("Run Prediction"):
        with st.spinner("Running predictions..."):
            try:
                # Load YOLOv8 model
                if not os.path.exists(model_path):
                    st.error(f"Model file not found at: {model_path}")
                    st.stop()

                model = YOLO(model_path)
                
                # Save the uploaded file locally
                input_image_path = f"temp_image.{uploaded_file.name.split('.')[-1]}"
                image.save(input_image_path)

                # Perform inference and save the predictions
                results = model.predict(source=input_image_path, save=True)

                # Find the latest 'predict' folder
                prediction_folders = glob.glob('C:/Users/magul/runs/detect/predict*')
                latest_folder = max(prediction_folders, key=os.path.getmtime)

                # Get the path of the saved predicted image
                predicted_image_path = os.path.join(latest_folder, "temp_image.jpg")

                # Display predictions
                st.markdown("<h3 class='header'>Model Predictions</h3>", unsafe_allow_html=True)
                st.image(predicted_image_path, caption="Predicted Image", use_column_width=True)
                st.success("Prediction Complete! 🎉")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
            finally:
                # Cleanup the temporary input image
                if os.path.exists(input_image_path):
                    os.remove(input_image_path)

# Footer
st.markdown("<p class='footer'>Built with ❤️ using Streamlit and YOLOv8</p>", unsafe_allow_html=True)
