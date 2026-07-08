#Import Libraries
import streamlit as st
import requests
from PIL import Image 
import os
import time

#just because running code in local and also in docker so it adjust both
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  
PREDICT_URL = f"{BASE_URL}/predict"


st.set_page_config(
    page_title="Steel Surface Defect Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)  


#sidebar 

with st.sidebar:

    st.title("About")

    st.write(
        """
        This application detects steel surface defects
        using a deep learning model.

        **Model Used**
        - MobileNetV2
        - Transfer Learning
        - TensorFlow/Keras

        **Classes**
        - Crazing
        - Inclusion
        - Patches
        - Pitted Surface
        - Rolled-in Scale
        - Scratches
        """
    )


#Main Title
st.title("🔍 Steel Surface Defect Detection") 

st.markdown(
    """
Upload a steel surface image and the trained AI model
will identify the defect category.

Supported image formats: JPG,JPEG,PNG
"""
) 

#Upload Image 

uploaded_file = st.file_uploader(
    "Choose a steel surface image",
    type=["jpg", "jpeg", "png"]
)
 
# Display Uploaded Image 

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([2, 3, 2])   #The numbers [2, 1, 2] are relative widths.

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            width=150
        )

#prediction coding start
# BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# PREDICT_URL = f"{BASE_URL}/predict"

# Add the "Predict" Button

if st.button("🔍 Predict Defect", use_container_width=True):
    with st.spinner("Predicting..."):
       # time.sleep(2)
        #st.write("Prediction Started...")

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
        PREDICT_URL,
        files=files
        ) 

        #output or replay 
        
        if response.status_code == 200:

            result = response.json()
            with col2:
                st.subheader("📋 Prediction Result :")
                st.divider()
                st.success(f"Predicted Defect: {result['predicted_class']}")
            with col3:
                st.subheader("Confidence :") 
                st.divider()  
                st.info(f"Confidence: {result['confidence']:.2%}")
                st.progress(result["confidence"]) #bar 
            
        else:
            with col2:
                try:
                    error_message = response.json()["detail"]
                except Exception:
                    error_message = "Unknown Error"

                    st.error(error_message)


st.divider()

st.caption(
    "Developed by chetan gode | Steel Surface Defect Detection | Version 1.0 "
    "Github - https://github.com/chetansgode/NEU_Surface_Defect_Detection_Project.git"
)