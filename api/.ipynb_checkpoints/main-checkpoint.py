#Step 1 Import Libraries
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import sys
import os 
import shutil
from predictor import predict_image 

# Fix path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)  

#Step 2 Create FastAPI App 

#load model
model = tf.keras.models.load_model(
    "steel_defect_classifier.keras",
    custom_objects={
        "preprocess_input": preprocess_input
    }
) 


#create Fastapi object
app=FastAPI()
Model_version=1.1 

#Create Endpoint 
@app.get("/")
def hello():
    return{'Message':'This API show type of defect on steel surface'}


#endpoint
@app.get('/health')               
def health_check() -> dict :                #for machine visualisation
    return {
            'status':'ok',
            'version':Model_version,

            'model_loaded':model is not None}


# Create Upload Folder
#If folder doesn't exist, create it.

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
