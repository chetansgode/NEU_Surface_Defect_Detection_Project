import json
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2  
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



#-------------------------------------------------------
# Load Model 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "steel_defect_classifier.keras")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

#------------------------------------------------------

# Load Class Names 

CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

#------------------------------------------------------

# Create Prediction Function 

def predict_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #require rgb form 
    img = cv2.resize(img, (224, 224))          
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)             #change dim(1,224,224,3)  ie 1 added

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)     #max index
    predicted_class = class_names[predicted_index]     # class name give 
    confidence = np.max(prediction)              #% probabilty of output

    return predicted_class, confidence  



# to check above code is running or not 
if __name__ == "__main__":
    image_path = "C:/Users/Cheta/Desktop/steel_defect_project/data/NEU-DET/validation/images/scratches/scratches_241.jpg"   # Replace with your image

    predicted_class, confidence = predict_image(image_path)

    print("Predicted Class :", predicted_class)
    print("Confidence      :", round(confidence * 100, 2), "%")

