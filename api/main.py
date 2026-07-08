#Step 1 Import Libraries
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import sys
import os 
import shutil
from api.schema import PredictionResponse
from api.predictor import predict_image ,model

# Fix path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)  

#Step 2 Create FastAPI App 




#create Fastapi object
app=FastAPI(
    title="Steel Surface Defect Detection API",
    version="1.1",
    description="Predicts steel surface defects using a MobileNetV2 deep learning model."
)
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

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    
}


# Prediction route
@app.post("/predict",response_model=PredictionResponse)

async def predict(file:UploadFile = File(...)):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail='Only JPG,JPEG and PNG images are allowed'
        )

    # save uploaded image 
    file_path = os.path.join(UPLOAD_FOLDER,file.filename) 

    try:

        with open(file_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer) 

        #predict 
        predicted_class,confidence =predict_image(file_path) 

        #return result 

        return PredictionResponse(predicted_class=predicted_class,
                                confidence=float(confidence))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'prediction failed : {str(e)}'
        )
    
    #to clean upload folder
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)