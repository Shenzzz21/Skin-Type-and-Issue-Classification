from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import io
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

import os
import cv2
from datetime import datetime

# Configurations
IMG_HEIGHT, IMG_WIDTH = 224, 224
CONFIDENCE_THRESHOLD = 0.01
TYPE_THRESHOLD = 0.65
TYPE_MARGIN = 0.15
EXCLUDED_ISSUE_CLASS = "Vitiligo"

# Class Labels
ISSUE_CLASS_LABELS = [
    'Acne', 'Blackheads', 'Dark circles (Puffy Eyes)', 'Hyperpigmentation',
    'Redness', 'Scars', 'Vitiligo', 'Whiteheads', 'Wrinkles'
]
TYPE_CLASS_LABELS = ['Dry', 'Oily']

# Load Models Once at Startup 
try:
    skin_type_model = load_model("models/skin_type_classifier.h5")
    skin_issue_model = load_model("models/tuned_finetuned_model.h5")
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")

# Initialize App 
app = FastAPI(title="Skin Analyzer & Face Capture API")

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates 
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load OpenCV cascades once 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Load TFLite glasses detection model once at startup 
glasses_interpreter = tf.lite.Interpreter(model_path="models/model-full.tflite")
glasses_interpreter.allocate_tensors()
glasses_input_details = glasses_interpreter.get_input_details()
glasses_output_details = glasses_interpreter.get_output_details()

def preprocess_face(face_img):
    """
    Preprocess face ROI for glasses detection model
    """
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
    face_array = np.expand_dims(face_img, axis=0)
    face_array = preprocess_input(face_array)
    return face_array.astype(np.float32)

def preprocess_uploaded_image(img_path):
    """
    Preprocess uploaded image from disk for CNN models
    """
    img = keras_image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras_image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def preprocess_image_from_array(img_array):
    """
    Preprocess image already loaded as NumPy array
    """
    img = Image.fromarray(img_array)
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = keras_image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def base64_to_opencv_image(base64_string):
    """
    Convert base64 encoded image to OpenCV BGR image
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_data = base64.b64decode(base64_string)
    img_pil = Image.open(io.BytesIO(img_data))
    img_array = np.array(img_pil)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array

# Face validation logic
def validate_face_image(img_array):
    """
    Validate:
    - Glasses presence
    - Face centering & orientation
    - Distance & lighting conditions
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    height, width = img_array.shape[:2]
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    
    if len(faces) == 0:
        return {
            "glasses": "no_face",
            "face_center": "no_face", 
            "distance": "no_face"
        }
    
    # Use the largest detected face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    
    # GLASSES DETECTION (TFLite model)
    try:
        face_roi = img_array[y:y+h, x:x+w]
        face_input = preprocess_face(face_roi)

        glasses_interpreter.set_tensor(glasses_input_details[0]['index'], face_input)
        glasses_interpreter.invoke()
        pred = glasses_interpreter.get_tensor(glasses_output_details[0]['index'])

        # Ensure pred is 1D
        pred = np.array(pred).flatten()

        if len(pred) == 2:
            glasses_prob = float(pred[1])  # two-class output: 1 = glasses
        else:
            glasses_prob = float(pred[0])  

        glasses_status = "ok" if glasses_prob > 0.5 else "not_ok"

    except Exception as e:
        # Fallback if TFLite model fails
        print(f"Glasses detection error: {e}")
        glasses_status = "not_ok"

    
    # FACE CENTERING CHECK
    # Check if face is centered (within middle 60% of image)
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    center_zone_x = width * 0.2, width * 0.8
    center_zone_y = height * 0.2, height * 0.8
    
    face_centered = (center_zone_x[0] <= face_center_x <= center_zone_x[1] and
                    center_zone_y[0] <= face_center_y <= center_zone_y[1])
    
    # FACING FORWARD CHECK
    # Check if facing forward using eyes
    face_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    facing_forward = False
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        eye1, eye2 = eyes[0], eyes[1]
        y_diff = abs(eye1[1] - eye2[1])
        if y_diff < h * 0.15:
            facing_forward = True
    
    face_center_status = "ok" if face_centered and facing_forward else "not_ok"
    
    # Check distance/size (face should be 15-40% of image width)
    face_size_ratio = w / width
    distance_ok = 0.15 <= face_size_ratio <= 0.4
    distance_status = "ok" if distance_ok else "not_ok"
    
    # Check lighting    
    face_roi = img_array[y:y+h, x:x+w]
    face_brightness = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
    lighting_ok = 100 <= face_brightness <= 160
    
    # Override distance status with lighting check
    if not lighting_ok:
        distance_status = "not_ok"
    
    return {
        "glasses": glasses_status,
        "face_center": face_center_status,
        "distance": distance_status
    }

# API Routes 

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Skin Analyzer API is running!"}

@app.get("/face_capture")
async def face_capture_page(request: Request):
    """Render face capture UI"""
    return templates.TemplateResponse("face_capture.html", {"request": request})

# API 1: Validation API
@app.post("/validate_face")
async def validate_face(image_file: UploadFile = File(...)):
    """Validate face image for glasses, centering, and distance"""
    try:
        # Read image
        image_data = await image_file.read()
        img_pil = Image.open(io.BytesIO(image_data))
        img_array = np.array(img_pil)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Validate
        validation_result = validate_face_image(img_array)
        
        return JSONResponse(content=validation_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

# Alternative validation API that accepts base64
@app.post("/validate_face_base64")
async def validate_face_base64(data: dict):
    """Validate face image from base64 data"""
    try:
        base64_image = data.get("image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Convert base64 to OpenCV image
        img_array = base64_to_opencv_image(base64_image)
        
        # Validate
        validation_result = validate_face_image(img_array)
        
        return JSONResponse(content=validation_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

# API 2: Final Image Processing API
@app.post("/process_final_image")
async def process_final_image(image_file: UploadFile = File(...)):
    """Process the final captured image for skin analysis"""
    try:
        # Save the image temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_{timestamp}.jpg"
        os.makedirs("static/captured", exist_ok=True)
        filepath = os.path.join("static/captured", filename)
        
        # Save uploaded file
        with open(filepath, "wb") as buffer:
            image_data = await image_file.read()
            buffer.write(image_data)
        
        # Preprocess for models
        img_array = preprocess_uploaded_image(filepath)
        
        # Skin type prediction
        preds_type = skin_type_model.predict(img_array)[0]
        dry_conf, oily_conf = preds_type
        
        if max(dry_conf, oily_conf) < TYPE_THRESHOLD or abs(dry_conf - oily_conf) < TYPE_MARGIN:
            skin_type_result = "Normal"
        else:
            skin_type_result = TYPE_CLASS_LABELS[np.argmax(preds_type)]
        
        # Skin issues prediction
        preds_issues = skin_issue_model.predict(img_array)[0]
        skin_issues = [
            ISSUE_CLASS_LABELS[i]
            for i, prob in enumerate(preds_issues)
            if prob >= CONFIDENCE_THRESHOLD and ISSUE_CLASS_LABELS[i] != EXCLUDED_ISSUE_CLASS
        ]
        
        return JSONResponse(content={
            "success": True,
            "image_path": f"/static/captured/{filename}",
            "skin_type": skin_type_result,
            "skin_issues": skin_issues,
            "processing_id": timestamp
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Alternative API for base64 image processing
@app.post("/process_final_image_base64")
async def process_final_image_base64(data: dict):
    """Process final image from base64 data"""
    try:
        base64_image = data.get("image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Convert and save image
        img_array = base64_to_opencv_image(base64_image)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_{timestamp}.jpg"
        os.makedirs("static/captured", exist_ok=True)
        filepath = os.path.join("static/captured", filename)
        
        # Save image
        cv2.imwrite(filepath, img_array)
        
        # Preprocess for models
        img_array_processed = preprocess_uploaded_image(filepath)
        
        # Skin type prediction
        preds_type = skin_type_model.predict(img_array_processed)[0]
        dry_conf, oily_conf = preds_type
        
        if max(dry_conf, oily_conf) < TYPE_THRESHOLD or abs(dry_conf - oily_conf) < TYPE_MARGIN:
            skin_type_result = "Normal"
        else:
            skin_type_result = TYPE_CLASS_LABELS[np.argmax(preds_type)]
        
        # Skin issues prediction
        preds_issues = skin_issue_model.predict(img_array_processed)[0]
        skin_issues = [
            ISSUE_CLASS_LABELS[i]
            for i, prob in enumerate(preds_issues)
            if prob >= CONFIDENCE_THRESHOLD and ISSUE_CLASS_LABELS[i] != EXCLUDED_ISSUE_CLASS
        ]
        
        return JSONResponse(content={
            "success": True,
            "image_path": f"/static/captured/{filename}",
            "skin_type": skin_type_result,
            "skin_issues": skin_issues,
            "processing_id": timestamp
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# API 3: Get Results API
@app.get("/get_results/{processing_id}")
async def get_results(processing_id: str):
    """Get processing results by ID"""
    try:
        return JSONResponse(content={
            "processing_id": processing_id,
            "status": "completed",
            "message": "Results retrieved successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

@app.get("/results")
async def results_page(request: Request, image_path: str, skin_type: str, skin_issues: str):
    """Results page"""
    # Parse skin_issues from string to list
    issues_list = skin_issues.split(",") if skin_issues else []
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "image_url": image_path,
        "skin_type": skin_type,
        "skin_issues": issues_list
    })

# Legacy APIs for compatibility
@app.post("/predict_skin_type")
async def predict_skin_type(image_file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{image_file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await image_file.read()
            buffer.write(content)

        img_array = preprocess_uploaded_image(temp_path)
        preds = skin_type_model.predict(img_array)[0]
        dry_conf, oily_conf = preds

        if max(dry_conf, oily_conf) < TYPE_THRESHOLD or abs(dry_conf - oily_conf) < TYPE_MARGIN:
            result = "Normal"
        else:
            result = TYPE_CLASS_LABELS[np.argmax(preds)]

        os.remove(temp_path)
        return JSONResponse(content={"skin_type": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_skin_issues")
async def predict_skin_issues(image_file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{image_file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await image_file.read()
            buffer.write(content)

        img_array = preprocess_uploaded_image(temp_path)
        preds = skin_issue_model.predict(img_array)[0]

        issues = [
            ISSUE_CLASS_LABELS[i]
            for i, prob in enumerate(preds)
            if prob >= CONFIDENCE_THRESHOLD and ISSUE_CLASS_LABELS[i] != EXCLUDED_ISSUE_CLASS
        ]

        os.remove(temp_path)
        return JSONResponse(content={"skin_issues": issues})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))