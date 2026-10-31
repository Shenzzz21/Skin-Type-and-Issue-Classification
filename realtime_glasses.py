import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load TFLite model (predicts whether a detected face is wearing glasses)
interpreter = tf.lite.Interpreter(model_path="models/model-full.tflite")
interpreter.allocate_tensors()
# Get model input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Input image size expected by the model
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Face Preprocessing Function
def preprocess_face(face_img):
    """
    Preprocess the detected face region:
    - Convert BGR (OpenCV) to RGB
    - Resize to model input size
    - Normalize using ResNet50 preprocessing
    """
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
    face_array = np.expand_dims(face_img, axis=0)
    face_array = preprocess_input(face_array)
    return face_array.astype(np.float32)

# Initialize Webcam Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Real-time Glasses Detection Loop
while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_input = preprocess_face(face_roi)

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], face_input)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]

        # Check shape
        if pred.shape[0] == 2:
            glasses_prob = pred[1]  # Two-class output: [no_glasses, glasses]
        else:
            glasses_prob = pred[0]  # Single probability output (glasses confidence)

        # Determine label based on probability threshold
        label = "No Glasses" if glasses_prob > 0.5 else "Glasses"
        color = (0, 0, 255) if label == "Glasses" else (0, 255, 0)
        # Draw bounding box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Display label and confidence score
        cv2.putText(frame, f"{label} ({glasses_prob:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show real-time output window
    cv2.imshow("Real-time Glasses Detection (.tflite)", frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
