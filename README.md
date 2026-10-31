# Create a virtual environment  
- "py -m venv venv"

# Install required packages
- "pip install -r requirements.txt"

# Run the server 
- " uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

# To view the API endpoints in Swagger UI
- "http://127.0.0.1:8000/docs"

# To view the frontend for the face capture and skin type and issue detection
- "http://127.0.0.1:8000/face_capture"