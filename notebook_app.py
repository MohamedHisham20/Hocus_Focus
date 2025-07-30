# ============================================================================
# HOCUS FOCUS - FACE DETECTION AND CLASSIFICATION WEB APPLICATION
# ============================================================================
# This Flask application provides real-time face detection and classification
# using computer vision and machine learning techniques.

# ============================================================================
# IMPORTS
# ============================================================================
from flask import Flask, render_template, Response
import cv2
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import io
import time
from mtcnn import MTCNN
from keras.models import load_model

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================
app = Flask(__name__)

# Load pre-trained VGG16 model for face classification
VGG16_model = load_model('hocusfocusplease.h5')

# ============================================================================
# GLOBAL VARIABLES AND CONSTANTS
# ============================================================================
SIZE = 224  # Input image size for the neural network model
video_stream = cv2.VideoCapture(0)  # Initialize camera capture (device 0)

# Application state variables
label_html = 'Capturing...'  # Status label for the web interface
bbox = ''  # Bounding box coordinates for detected faces
count = 0  # Frame counter for processing every nth frame
i = 0  # Index for cycling through image filenames

# Image storage configuration
filenames = [
    '/static/image_1.jpg', 
    '/static/image_2.jpg', 
    '/static/image_3.jpg', 
    '/static/image_4.jpg', 
    '/static/image_5.jpg'
]

# Model predictions storage
predictions = []  # Store predictions from the neural network


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def crop_face_and_return(image):
    """
    Detect and crop faces from an input image using MTCNN face detector.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        numpy.ndarray or None: Cropped face image if face is detected, None otherwise
    """
    cropped_face = None
    
    # Initialize MTCNN face detector
    detector = MTCNN()
    
    # Detect faces in the image
    faces = detector.detect_faces(image)
    
    # If at least one face is detected, crop the first one
    if faces:
        # Extract bounding box coordinates
        x, y, width, height = faces[0]['box']
        
        # Crop the face from the original image
        cropped_face = image[y:y + height, x:x + width]
    
    return cropped_face


def gen_frames():
    """
    Generator function that captures video frames from the camera and processes them
    for face detection and classification. Yields JPEG-encoded frames for streaming.
    
    Yields:
        bytes: JPEG-encoded frame data with multipart headers for HTTP streaming
    """
    global bbox, count, predictions, i
    
    while True:
        # Capture frame from video stream
        success, frame = video_stream.read()
        if not success:
            break

        # Initialize data dictionary for frame processing results
        data = {'create': 0, 'show': 0, 'capture': 0, 'img': ''}
        
        # Process frames for face detection and classification
        if count < 5:
            # Process every 5th frame to reduce computational load
            if count % 5 == 0:
                # Save current frame to disk
                cv2.imwrite(filenames[i], frame)

                # Load and process the saved image
                image = cv2.imread(filenames[i])
                cropped_face = crop_face_and_return(image)
                
                # Convert from BGR to RGB color space
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

                # Validate that face was successfully cropped
                if cropped_face is not None and cropped_face.size != 0:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(cropped_face, 'RGB')
                    
                    # Resize image to match model input requirements
                    pil_image = pil_image.resize((SIZE, SIZE))
                    
                    # Convert back to numpy array
                    cropped_face = np.array(pil_image)
                    
                    # Reshape for model input (batch_size=1, height, width, channels)
                    image = tf.reshape(cropped_face, (1, SIZE, SIZE, 3))
                    
                    # Make prediction and store result
                    predictions.append(VGG16_model.predict(image))

                # Move to next filename index
                i += 1
            
            # Increment frame counter
            count += 1
        else:
            # Process accumulated predictions after collecting 5 samples
            # Convert predictions to class indices (argmax)
            predictions = [np.argmax(pr) for pr in predictions]
            
            # Reset counters for next batch
            count = 0
            i = 0

        # Encode frame as JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in multipart format for HTTP streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """
    Main route that serves the web application's home page.
    
    Returns:
        str: Rendered HTML template for the main interface
    """
    return render_template('index1.html')


@app.route('/video_feed')
def video_feed():
    """
    Route that provides the video stream for real-time face detection.
    Uses the gen_frames() generator to continuously stream video frames.
    
    Returns:
        Response: HTTP response with multipart video stream
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run the Flask application in debug mode when script is executed directly.
    Debug mode enables auto-reload on code changes and detailed error messages.
    """
    app.run(debug=True)
