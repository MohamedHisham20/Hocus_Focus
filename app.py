# ============================================================================
# HOCUS FOCUS - REAL-TIME DROWSINESS DETECTION WITH VIDEO STREAMING
# ============================================================================
# This Flask application provides real-time drowsiness detection using PyTorch
# models with Spatial Transformer Networks (STN) and live video streaming.
# It continuously monitors user engagement through webcam feed analysis.

# ============================================================================
# IMPORTS
# ============================================================================
import time
from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
num_classes = 2  # Binary classification: 0 = closed/inactive, 1 = open/active

# Define image preprocessing transformation pipeline
# This prepares images for the neural network model
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to standard input size (224x224)
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])


# ============================================================================
# IMAGE PREPROCESSING UTILITIES
# ============================================================================

def load_image(image, transform):
    """
    Load and preprocess an image for neural network inference.
    
    Args:
        image (numpy.ndarray): Input image as numpy array (BGR format from OpenCV)
        transform (torchvision.transforms.Compose): Preprocessing transformation pipeline
        
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension (1, C, H, W)
    """
    # Convert numpy array to PIL Image for processing
    pil_image = Image.fromarray(image)
    
    # Ensure the image is in RGB format (required for model)
    pil_image = pil_image.convert("RGB")
    
    # Apply preprocessing transformations (resize, normalize, etc.)
    pil_image = transform(pil_image)
    
    # Add batch dimension (unsqueeze) - model expects (batch_size, channels, height, width)
    return pil_image.unsqueeze(0)


# ============================================================================
# NEURAL NETWORK MODEL DEFINITIONS
# ============================================================================

class LeNet(nn.Module):
    """
    Modified LeNet architecture for feature extraction.
    This serves as the feature extractor component in the drowsiness detection pipeline.
    """
    
    def __init__(self):
        super(LeNet, self).__init__()
        
        # Feature extraction layers with convolutional and pooling operations
        self.feature_extractor = nn.Sequential(
            # First convolutional block: 3 -> 6 channels
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # Second convolutional block: 6 -> 16 channels
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block: 16 -> 120 channels
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass through the feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Extracted features
        """
        features = self.feature_extractor(x)
        return features


class STN(nn.Module):
    """
    Spatial Transformer Network (STN) for geometric transformation learning.
    This network learns to apply spatial transformations to improve feature alignment
    and enhance the model's robustness to pose variations.
    """
    
    def __init__(self):
        super(STN, self).__init__()
        
        # Localization network to predict transformation parameters
        self.localization = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Fully connected layers to output 6 transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(90, 64),  # Flattened size from localization network
            nn.ReLU(True),
            nn.Linear(64, 6)  # 6 parameters for affine transformation matrix
        )
        
        # Initialize transformation parameters to identity matrix
        # This ensures the network starts with no transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x_encoder, x):
        """
        Apply spatial transformation to the input.
        
        Args:
            x_encoder (torch.Tensor): Input from encoder for transformation prediction
            x (torch.Tensor): Original input to be transformed
            
        Returns:
            torch.Tensor: Spatially transformed input
        """
        # Predict transformation parameters
        xs = self.localization(x_encoder)
        xs = xs.view(xs.size(0), -1)  # Flatten for FC layers
        theta = self.fc_loc(xs)
        
        # Reshape to 2x3 affine transformation matrix
        theta = theta.view(-1, 2, 3)
        
        # Generate sampling grid and apply transformation
        grid = F.affine_grid(theta, x_encoder.size(), align_corners=True)
        x_transformed = F.grid_sample(x_encoder, grid, align_corners=True)
        
        return x_transformed


class Drowness(nn.Module):
    """
    Main drowsiness detection model combining ResNet encoder, STN, and LeNet.
    This model analyzes facial features to detect drowsiness states.
    """
    
    def __init__(self, num_classes):
        super(Drowness, self).__init__()
        
        # ResNet18 encoder (pretrained) for initial feature extraction
        # Remove the last few layers to get intermediate features
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-5])
        
        # Channel reduction layer to convert ResNet features to 3 channels for STN
        self.channel_reshape = nn.Conv2d(64, 3, kernel_size=1)
        
        # LeNet feature extractor
        self.model = LeNet()
        
        # Spatial Transformer Network for geometric alignment
        self.stn = STN()
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(5880, num_classes)  # 5880 is the flattened feature size
        )

    def forward(self, x):
        """
        Forward pass through the complete drowsiness detection pipeline.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            tuple: (predictions, transformed_features)
                - predictions: Class predictions for drowsiness state
                - transformed_features: STN-transformed features for visualization
        """
        # Initial feature encoding with ResNet
        x_encoded = self.encoder(x)
        
        # Reduce channels for STN compatibility
        x_encoded = self.channel_reshape(x_encoded)
        
        # Apply spatial transformation
        x_transformed = self.stn(x_encoded, x)
        
        # Extract features using LeNet
        features = self.model(x_transformed)
        
        # Flatten features for classification
        features = torch.flatten(features, 1)
        
        # Final classification
        predictions = self.fc(features)
        
        return predictions, x_transformed


# ============================================================================
# MODEL INITIALIZATION AND DEVICE SETUP
# ============================================================================

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the drowsiness detection model
model = Drowness(num_classes)  # Create model instance for eye state detection

# Load pre-trained model weights
# This model has been trained specifically for drowsiness/eye state classification
model.load_state_dict(torch.load("Model2_stn.pth", map_location=torch.device('cpu')))

# Move model to the appropriate device (GPU/CPU)
model.to(device)

# Set model to evaluation mode (disables dropout, batch norm updates, etc.)
model.eval()


# ============================================================================
# PREDICTION AND INFERENCE FUNCTIONS
# ============================================================================

def predict(image_path):
    """
    Make predictions using the trained model on the given image.
    
    Args:
        image_path (numpy.ndarray): Input image as numpy array
        
    Returns:
        None: Prints prediction results and displays visualization
    """
    # Preprocess the image and move to device
    image = load_image(image_path, transform).to(device)
    
    # Forward pass through the model
    pred_state, stn_output = model(image)
    
    # Convert logits to probabilities using softmax
    pred_probs = torch.nn.functional.softmax(pred_state, dim=1).cpu().detach().numpy()
    
    # Get the predicted class (argmax of probabilities)
    predicted_state = np.argmax(pred_probs, axis=1)
    print(f"Predicted state: {predicted_state}")
    
    # Prepare visualization data (convert tensors to numpy for plotting)
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stn_np = stn_output[0].permute(1, 2, 0).cpu().detach().numpy()
    stn_np = np.clip(stn_np, 0, 1)  # Clip values to valid range for display

    # Display original and transformed images side by side
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(stn_np)
    plt.title('STN Transformed')

    plt.show()

    print(f'Detected state: {predicted_state}')


# ============================================================================
# FLASK APPLICATION SETUP
# ============================================================================

# Initialize Flask application
app = Flask(__name__)

# Initialize camera capture (device 0 = default camera)
camera = cv2.VideoCapture(0)

# ============================================================================
# COMPUTER VISION HELPER FUNCTIONS
# ============================================================================

def crop_face_and_return(image):
    """
    Detect and crop faces from an input image using Haar Cascade classifier.
    
    Args:
        image (numpy.ndarray): Input image in RGB format
        
    Returns:
        numpy.ndarray or None: Cropped face image if face is detected, None otherwise
    """
    cropped_face = None
    
    # Initialize Haar Cascade face detector
    detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    # Parameters: scaleFactor=1.1, minNeighbors=7
    faces = detector.detectMultiScale(image, 1.1, 7)
    
    # Extract the first detected face
    for (x, y, w, h) in faces:
        cropped_face = image[y:y + h, x:x + w]
        break  # Only take the first face
        
    return cropped_face

# Alternative MTCNN implementation (commented out for reference)
# def crop_face_and_return(image):
#    """Alternative face detection using MTCNN (more accurate but slower)"""
#    cropped_face = None
#    detector = MTCNN()
#    faces = detector.detect_faces(image)
#    if faces:
#         x, y, width, height = faces[0]['box']
#         cropped_face = image[y:y + height, x:x + width]
#    return cropped_face


# Function to check if eyes are closed based on aspect ratio
#takes array of eyes that are detected
# def are_eyes_closed(eyes):
#     awake = 0
#     for eye in eyes:
#         #get the exact ratio of the eye
#         (x, y, w, h) = eye
#         aspect_ratio = float(w) / h  # the greater the value the more sleepy
#         # Set a threshold for the aspect ratio to determine closed eyes
#         closed_threshold = 5.0  # may be modified
#         if aspect_ratio < closed_threshold:
#             awake += 1 #an eye is detected as open
#     if awake > 0:
#         return False
#     else:
#         return True

# ============================================================================
# GLOBAL APPLICATION STATE VARIABLES
# ============================================================================

# Prediction storage for calculating engagement averages
prediction = []  # Stores prediction history for statistical analysis

# ============================================================================
# VIDEO STREAMING AND PROCESSING FUNCTIONS
# ============================================================================

#main function of the video and prediction
def generate_frames():
    """
    Generator function that captures video frames from the camera and processes them
    for real-time drowsiness detection. Yields JPEG-encoded frames for streaming.
    
    This function:
    1. Captures frames from the webcam
    2. Processes frames every 3 seconds to reduce computational load
    3. Detects faces and analyzes drowsiness state
    4. Yields processed frames for web streaming
    
    Yields:
        bytes: JPEG-encoded frame data with multipart headers for HTTP streaming
    """
    timey = 0       # Timer for controlling processing intervals
    last_pred = 0   # Previous prediction state (for consistency checking)
    
    while True:
        # Capture frame from camera
        success, frame_bgr = camera.read()
        
        # Convert BGR to RGB for processing
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Flip frame horizontally for mirror effect (better user experience)
        frame = cv2.flip(frame, 1)
        
        # Process frame every 3 seconds to reduce computational overhead
        if time.time() - timey > 3:
            timey = time.time()  # Update timer
            
            if not success:  # Check if frame capture failed
                break
            else:
                # Face detection and analysis
                detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
                faces = detector.detectMultiScale(frame, 1.1, 7)

                # Draw rectangles around detected faces for visualization
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Extract and analyze the detected face
                cropped_face = crop_face_and_return(frame)
                
                if cropped_face is not None:  # Face successfully detected
                    print(f"Cropped face shape: {cropped_face.shape}")
                    
                    # Analyze drowsiness state using the trained model
                    predict(cropped_face)
                    
                    # Note: Prediction logic for storing results is commented out
                    # This would normally update the prediction array for analytics
                    # Uncomment and modify as needed for your specific requirements

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format for HTTP streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


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
    return render_template('index.html')


# ============================================================================
# ANALYTICS AND REPORTING VARIABLES
# ============================================================================

# Global variables for engagement analytics calculation
summ = 0    # Sum counter for engagement calculation
timeyy = 0  # Timestamp for periodic calculations


@app.route('/_stuff', methods=['GET'])
def stuff():
    """
    Calculate and return engagement analytics based on prediction history.
    
    This endpoint provides real-time engagement statistics by analyzing
    the accumulated prediction data over time intervals.
    
    Returns:
        JSON response containing:
        - Engagement percentage (when 5 predictions are collected)
        - Current state description ('Engaged', 'Absent', 'Disengaged')
    """
    global summ, timeyy
    
    message = ''
    
    # Process predictions only if we have data
    if len(prediction):
        # Update every 1 second (time-based processing)
        while time.time() - timeyy > 1:
            timeyy = time.time()
            
            # Calculate average engagement every 5 predictions
            if len(prediction) % 5 == 0:
                # Cap the sum at 5 for percentage calculation
                if summ > 5:
                    summ = 5
                    
                # Calculate engagement percentage
                avg = (summ / 5) * 100
                message = f'Engagement: {round(avg, 2)}%'
                
                # Reset counter for next calculation cycle
                summ = 0
            else:
                # Provide real-time state feedback
                latest_prediction = prediction[-1]  # Get most recent prediction
                
                if latest_prediction == 0:
                    message = 'Engaged'
                    summ += 1  # Count as engaged
                elif latest_prediction == -1:
                    message = "Absent"
                else:  # latest_prediction == 1 or 2
                    message = "Disengaged"
    
    return jsonify(result=message)


@app.route('/video')
def video():
    """
    Route that provides the video stream for real-time drowsiness detection.
    Uses the generate_frames() generator to continuously stream video frames.
    
    Returns:
        Response: HTTP response with multipart video stream
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================================
# STATIC FILE SERVING
# ============================================================================

@app.route('/static/<path:filename>')
def static_files(filename):
    """
    Serve static files (CSS, JS, images) for the web interface.
    
    Args:
        filename (str): Requested static file path
        
    Returns:
        File response from the static directory
    """
    return send_from_directory('static', filename)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Start the Flask application server.
    
    Configuration:
    - Debug: True (enables auto-reload and detailed error messages)
    - Default host: 127.0.0.1 (localhost)
    - Default port: 5000
    """
    app.run(debug=True)
