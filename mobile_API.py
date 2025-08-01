# ============================================================================
# HOCUS FOCUS - ADVANCED DROWSINESS DETECTION WITH SPATIAL TRANSFORMER NETWORKS
# ============================================================================
# This Flask application provides real-time drowsiness detection using PyTorch
# models with Spatial Transformer Networks (STN) for enhanced accuracy.
# It analyzes both eye and mouth states to determine engagement levels.
# It is used as backend for the mobile app

# ============================================================================
# IMPORTS
# ============================================================================
import json
import time
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import cv2
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# FLASK APPLICATION SETUP
# ============================================================================
app = Flask(__name__)

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
num_classes = 2  # Binary classification: 0 = closed/inactive, 1 = open/active

# Define image preprocessing transformation pipeline
# This prepares images for the neural network models
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
        self.encoder = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-5])
        
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

# Initialize separate models for eye and mouth state detection
eye_model = Drowness(num_classes)    # Model for detecting eye states (open/closed)
mouth_model = Drowness(num_classes)  # Model for detecting mouth states (open/closed)

# Load pre-trained model weights
# These models have been trained specifically for eye and mouth state classification
eye_model.load_state_dict(
    torch.load("Model2_stn.pth", map_location=torch.device('cpu'), weights_only=True)
)
mouth_model.load_state_dict(
    torch.load("Model_mouth_stn.pth", map_location=torch.device('cpu'), weights_only=True)
)

# Move models to the appropriate device (GPU/CPU)
eye_model.to(device)
mouth_model.to(device)

# Set models to evaluation mode (disables dropout, batch norm updates, etc.)
eye_model.eval()
mouth_model.eval()


# ============================================================================
# PREDICTION AND INFERENCE FUNCTIONS
# ============================================================================

def predict(passed_model, image_path):
    """
    Make predictions using the specified model on the given image.
    
    Args:
        passed_model (torch.nn.Module): The neural network model (eye_model or mouth_model)
        image_path (numpy.ndarray): Input image as numpy array
        
    Returns:
        dict: Dictionary containing the predicted state (0 or 1)
              0 = closed/inactive, 1 = open/active
    """
    try:
        # Preprocess the image and move to device
        image = load_image(image_path, transform).to(device)
        
        # Forward pass through the model
        with torch.no_grad():  # Disable gradient computation for inference
            pred_state, stn_output = passed_model(image)
        
        # Convert logits to probabilities using softmax
        pred_probs = torch.nn.functional.softmax(pred_state, dim=1).cpu().detach().numpy()
        
        # Get the predicted class (argmax of probabilities)
        predicted_state = np.argmax(pred_probs, axis=1)
        print(f"Predicted state: {predicted_state}")
        
        # Convert numpy array to integer for JSON serialization
        predicted_state = int(predicted_state[0])
        
        print(f"Prediction result: {{'state': {predicted_state}}}")
        return {'state': predicted_state}
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {'state': 0}  # Default to closed/inactive state on error


# ============================================================================
# COMPUTER VISION HELPER FUNCTIONS
# ============================================================================

def crop_face_and_return(image):
    """
    Detect and crop faces from an input image using Haar Cascade classifier.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
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


# def crop_face_and_return(image):
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


prediction = []  # prediction array used to calculate the average

# Global variables for analytics
summ = 0    # Sum counter for engagement calculation
timeyy = 0  # Timestamp for periodic calculations
pred = -1   # Current prediction state
last_pred = -1  # Previous prediction state


# main application
@app.route('/')
def index():
    """
    Simple root route for basic application health check.
    
    Returns:
        str: Welcome message
    """
    return "Hocus Focus API - Drowsiness Detection Backend"


@app.route('/video', methods=['POST'])
def generate_frames():
    """
    Main video processing endpoint that analyzes uploaded images for drowsiness detection.
    
    This function:
    1. Receives an image via POST request
    2. Detects faces in the image
    3. Analyzes eye and mouth states using neural networks
    4. Determines overall engagement state
    5. Updates prediction history
    
    Returns:
        JSON response with current state:
        - 0: Active/Engaged
        - 1: Sleepy/Drowsy (closed eyes)
        - 2: Yawning (open eyes, open mouth)
        - -1: Absent (no face detected)
    """
    global pred, last_pred, prediction  # Use global variables for state persistence

    # # Initialize persistent variables on first request
    # if not hasattr(generate_frames, 'last_pred'):
    #     last_pred = -1  # Previous prediction state
    # if not hasattr(generate_frames, 'pred'):
    #     pred = -1  # Current prediction state

    # Validate image upload
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image found"}), 400
    
    # Process uploaded image
    try:
        # Convert uploaded file to numpy array
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Convert BGR to RGB for processing
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Face detection using Haar Cascade
        detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(frame, 1.1, 7)

        # Draw rectangles around detected faces (for debugging/visualization)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Extract and analyze the face
        cropped_face = crop_face_and_return(frame)
        
        if cropped_face is not None:  # Face successfully detected
            # Analyze eye state (0=closed, 1=open)
            eye_state_result = predict(eye_model, cropped_face)
            eye_state = eye_state_result['state']
            
            # Analyze mouth state (0=closed, 1=open)
            mouth_state_result = predict(mouth_model, cropped_face)
            mouth_state = mouth_state_result['state']

            # Determine overall engagement state based on eye and mouth analysis
            if eye_state == 0:  # Eyes closed
                pred = 1  # Sleepy/Drowsy state
            elif eye_state == 1:  # Eyes open
                if mouth_state == 0:  # Mouth closed
                    pred = 0  # Active/Engaged
                else:  # Mouth open
                    pred = 2  # Yawning state
        else:  # No face detected
            pred = -1  # Absent state

        # Update prediction history for statistical analysis
        # Only update history for significant state changes or consistent states
        if pred == 1 or pred == -1 or pred == 2:  # Sleep/Absent/Yawn states
            if pred == last_pred:  # Consistent state detection
                prediction.append(pred)
        else:  # Active state
            prediction.append(pred)

        # Update state tracking
        last_pred = pred
        
        # Debug logging
        print(f'Last prediction: {last_pred}')
        print(f'Current prediction: {pred}')
        print(f'Prediction history length: {len(prediction)}')
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Image processing failed"}), 500

    return jsonify({"state": pred})


# ============================================================================
# ANALYTICS AND REPORTING ROUTES
# ============================================================================

@app.route('/_stuff', methods=['GET'])
def stuff():
    """
    Calculate and return engagement analytics based on prediction history.
    
    This endpoint provides real-time engagement statistics by analyzing
    the accumulated prediction data over time intervals.
    
    Returns:
        JSON response containing:
        - Current state description and engagement statistics
    """
    global summ, timeyy, pred, prediction
    
    message = 'No Data Available'
    
    # Process predictions only if we have data
    if len(prediction) > 0:
        # Get the most recent prediction
        latest_prediction = prediction[-1]
        
        # Provide real-time state feedback
        if latest_prediction == 0:
            current_state = 'Engaged'
        elif latest_prediction == 1:
            current_state = 'Drowsy'
        elif latest_prediction == 2:
            current_state = 'Yawning'
        elif latest_prediction == -1:
            current_state = 'Absent'
        else:
            current_state = 'Unknown'
            
        # Calculate engagement statistics over the last 10 predictions
        if len(prediction) >= 10:
            recent_predictions = prediction[-10:]
            engaged_count = sum(1 for p in recent_predictions if p == 0)
            engagement_percentage = (engaged_count / 10) * 100
            message = f'Engagement: {engagement_percentage:.0f}% - Currently {current_state}'
        else:
            message = f'Currently {current_state} - Collecting data ({len(prediction)}/10)'
    
    return jsonify(result=message)


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
    - Host: 0.0.0.0 (accessible from any network interface)
    - Port: 8080
    - Debug: True (enables auto-reload and detailed error messages)
    """
    app.run(host='0.0.0.0', port=8080, debug=True)
