# 🎯 Hocus Focus - Advanced Drowsiness Detection System

**Hocus Focus** is an advanced real-time drowsiness detection system that uses deep learning and computer vision to monitor user engagement and alertness. The system employs Spatial Transformer Networks (STN) with PyTorch and traditional computer vision techniques to analyze facial features and detect drowsiness states in real-time.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Deployment Options](#deployment-options)
- [Model Architecture](#model-architecture)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)

## 🚀 Project Overview

The Hocus Focus system provides two main deployment options:

1. **Web Application** (`web_app.py`) - Full-featured web interface with real-time video streaming
2. **API Backend** (`mobile_API.py`) - RESTful API for mobile app integration

The system analyzes both eye and mouth states to determine user engagement levels, making it suitable for:
- Driver monitoring systems
- Student attention tracking
- Workplace safety monitoring
- Remote learning applications

## ✨ Features

- **Real-time drowsiness detection** using advanced deep learning models
- **Spatial Transformer Networks (STN)** for improved geometric invariance
- **Multi-modal analysis** - combines eye and mouth state detection
- **Web streaming interface** with live video feed
- **RESTful API** for mobile app integration
- **Audio alerts** for detected drowsiness
- **Engagement analytics** and reporting
- **Face detection** using both Haar Cascades and MTCNN

## 📁 Project Structure

```
Hocus_Focus/
├── 📁 Machine Learning Models
│   ├── hocusfocusplease.h5          # VGG16-based TensorFlow model
│   ├── Model_stn.pth                # Main PyTorch STN model
│   ├── Model2_stn.pth               # Eye detection STN model
│   └── Model_mouth_stn.pth          # Mouth detection STN model
│
├── 📁 Flask Applications
│   ├── web_app.py                       # Main web application
│   ├── mobile_API.py                    # API backend for mobile apps
│   └── notebook_app.py              # Alternative TensorFlow-based app
│
├── 📁 Training & Development
│   └── STN_Egyption_Data.ipynb      # Model training notebook
│
├── 📁 Computer Vision Resources
│   └── Haarcascades/
│       ├── haarcascade_frontalface_default.xml
│       ├── haarcascade_eye.xml
│       ├── haarcascade_car.xml
│       └── haarcascade_fullbody.xml
│
├── 📁 Web Interface
│   ├── templates/
│   │   ├── index.html               # Main interface template
│   │   ├── index0.html              # Base template
│   │   └── index1.html              # Alternative interface
│   └── static/
│       ├── css/main.css             # Styling
│       ├── js/main.js               # Frontend JavaScript
│       ├── images/                  # Logo and UI images
│       └── audio/                   # Alert sound effects
│
├── 📁 Configuration & Deployment
│   ├── requirments.txt              # Python dependencies
│   ├── vercel.json                  # Vercel deployment config
│   └── README.md                    # Project documentation
│
└── 📁 Documentation & Reports
    ├── final printing hocus focus poster.pdf
    ├── Final Repor Hocus Focus last version.pdf
    └── WhatsApp Image 2023-12-15 at 9.10.50 PM.jpeg
```

## 📄 File Descriptions

### Core Applications

- **`web_app.py`** - Main Flask web application with real-time video streaming
  - Features: Live drowsiness detection, web interface, engagement analytics
  - Models: Uses PyTorch STN models for eye detection
  - Use case: Full-featured web application for desktop/browser use

- **`mobile_API.py`** - API backend for mobile applications
  - Features: RESTful endpoints, dual-model analysis (eye + mouth)
  - Models: Uses both eye and mouth STN models
  - Use case: Backend API for mobile app integration

- **`notebook_app.py`** - Older implementation using TensorFlow
  - Features: MTCNN face detection, VGG16-based classification
  - Models: Uses TensorFlow/Keras H5 model
  - Use case: Alternative approach with different ML stack

### Machine Learning Models

- **`hocusfocusplease.h5`** - VGG16-based TensorFlow model for face classification
- **`Model_stn.pth`** - Main PyTorch STN model for drowsiness detection
- **`Model2_stn.pth`** - Specialized eye state detection model
- **`Model_mouth_stn.pth`** - Specialized mouth state detection model

### Training & Development

- **`STN_Egyption_Data.ipynb`** - Jupyter notebook containing:
  - Model architecture definitions (LeNet, STN, Drowness classes)
  - Training pipeline with custom dataset handling
  - Data preprocessing and augmentation
  - Model evaluation and visualization

## 📦 Requirements

### Python Dependencies

The project requires Python 3.8+ and the following essential packages for `web_app.py` and `mobile_API.py`:

```
# Core Flask Framework
Flask==2.3.2
Werkzeug==2.3.6

# PyTorch and Computer Vision
torch>=1.9.0
torchvision>=0.10.0
opencv-python==4.8.1.78
Pillow==10.1.0

# Scientific Computing
numpy==1.25.1

# Production Server
gunicorn

# Required Flask dependencies
click>=7.0
itsdangerous>=2.0
Jinja2>=3.0
MarkupSafe>=2.0
```

**Note:** The streamlined `requirments.txt` contains only the essential dependencies for `web_app.py` and `mobile_API.py`. 

For the TensorFlow-based `notebook_app.py`, you'll need additional packages:
```
tensorflow==2.15.0
keras==2.15.0
mtcnn==0.1.1
```

For development and model training (jupyter notebook), you'll also need:
```
matplotlib==3.8.2  # For visualization
jupyter  # For notebook development
```

See `requirments.txt` for the complete streamlined dependency list.

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MohamedHisham20/Hocus_Focus.git
cd Hocus_Focus
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirments.txt
```

### 4. Verify Model Files

Ensure the following model files are present:
- `hocusfocusplease.h5`
- `Model_stn.pth`
- `Model2_stn.pth`
- `Model_mouth_stn.pth`

## 🚀 Deployment Options

### Option 1: Web Application (Recommended for Desktop)

The main web application provides a complete interface with real-time video streaming.

```bash
# Run the main web application
python web_app.py
```

**Features:**
- Real-time video streaming at `http://localhost:5000/video`
- Web interface at `http://localhost:5000/`
- Engagement analytics at `http://localhost:5000/_stuff`
- Automatic drowsiness detection with visual feedback

**Configuration:**
- Camera: Uses default camera (device 0)
- Processing: Analyzes frames every 3 seconds
- Models: PyTorch STN model for eye detection

### Option 2: API Backend (Recommended for Mobile Apps)

The API backend provides RESTful endpoints for mobile app integration.

```bash
# Run the API backend
python mobile_API.py
```

**Endpoints:**
- **POST** `/video` - Upload image/video for analysis
- **GET** `/_stuff` - Get analytics data
- **GET** `/static/<filename>` - Serve static files

**Features:**
- Dual-model analysis (eyes + mouth)
- JSON response format
- Suitable for mobile app backends
- Runs on `http://localhost:8080`

### Option 3: Alternative TensorFlow Implementation

```bash
# Run the TensorFlow-based application
python notebook_app.py
```

**Features:**
- MTCNN face detection
- VGG16-based classification
- Alternative ML stack
- Runs on `http://localhost:5000`

## 🏛️ Model Architecture

### Spatial Transformer Network (STN) Architecture

The system uses a sophisticated architecture combining:

1. **ResNet18 Encoder** - Feature extraction from input images
2. **Spatial Transformer Network** - Geometric transformation learning
3. **LeNet Feature Extractor** - Final feature processing
4. **Fully Connected Classifier** - Binary classification (awake/drowsy)

```python
class Drowness(nn.Module):
    def __init__(self, num_classes):
        super(Drowness, self).__init__()
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-5])
        self.stn = STN()
        self.model = LeNet()
        self.fc = nn.Sequential(...)
    
    def forward(self, x):
        x_encoded = self.encoder(x)
        x_transformed = self.stn(x_encoded, x)
        features = self.model(x_transformed)
        prediction = self.fc(features)
        return prediction, x_transformed
```

### Model Performance

- **Input Size:** 224x224 RGB images
- **Output:** Binary classification (0=closed/drowsy, 1=open/awake)
- **Architecture:** ResNet18 + STN + LeNet hybrid
- **Training:** Custom Egyptian dataset with data augmentation

## 🔗 API Endpoints

### POST /video (nouran.py)

Upload image data for drowsiness analysis.

**Request:**
```json
{
  "image": "base64_encoded_image"
}
```

**Response:**
```json
{
  "state": 0,        // 0=active, 1=sleep(closed eyes), 2=yawn
}
```

### GET /_stuff

Get engagement analytics and system status.

**Response:**
```json
{
  "result": "Present",   // Present/Absent
  "engagement_score": 85,
  "session_duration": 1200,
  "alert_count": 3
}
```

## 💡 Usage Examples

### Basic Web Interface Usage

1. Start the web application:
   ```bash
   python web_app.py
   ```

2. Open browser to `http://localhost:5000`

3. Allow camera access when prompted

4. The system will automatically:
   - Detect faces in the video stream
   - Analyze eye states for drowsiness
   - Display real-time alerts
   - Play audio alerts when drowsiness is detected

### API Integration Example

```python
import requests
import base64

# Encode image
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8080/video",
    json={"image": image_data}
)

# Process response
result = response.json()
print(f"Drowsiness state: {result['eye_state']}")
```

### Training New Models

To train models on custom data:

1. Open `STN_Egyption_Data.ipynb`
2. Prepare your dataset in the required format
3. Modify data paths and parameters
4. Run training cells
5. Save the trained model weights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the `Final Repor Hocus Focus last version.pdf`
- Review the project poster in `final printing hocus focus poster.pdf`

## 📄 License

This project is part of an academic research initiative. Please cite appropriately if used in academic work.

---

**⚠️ Note:** Ensure you have a working camera/webcam for real-time detection features. The system is optimized for well-lit environments with clear facial visibility.
