# 🌪️ Disaster Classifier - Deep Learning for Natural Disaster Recognition

A robust deep learning-based image classification system that identifies and classifies natural disasters from images into four categories: **Cyclone**, **Earthquake**, **Flood**, and **Wildfire**. Built with TensorFlow/Keras using ResNet50 architecture and deployed with Flask.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training Details](#training-details)
- [API Documentation](#api-documentation)
- [Results & Visualizations](#results--visualizations)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a state-of-the-art CNN-based disaster classification system that can automatically identify different types of natural disasters from images. The system achieves **87.75% accuracy** on the test set with an **AUC of 0.96**, making it suitable for real-world applications in emergency response, disaster management, and automated monitoring systems.

### Key Highlights
- 🎯 **87.75% Test Accuracy** with 96% AUC
- 🚀 **ResNet50** pre-trained on ImageNet for transfer learning
- 📊 **Advanced data augmentation** to prevent overfitting
- 🔄 **Two-phase training** strategy (frozen base → fine-tuning)
- 🌐 **Flask-based web deployment** for easy inference
- 📈 **Comprehensive monitoring** with TensorBoard integration

---

## ✨ Features

### Model Features
- **Transfer Learning**: Leverages ResNet50 pre-trained on ImageNet
- **Data Augmentation**: Rotation, shifts, zoom, brightness adjustments, and flips
- **Overfitting Detection**: Custom callback to monitor and prevent overfitting
- **Two-Phase Training**: 
  - Phase 1: Train classifier head with frozen base model
  - Phase 2: Fine-tune top layers for improved performance
- **Multi-Metric Evaluation**: Accuracy, AUC, Precision, Recall, F1-Score
- **Class Balance Analysis**: Ensures fair representation across disaster types

### Deployment Features
- **REST API**: Easy-to-use Flask-based inference endpoint
- **Web Interface**: User-friendly HTML interface for image upload
- **Real-time Predictions**: Fast inference with confidence scores
- **Cross-platform**: Works on Linux, Windows, and macOS

---

## 🏗️ Model Architecture

### Architecture Details
```
Input (224x224x3)
    ↓
ResNet50 Base Model (ImageNet weights, frozen initially)
    ↓
Global Average Pooling 2D
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense (512 units, ReLU activation)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense (256 units, ReLU activation)
    ↓
Dropout (0.2)
    ↓
Output Dense (4 units, Softmax activation)
```

### Technical Specifications
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Input Shape**: 224 × 224 × 3 (RGB images)
- **Output Classes**: 4 (cyclone, earthquake, flood, wildfire)
- **Preprocessing**: ResNet-specific normalization ([-1, 1] range)
- **Total Parameters**: ~24M (trainable: ~3.5M after phase 1)
- **Framework**: TensorFlow 2.x / Keras

---

## 📊 Dataset

### Dataset Structure
```
Dataset/
├── train/          # Training set (70%)
│   ├── cyclone/
│   ├── earthquake/
│   ├── flood/
│   └── wildfire/
├── validation/     # Validation set (15%)
│   ├── cyclone/
│   ├── earthquake/
│   ├── flood/
│   └── wildfire/
└── test/           # Test set (15%)
    ├── cyclone/
    ├── earthquake/
    ├── flood/
    └── wildfire/
```

### Dataset Characteristics
- **Image Format**: JPEG/PNG
- **Average Image Size**: ~300KB
- **Aspect Ratios**: Variable (normalized to 224×224 during training)
- **Color Mode**: RGB
- **Classes**: Perfectly balanced across all splits

### Data Augmentation Applied
- **Rotation**: ±20 degrees
- **Width/Height Shifts**: 20% of total dimensions
- **Shear Transformation**: 15% intensity
- **Zoom**: 20% in/out
- **Horizontal Flip**: Random
- **Brightness**: 80-120% of original
- **Fill Mode**: Nearest neighbor interpolation

---

## 🎯 Performance Metrics

### Overall Test Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 87.75% |
| **AUC** | 0.9602 |
| **Precision** | 0.8797 |
| **Recall** | 0.8775 |
| **Loss** | 0.6581 |

### Per-Class Performance

| Disaster Type | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| **Cyclone** | 0.93 | 0.88 | 0.90 | Strong |
| **Earthquake** | 0.82 | 0.97 | 0.89 | Excellent Recall |
| **Flood** | 0.97 | 0.75 | 0.85 | High Precision |
| **Wildfire** | 0.83 | 0.91 | 0.87 | Balanced |

### Key Insights
- ✅ **Best Performance**: Cyclone (F1: 0.90) with 93% precision
- ✅ **Highest Recall**: Earthquake (97%) - rarely misses earthquakes
- ✅ **Highest Precision**: Flood (97%) - very confident predictions
- ⚠️ **Area for Improvement**: Flood recall (75%) - some floods misclassified

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/Dhanugupta0/Disaster-classifier.git
cd Disaster-classifier
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r deployment/requirements.txt
```

### Required Packages
```
tensorflow>=2.10.0
flask>=2.0.0
numpy>=1.21.0
pillow>=9.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

---

## 💻 Usage

### Option 1: Web Interface

1. **Start the Flask server:**
```bash
cd deployment
python app.py
```

2. **Open browser and navigate to:**
```
http://localhost:5000
```

3. **Upload an image** of a disaster and get instant predictions!

### Option 2: REST API

**Endpoint**: `POST /predict`

**Example using cURL:**
```bash
curl -X POST -F "image=@path/to/disaster_image.jpg" \
     http://localhost:5000/predict
```

**Example Response:**
```json
{
  "predicted_class": "wildfire",
  "confidence": 0.94,
  "all_predictions": {
    "cyclone": 0.02,
    "earthquake": 0.01,
    "flood": 0.03,
    "wildfire": 0.94
  }
}
```

### Option 3: Python Script

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import numpy as np
import json

# Load model
model = tf.keras.models.load_model('model/disaster_model.keras')

# Load config
with open('config/model_info.json', 'r') as f:
    config = json.load(f)
CLASSES = config['classes']

# Prepare image
def prepare_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Predict
image_path = 'path/to/disaster_image.jpg'
prepared_img = prepare_image(image_path)
predictions = model.predict(prepared_img)
predicted_class = CLASSES[np.argmax(predictions[0])]
confidence = predictions[0][np.argmax(predictions[0])]

print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
```

---

## 📁 Project Structure

```
Disaster-classifier/
├── config/
│   └── model_info.json              # Model configuration & metadata
├── Dataset/                          # Dataset directory (gitignored)
│   ├── train/
│   ├── validation/
│   └── test/
├── deployment/
│   ├── app.py                       # Flask application
│   ├── requirements.txt             # Python dependencies
│   ├── model/
│   │   └── disaster_model.keras     # Trained model
│   ├── static/                      # CSS, JS files
│   └── templates/
│       └── index.html               # Web interface
├── model/                           # Model artifacts (gitignored)
│   ├── disaster_model.keras
│   ├── disaster_model_savedmodel/
│   ├── checkpoints/
│   └── logs/
├── notebooks/
│   └── Model-training.ipynb         # Complete training pipeline
├── results/                         # Visualizations & reports
│   ├── training_history_enhanced.png
│   ├── test_confusion_matrix.png
│   ├── class_distribution.png
│   └── augmentation_samples.png
├── src/
│   ├── __init__.py
│   ├── data_exploration.py          # Data analysis utilities
│   ├── preprocessing.py             # Data preprocessing
│   ├── train.py                     # Training script
│   └── predict.py                   # Inference script
├── .gitignore
└── README.md
```

---

## 🎓 Training Details

### Training Configuration
- **Optimizer**: Adam
  - Phase 1 Learning Rate: 1e-3
  - Phase 2 Learning Rate: 1e-5 (fine-tuning)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Image Size**: 224×224 pixels
- **Training Strategy**: Two-phase approach

### Phase 1: Classifier Head Training
- **Duration**: Up to 25 epochs
- **Base Model**: Frozen (weights locked)
- **Target**: Train only the classifier head
- **Early Stopping**: Based on validation loss (patience: 8)
- **Learning Rate Reduction**: Factor 0.5 on plateau (patience: 4)

### Phase 2: Fine-tuning (Conditional)
- **Trigger**: Only if Phase 1 validation accuracy < 70%
- **Duration**: Up to 35 epochs
- **Base Model**: Top 50 layers unfrozen
- **Learning Rate**: Reduced to 1e-5 for stable fine-tuning
- **Goal**: Improve feature extraction for disaster-specific patterns

### Callbacks & Monitoring
1. **ModelCheckpoint**: Saves best model based on validation accuracy
2. **EarlyStopping**: Prevents overfitting (patience: 8 epochs)
3. **ReduceLROnPlateau**: Adaptive learning rate adjustment
4. **TensorBoard**: Real-time training visualization
5. **OverfittingMonitor**: Custom callback to detect train-val gap > 15%

### Training Results
- **Total Epochs**: ~25 (Phase 1 only, sufficient performance achieved)
- **Training Accuracy**: 98%
- **Validation Accuracy**: 97.5%
- **Test Accuracy**: 87.75%
- **Training Time**: ~45 minutes on GPU

---

## 📡 API Documentation

### Endpoints

#### 1. Home Page
```
GET /
```
Returns the web interface HTML page.

#### 2. Predict Disaster
```
POST /predict
Content-Type: multipart/form-data
```

**Request Parameters:**
- `image` (required): Image file (JPEG/PNG)

**Response Format:**
```json
{
  "predicted_class": "string",
  "confidence": float,
  "all_predictions": {
    "cyclone": float,
    "earthquake": float,
    "flood": float,
    "wildfire": float
  }
}
```

**Error Response:**
```json
{
  "error": "Error message description"
}
```

**Status Codes:**
- `200`: Successful prediction
- `400`: Bad request (no image provided)
- `500`: Internal server error

---

## 📈 Results & Visualizations

### Training History
The model shows excellent convergence with controlled overfitting:
- Training and validation curves closely aligned
- No significant overfitting detected (gap < 15%)
- Smooth learning curve with gradual improvement

### Confusion Matrix
The confusion matrix reveals:
- Strong diagonal (correct predictions)
- Minimal confusion between disaster types
- Slight confusion between visually similar disasters

### Class Distribution
The dataset is perfectly balanced:
- Each class has equal representation
- No class imbalance issues
- Fair evaluation across all categories

### Augmentation Samples
Data augmentation creates diverse training samples:
- Realistic variations in orientation
- Lighting condition variations
- Scale and position changes
- Enhanced model robustness

---

## 🔍 Key Observations

### Strengths
✅ **High Overall Accuracy** (87.75%) suitable for production use  
✅ **Excellent AUC** (0.96) indicates strong classification capability  
✅ **Balanced Performance** across all disaster types  
✅ **Robust Augmentation** prevents overfitting effectively  
✅ **Fast Inference** suitable for real-time applications  

### Areas for Future Improvement
🔄 Improve flood recall through additional training data  
🔄 Explore ensemble methods for higher accuracy  
🔄 Add more disaster categories (tsunami, tornado, etc.)  
🔄 Implement model quantization for edge deployment  
🔄 Add explainability features (Grad-CAM visualizations)  

---

## 🛠️ Development

### Training from Scratch

1. **Prepare your dataset** in the structure shown above
2. **Open the training notebook:**
```bash
jupyter notebook notebooks/Model-training.ipynb
```
3. **Run all cells** to train the model
4. **Monitor training** with TensorBoard:
```bash
tensorboard --logdir=model/logs
```

### Customization Options

**Modify hyperparameters** in the notebook:
```python
IMG_SIZE = 224              # Change input image size
BATCH_SIZE = 32             # Adjust batch size
NUM_CLASSES = 4             # Add more disaster types
```

**Adjust augmentation intensity:**
```python
rotation_range=20           # Rotation degrees
zoom_range=0.2              # Zoom percentage
brightness_range=[0.8, 1.2] # Brightness multiplier
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Dhanu Gupta**
- GitHub: [@Dhanugupta0](https://github.com/Dhanugupta0)
- Repository: [Disaster-classifier](https://github.com/Dhanugupta0/Disaster-classifier)

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** team for the excellent deep learning framework
- **ResNet50** architecture by He et al. for transfer learning
- **ImageNet** dataset for pre-trained weights
- Disaster dataset contributors for making this research possible

---

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the notebook documentation

---

## 🔮 Future Roadmap

- [ ] Add mobile deployment (TensorFlow Lite)
- [ ] Implement real-time video stream processing
- [ ] Add multi-language support for web interface
- [ ] Create Docker container for easy deployment
- [ ] Integrate with emergency response systems
- [ ] Add Grad-CAM visualizations for interpretability
- [ ] Implement model versioning and A/B testing
- [ ] Create automated retraining pipeline

---

## 📊 Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{disaster_classifier_2025,
  author = {Dhanu Gupta},
  title = {Disaster Classifier: Deep Learning for Natural Disaster Recognition},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Dhanugupta0/Disaster-classifier}
}
```

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Dhanu Gupta](https://github.com/Dhanugupta0)

</div>
