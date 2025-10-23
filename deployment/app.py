from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

MODEL_PATH = os.path.join('deployment', 'model', 'disaster_model.keras')
CONFIG_PATH = os.path.join('config', 'model_info.json')

model = tf.keras.models.load_model(MODEL_PATH)
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
CLASSES = config['classes']
IMG_SIZE = config['input_shape'][0]

print(f"✅ Model loaded successfully!")
print(f"Classes: {CLASSES}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")

def prepare_image(image, target_size):
    """Prepare image using ResNet preprocessing (same as training)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((target_size, target_size))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    # Use ResNet preprocessing (normalizes to [-1, 1])
    image_array = preprocess_input(image_array)
    return image_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    try:
        image = Image.open(file.stream)
        input_arr = prepare_image(image, IMG_SIZE)
        preds = model.predict(input_arr)
        pred_class_idx = np.argmax(preds[0])
        confidence = float(preds[0][pred_class_idx])
        pred_class = CLASSES[pred_class_idx]
        
        # Log predictions for debugging
        print("\n🔍 Prediction probabilities:")
        for i, cls in enumerate(CLASSES):
            print(f"  {cls}: {preds[0][i]*100:.2f}%")
        
        return jsonify({
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_predictions': {cls: float(preds[0][i]) for i, cls in enumerate(CLASSES)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
