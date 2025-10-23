import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# =============================================
# LOAD MODEL AND CONFIG
# =============================================
model = tf.keras.models.load_model("model/disaster_model.keras")

with open("config/model_info.json", "r") as f:
    info = json.load(f)
CLASSES = info["classes"]
IMG_SIZE = info["input_shape"][0]

print("\n✅ Loaded model and configuration successfully!")
print(f"Classes: {CLASSES}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")

# =============================================
# LOAD TEST DATASET (Using same preprocessing as training)
# =============================================
test_dir = os.path.join("Dataset", "test")  # Adjust automatically for your setup
if not os.path.exists(test_dir):
    test_dir = os.path.join("..", "Dataset", "test")  # fallback if one level above

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    classes=CLASSES,  # Force same class order as training
    shuffle=False
)

print("\n🧠 Starting model evaluation...")

# =============================================
# MODEL EVALUATION
# =============================================
loss, accuracy, auc, precision, recall = model.evaluate(test_gen)
print("\n📊 Test Metrics:")
print(f"  • Accuracy:  {accuracy*100:.2f}%")
print(f"  • AUC:       {auc:.4f}")
print(f"  • Precision: {precision:.4f}")
print(f"  • Recall:    {recall:.4f}")
print(f"  • Loss:      {loss:.4f}")

# =============================================
# PREDICTIONS & REPORTS
# =============================================
print("\n🔍 Generating classification report...")
y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

# =============================================
# CONFUSION MATRIX VISUALIZATION
# =============================================
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Test Set Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

os.makedirs("model", exist_ok=True)
plt.savefig("model/test_confusion_matrix.png", dpi=150)
plt.show()

print("\n✅ Evaluation complete! Confusion matrix saved as 'model/test_confusion_matrix.png'")
