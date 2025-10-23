import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import numpy as np

# Configuration
MODEL_NAME = 'disaster_model'
IMG_SIZE = 224
NUM_CLASSES = 4
CLASSES = ['cyclone', 'earthquake', 'flood', 'wildfire']

# =============================================
# BUILD MODEL FIRST
# =============================================

from tensorflow.keras.applications import ResNet50

print("\n🔨 Building ResNet50 Model with ImageNet weights...")

base_model = ResNet50(
    include_top=False,
    weights='imagenet',  # This WILL work
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

print("✅ ResNet50 model built with ImageNet weights!")
print(f"Total parameters: {model.count_params():,}")

# =============================================
# OVERFITTING MONITOR CALLBACK
# =============================================

class OverfittingMonitor(keras.callbacks.Callback):
    """Custom callback to detect and stop training when overfitting occurs"""
    
    def __init__(self, threshold=0.1, patience=3):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.overfit_count = 0
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        
        if train_acc is None or val_acc is None:
            return
        
        acc_gap = train_acc - val_acc
        
        if acc_gap > self.threshold:
            self.overfit_count += 1
            print(f"\n⚠️ Overfitting detected! Train-Val accuracy gap: {acc_gap:.4f}")
            
            if self.overfit_count >= self.patience:
                print(f"\n🛑 Stopping training: Persistent overfitting detected for {self.patience} epochs")
                self.model.stop_training = True
        else:
            self.overfit_count = 0
        
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

def create_enhanced_callbacks(model_name='disaster_model', monitor_overfitting=True):
    """Create enhanced training callbacks with overfitting detection"""
    
    os.makedirs('model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=f'model/{model_name}_best.keras',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    if monitor_overfitting:
        callbacks.append(
            OverfittingMonitor(threshold=0.15, patience=3)
        )
    
    return callbacks

# =============================================
# ENHANCED TRAINING FUNCTION
# =============================================

def train_model_enhanced(model, base_model, train_gen, val_gen,
                        epochs_phase1=25, epochs_phase2=35):
    """Enhanced two-phase training with overfitting detection"""
    
    print("\n" + "="*70)
    print("PHASE 1: Training Classifier Head (Base Model Frozen)")
    print("="*70)
    
    base_model.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase1,
        callbacks=create_enhanced_callbacks(f'{MODEL_NAME}_phase1', monitor_overfitting=True),
        verbose=1
    )
    
    final_train_acc = history_phase1.history['accuracy'][-1]
    final_val_acc = history_phase1.history['val_accuracy'][-1]
    
    print(f"\n📊 Phase 1 Results:")
    print(f"  • Final Training Accuracy: {final_train_acc:.4f}")
    print(f"  • Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"  • Accuracy Gap: {final_train_acc - final_val_acc:.4f}")
    
    if final_val_acc < 0.70:
        print("\n" + "="*70)
        print("PHASE 2: Fine-tuning Top Layers")
        print("="*70)
        
        base_model.trainable = True
        freeze_until = len(base_model.layers) - 50
        
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        
        trainable_count = len([l for l in base_model.layers if l.trainable])
        print(f"Unfrozen {trainable_count}/{len(base_model.layers)} layers in base model")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        
        print("Reduced learning rate to 1e-5 for fine-tuning")
        print("="*70)
        
        history_phase2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_phase2,
            initial_epoch=len(history_phase1.history['loss']),
            callbacks=create_enhanced_callbacks(f'{MODEL_NAME}_phase2', monitor_overfitting=True),
            verbose=1
        )
        
        final_train_acc = history_phase2.history['accuracy'][-1]
        final_val_acc = history_phase2.history['val_accuracy'][-1]
        
        print(f"\n📊 Phase 2 Results:")
        print(f"  • Final Training Accuracy: {final_train_acc:.4f}")
        print(f"  • Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"  • Accuracy Gap: {final_train_acc - final_val_acc:.4f}")
        
    else:
        print("\n✅ Phase 1 achieved good performance. Skipping Phase 2.")
        history_phase2 = None
    
    return model, history_phase1, history_phase2

# =============================================
# ENHANCED VISUALIZATION
# =============================================

def plot_training_history_enhanced(history_phase1, history_phase2=None):
    """Enhanced visualization with overfitting indicators"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History with Overfitting Analysis', fontsize=16, fontweight='bold')
    
    if history_phase2:
        train_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
        val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
        train_loss = history_phase1.history['loss'] + history_phase2.history['loss']
        val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
        train_auc = history_phase1.history['auc'] + history_phase2.history['auc']
        val_auc = history_phase1.history['val_auc'] + history_phase2.history['val_auc']
        phase_split = len(history_phase1.history['loss'])
    else:
        train_acc = history_phase1.history['accuracy']
        val_acc = history_phase1.history['val_accuracy']
        train_loss = history_phase1.history['loss']
        val_loss = history_phase1.history['val_loss']
        train_auc = history_phase1.history['auc']
        val_auc = history_phase1.history['val_auc']
        phase_split = None
    
    epochs = range(1, len(train_acc) + 1)
    
    # Plot 1: Accuracy
    axes[0, 0].plot(epochs, train_acc, 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
    if phase_split:
        axes[0, 0].axvline(x=phase_split, color='green', linestyle='--', label='Fine-tuning starts')
    axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss
    axes[0, 1].plot(epochs, train_loss, 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
    if phase_split:
        axes[0, 1].axvline(x=phase_split, color='green', linestyle='--', label='Fine-tuning starts')
    axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: AUC
    axes[0, 2].plot(epochs, train_auc, 'b-', label='Training', linewidth=2)
    axes[0, 2].plot(epochs, val_auc, 'r-', label='Validation', linewidth=2)
    if phase_split:
        axes[0, 2].axvline(x=phase_split, color='green', linestyle='--', label='Fine-tuning starts')
    axes[0, 2].set_title('AUC', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Overfitting Gap
    acc_gap = [t - v for t, v in zip(train_acc, val_acc)]
    axes[1, 0].plot(epochs, acc_gap, 'purple', linewidth=2)
    axes[1, 0].axhline(y=0.15, color='red', linestyle='--', label='Overfitting threshold')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    if phase_split:
        axes[1, 0].axvline(x=phase_split, color='green', linestyle='--', label='Fine-tuning starts')
    axes[1, 0].fill_between(epochs, 0, acc_gap, where=[g > 0.15 for g in acc_gap],
                            color='red', alpha=0.3, label='Overfitting zone')
    axes[1, 0].set_title('Train-Val Accuracy Gap', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gap')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Loss Comparison
    axes[1, 1].plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    axes[1, 1].plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')
    axes[1, 1].set_title('Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary
    axes[1, 2].axis('off')
    summary_text = f"""
    TRAINING SUMMARY
    
    Best Validation Accuracy: {max(val_acc):.4f}
    Final Validation Accuracy: {val_acc[-1]:.4f}
    
    Best Training Accuracy: {max(train_acc):.4f}
    Final Training Accuracy: {train_acc[-1]:.4f}
    
    Max Accuracy Gap: {max(acc_gap):.4f}
    Final Accuracy Gap: {acc_gap[-1]:.4f}
    
    Total Epochs: {len(epochs)}
    Overfitting Status: {'⚠️ DETECTED' if max(acc_gap) > 0.15 else '✅ CONTROLLED'}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('model/training_history_enhanced.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================
# START TRAINING
# =============================================

print("\n" + "="*70)
print("🚀 STARTING ENHANCED TRAINING WITH OVERFITTING DETECTION")
print("="*70)

trained_model, hist1, hist2 = train_model_enhanced(
    model, 
    base_model, 
    train_gen, 
    val_gen,
    epochs_phase1=25,
    epochs_phase2=35
)

plot_training_history_enhanced(hist1, hist2)

# =============================================
# SAVE FOR DEPLOYMENT
# =============================================

os.makedirs("model", exist_ok=True)
os.makedirs("config", exist_ok=True)

# Use .export() for SavedModel format (Keras 3 requirement)
trained_model.export("model/disaster_model_savedmodel")

# Also save as .keras for backup
trained_model.save("model/disaster_model.keras")

# Save model info
model_info = {
    "classes": CLASSES,
    "input_shape": [IMG_SIZE, IMG_SIZE, 3],
    "best_val_acc": float(max(hist1.history['val_accuracy'] + (hist2.history['val_accuracy'] if hist2 else [])))
}

with open("config/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print("\n📦 Deployment Artifacts:")
print("  • model/disaster_model_savedmodel/ (for TF Serving/deployment)")
print("  • model/disaster_model.keras (native Keras format)")
print("  • config/model_info.json (class names & config)")
print("  • model/training_history_enhanced.png (training visualization)")
print("\n🚀 Ready for Flask deployment!")
