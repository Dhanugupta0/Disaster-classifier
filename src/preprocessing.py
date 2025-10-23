import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# =============================================
# CONFIGURATION
# =============================================

BASE_PATH = "../Dataset"
TRAIN_DIR = os.path.join(BASE_PATH, 'train')
VAL_DIR = os.path.join(BASE_PATH, 'validation')
TEST_DIR = os.path.join(BASE_PATH, 'test')

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASSES = ['cyclone', 'earthquake', 'flood', 'wildfire']

np.random.seed(42)
tf.random.set_seed(42)

# =============================================
# DATA GENERATORS (FOR TRAINING)
# =============================================

def create_data_generators():
    """Create data generators with augmentation for training"""
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

# =============================================
# FIXED VISUALIZATION (WITHOUT PREPROCESSING)
# =============================================

def visualize_augmentations_fixed(num_samples=9):
    """
    Visualize augmented images WITHOUT preprocessing for proper display
    """
    # Create augmentation generator WITHOUT preprocessing
    vis_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        rescale=1./255  # Only rescale for visualization
    )
    
    vis_generator = vis_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    # Get a batch of images
    images, labels = next(vis_generator)
    
    # Create subplot
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Data Augmentation Samples', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        
        # Get class label
        class_idx = np.argmax(labels[i])
        class_name = CLASSES[class_idx]
        
        # Display
        axes[i].imshow(img)
        axes[i].set_title(f'{class_name.capitalize()}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_original_vs_augmented_fixed():
    """
    Compare original and augmented versions - FIXED VERSION
    """
    # Load a sample image
    sample_class = 'cyclone'
    sample_dir = os.path.join(TRAIN_DIR, sample_class)
    sample_image = os.listdir(sample_dir)[0]
    sample_path = os.path.join(sample_dir, sample_image)
    
    # Load and preprocess original
    img = Image.open(sample_path)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    
    # Create augmentation generator (WITHOUT preprocessing for visualization)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
    )
    
    # Generate augmented versions
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Original vs Augmented - {sample_class.capitalize()}', fontsize=16, fontweight='bold')
    
    # Show original
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Generate and show 7 augmented versions
    aug_iter = datagen.flow(img_array_expanded, batch_size=1)
    
    positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    aug_names = ['Rotation + Shift', 'Zoom + Flip', 'Brightness', 'Shear', 
                 'Combined 1', 'Combined 2', 'Combined 3']
    
    for idx, (row, col) in enumerate(positions):
        aug_img = next(aug_iter)[0].astype('uint8')
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(aug_names[idx], fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('original_vs_augmented.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================
# SIMPLE SIDE-BY-SIDE COMPARISON
# =============================================

def show_original_vs_augmented_simple():
    """Show 4 original images vs their augmented versions"""
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Original (Left) vs Augmented (Right) for Each Class', fontsize=16, fontweight='bold')
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    
    for i, class_name in enumerate(CLASSES):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        sample_image = os.listdir(class_dir)[0]
        sample_path = os.path.join(class_dir, sample_image)
        
        # Load original
        img = Image.open(sample_path)
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized)
        
        # Show original
        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f'{class_name.capitalize()} - Original', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Generate 3 augmented versions
        img_array_expanded = np.expand_dims(img_array, axis=0)
        aug_iter = datagen.flow(img_array_expanded, batch_size=1)
        
        for j in range(3):
            aug_img = next(aug_iter)[0].astype('uint8')
            axes[i, j+1].imshow(aug_img)
            axes[i, j+1].set_title(f'Augmented {j+1}')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('class_augmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================
# MAIN EXECUTION
# =============================================

print("="*70)
print("DATA AUGMENTATION PIPELINE SETUP")
print("="*70)

print("\n📊 Configuration:")
print(f"  • Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  • Batch Size: {BATCH_SIZE}")
print(f"  • Number of Classes: {NUM_CLASSES}")
print(f"  • Classes: {', '.join(CLASSES)}")

print("\n🔄 Creating Data Generators...")

# Create generators for training
train_gen, val_gen, test_gen = create_data_generators()

print("\n✅ Generators Created Successfully!")
print(f"\nDataset Statistics:")
print(f"  • Training samples: {train_gen.samples}")
print(f"  • Validation samples: {val_gen.samples}")
print(f"  • Test samples: {test_gen.samples}")
print(f"  • Steps per epoch (train): {len(train_gen)}")
print(f"  • Steps per epoch (val): {len(val_gen)}")

print("\n📈 Class Indices Mapping:")
for class_name, class_idx in train_gen.class_indices.items():
    print(f"  • {class_name.capitalize()}: {class_idx}")

print("\n" + "="*70)
print("VISUALIZING AUGMENTATIONS (FIXED)")
print("="*70)

# Fixed visualizations
visualize_augmentations_fixed(num_samples=9)
show_original_vs_augmented_simple()

print("\n✅ Augmentation visualization complete!")
print("📊 Generated files: augmentation_samples.png, class_augmentation_comparison.png")

print("\n" + "="*70)
print("✅ DATA GENERATORS READY FOR TRAINING!")
print("="*70)
print("\nUse these generators:")
print("  • train_gen - For training with augmentation")
print("  • val_gen - For validation (no augmentation)")
print("  • test_gen - For testing (no augmentation)")
print("\nNext Step: Build EfficientNet model!")
