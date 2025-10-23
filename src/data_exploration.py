import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# LOCAL paths
BASE_PATH = "../Dataset"
FULL_DATASET_PATH = os.path.join(BASE_PATH, "Cyclone_Wildfire_Flood_Earthquake_Dataset")

SPLIT_PATH = BASE_PATH  # 'train', 'validation', 'test' are subfolders here
SPLITS = ['train', 'test', 'validation']
CLASSES = ['cyclone', 'earthquake', 'flood', 'wildfire']

# ---- Functions below use these paths! ----

def count_images():
    split_results = {}
    for split in SPLITS:
        split_results[split] = {}
        for cls in CLASSES:
            path = os.path.join(SPLIT_PATH, split, cls)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                split_results[split][cls] = count
            else:
                split_results[split][cls] = 0

    full_results = {}
    for cls in CLASSES:
        cls_capitalized = cls.capitalize()
        path = os.path.join(FULL_DATASET_PATH, cls_capitalized)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            full_results[cls] = count
        else:
            full_results[cls] = 0

    return split_results, full_results

def analyze_image_properties(sample_size=100):
    properties = {
        'dimensions': [],
        'formats': [],
        'file_sizes': [],
        'aspect_ratios': [],
        'widths': [],
        'heights': []
    }
    for split in SPLITS:
        for cls in CLASSES:
            path = os.path.join(SPLIT_PATH, split, cls)
            if not os.path.exists(path):
                continue
            images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            sample = images[:min(sample_size, len(images))]
            for img_name in sample:
                img_path = os.path.join(path, img_name)
                try:
                    file_size = os.path.getsize(img_path) / 1024  # KB
                    properties['file_sizes'].append(file_size)
                    img = Image.open(img_path)
                    properties['dimensions'].append(img.size)
                    properties['formats'].append(img.format)
                    properties['widths'].append(img.size[0])
                    properties['heights'].append(img.size[1])
                    aspect_ratio = img.size[0] / img.size[1]
                    properties['aspect_ratios'].append(aspect_ratio)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    return properties

def check_corrupted_images():
    corrupted = []
    for split in SPLITS:
        for cls in CLASSES:
            path = os.path.join(SPLIT_PATH, split, cls)
            if not os.path.exists(path):
                continue
            images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for img_name in images:
                img_path = os.path.join(path, img_name)
                try:
                    img = Image.open(img_path)
                    img.verify()
                except Exception as e:
                    corrupted.append({'path': img_path, 'error': str(e)})
    return corrupted

def visualize_samples(samples_per_class=3):
    fig, axes = plt.subplots(len(CLASSES), samples_per_class, figsize=(15, 12))
    fig.suptitle('Sample Images from Each Disaster Class', fontsize=16, fontweight='bold')
    for i, cls in enumerate(CLASSES):
        path = os.path.join(SPLIT_PATH, 'train', cls)
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for j in range(min(samples_per_class, len(images))):
                img_path = os.path.join(path, images[j])
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f"{cls.capitalize()}\n{img.size[0]}x{img.size[1]}", fontweight='bold', fontsize=10)
                else:
                    axes[i, j].set_title(f"{img.size[0]}x{img.size[1]}", fontsize=9)
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_class_distribution(counts):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    df_counts = pd.DataFrame(counts).T
    df_counts.plot(kind='bar', stacked=False, ax=axes[0], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[0].set_title('Image Count by Split and Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Split', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    class_totals = {}
    for cls in CLASSES:
        class_totals[cls] = sum([counts[split][cls] for split in SPLITS])
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    axes[1].pie(class_totals.values(), labels=[c.capitalize() for c in class_totals.keys()],
                autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_image_properties(properties):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].hist(properties['widths'], bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Image Width Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axvline(np.mean(properties['widths']), color='red', linestyle='--',
                      label=f"Mean: {np.mean(properties['widths']):.0f}")
    axes[0, 0].legend()
    axes[0, 1].hist(properties['heights'], bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Image Height Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].axvline(np.mean(properties['heights']), color='red', linestyle='--',
                      label=f"Mean: {np.mean(properties['heights']):.0f}")
    axes[0, 1].legend()
    axes[1, 0].hist(properties['aspect_ratios'], bins=30, color='#45B7D1', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Aspect Ratio (Width/Height)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].axvline(np.mean(properties['aspect_ratios']), color='red', linestyle='--',
                      label=f"Mean: {np.mean(properties['aspect_ratios']):.2f}")
    axes[1, 0].legend()
    axes[1, 1].hist(properties['file_sizes'], bins=30, color='#FFA07A', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('File Size Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('File Size (KB)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].axvline(np.mean(properties['file_sizes']), color='red', linestyle='--',
                      label=f"Mean: {np.mean(properties['file_sizes']):.0f} KB")
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig('image_properties.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_summary():
    print("="*70)
    print("DISASTER CLASSIFICATION DATASET - EXPLORATION REPORT")
    print("="*70)
    print("\n📊 DATASET OVERVIEW")
    print("-"*70)
    split_counts, full_counts = count_images()
    print("\n1. PRE-SPLIT DATASET (train/test/validation)")
    print("-"*70)
    df_counts = pd.DataFrame(split_counts).T
    print(df_counts.to_string())
    total_split = sum([sum(split_counts[split].values()) for split in SPLITS])
    print(f"\nTotal Images in Split Dataset: {total_split}")
    print("\n2. FULL ORIGINAL DATASET")
    print("-"*70)
    for cls, count in full_counts.items():
        print(f"{cls.capitalize():12s}: {count:5d} images")
    total_full = sum(full_counts.values())
    print(f"\nTotal Images in Full Dataset: {total_full}")
    print("\n3. SPLIT DISTRIBUTION (Pre-split Dataset)")
    print("-"*70)
    for split in SPLITS:
        split_total = sum(split_counts[split].values())
        percentage = (split_total / total_split) * 100 if total_split > 0 else 0
        print(f"{split.capitalize():12s}: {split_total:5d} images ({percentage:5.2f}%)")
    print("\n4. CLASS BALANCE (Pre-split Dataset)")
    print("-"*70)
    class_totals = {}
    for cls in CLASSES:
        class_totals[cls] = sum([split_counts[split][cls] for split in SPLITS])
    for cls, count in class_totals.items():
        percentage = (count / total_split) * 100 if total_split > 0 else 0
        print(f"{cls.capitalize():12s}: {count:5d} images ({percentage:5.2f}%)")
    counts_list = list(class_totals.values())
    if len(set(counts_list)) == 1:
        print("\n✅ Dataset is perfectly balanced across all classes!")
    else:
        max_diff = max(counts_list) - min(counts_list)
        print(f"\n⚠️ Class imbalance detected. Max difference: {max_diff} images")
    print("\n5. IMAGE PROPERTIES ANALYSIS")
    print("-"*70)
    print("Analyzing image properties...")
    properties = analyze_image_properties(sample_size=100)
    if properties['dimensions']:
        print(f"Sample Size Analyzed: {len(properties['dimensions'])} images\n")
        print(f"Dimensions:")
        print(f"  Width  - Min: {min(properties['widths']):4d}, Max: {max(properties['widths']):4d}, Mean: {np.mean(properties['widths']):.0f}, Std: {np.std(properties['widths']):.0f}")
        print(f"  Height - Min: {min(properties['heights']):4d}, Max: {max(properties['heights']):4d}, Mean: {np.mean(properties['heights']):.0f}, Std: {np.std(properties['heights']):.0f}")
        print(f"\nAspect Ratios:")
        print(f"  Min: {min(properties['aspect_ratios']):.2f}, Max: {max(properties['aspect_ratios']):.2f}, Mean: {np.mean(properties['aspect_ratios']):.2f}")
        print(f"\nFile Sizes:")
        print(f"  Min: {min(properties['file_sizes']):.1f} KB, Max: {max(properties['file_sizes']):.1f} KB, Mean: {np.mean(properties['file_sizes']):.1f} KB")
        print(f"\nFormats:")
        format_counts = Counter(properties['formats'])
        for fmt, count in format_counts.items():
            print(f"  {fmt}: {count} images")
    print("\n6. DATA QUALITY CHECK")
    print("-"*70)
    corrupted = check_corrupted_images()
    if corrupted:
        print(f"⚠️ Found {len(corrupted)} corrupted images:")
        for item in corrupted[:10]:
            print(f"  - {item['path']}: {item['error']}")
    else:
        print("✅ No corrupted images found!")
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS...")
    print("="*70 + "\n")
    plot_class_distribution(split_counts)
    if properties['dimensions']:
        plot_image_properties(properties)
    visualize_samples(samples_per_class=4)
    print("\n" + "="*70)
    print("✅ DATA EXPLORATION COMPLETE!")
    print("="*70)
    print("\n📊 Generated Files:")
    print("  - class_distribution.png")
    print("  - image_properties.png")
    print("  - sample_images.png")
    print("\n💡 KEY INSIGHTS:")
    print(f"  • Total dataset size: {total_full:,} images")
    print(f"  • Pre-split dataset: {total_split:,} images")
    print(f"  • Training set: {sum(split_counts['train'].values())} images ({sum(split_counts['train'].values())/total_split*100:.1f}%)")
    print(f"  • Classes: {len(CLASSES)} (Cyclone, Earthquake, Flood, Wildfire)")
    if properties['dimensions']:
        print(f"  • Avg image size: {np.mean(properties['widths']):.0f}x{np.mean(properties['heights']):.0f} pixels")
        print(f"  • Avg file size: {np.mean(properties['file_sizes']):.1f} KB")
    return split_counts, full_counts, properties

# Run!
split_counts, full_counts, properties = generate_summary()
