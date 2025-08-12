import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from pathlib import Path
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# =================== CONFIGURATION ===================

class Config:
    # Paths - UPDATE THESE TO YOUR PATHS
    DATASET_DIR = "./dataset"  # Main dataset directory
    TRAIN_DIR = os.path.join(DATASET_DIR, "anorganik")  # anorganik folder
    ORGANIC_DIR = os.path.join(DATASET_DIR, "organik")  # organik folder
    
    # Model settings
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 100
    INITIAL_LR = 1e-3
    
    # Training settings
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model architecture
    USE_EFFICIENTNET = True  # True for EfficientNet, False for MobileNetV2
    DROPOUT_RATE = 0.5
    
    # Output
    MODEL_NAME = "high_accuracy_waste_classifier"
    SAVE_DIR = "./models"

config = Config()

# Create output directory
os.makedirs(config.SAVE_DIR, exist_ok=True)

# =================== GPU CONFIGURATION ===================

def configure_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured successfully: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")

configure_gpu()

# =================== DATA PREPROCESSING ===================

def create_dataset_structure():
    """Convert your existing structure to train/val format"""
    
    # Create organized structure
    organized_dir = "./dataset_organized"
    
    splits = ['train', 'validation']
    classes = ['organic', 'inorganic']
    
    # Create directory structure
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(organized_dir, split, class_name), exist_ok=True)
    
    # Get all files
    organic_files = list(Path(config.ORGANIC_DIR).glob('**/*'))
    inorganic_files = list(Path(config.TRAIN_DIR).glob('**/*'))  # anorganik
    
    # Filter only image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    organic_files = [f for f in organic_files if f.suffix.lower() in image_extensions]
    inorganic_files = [f for f in inorganic_files if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(organic_files)} organic images")
    print(f"Found {len(inorganic_files)} inorganic images")
    
    # Shuffle and split
    np.random.shuffle(organic_files)
    np.random.shuffle(inorganic_files)
    
    # Split organic
    org_split = int(len(organic_files) * (1 - config.VALIDATION_SPLIT))
    org_train = organic_files[:org_split]
    org_val = organic_files[org_split:]
    
    # Split inorganic  
    inorg_split = int(len(inorganic_files) * (1 - config.VALIDATION_SPLIT))
    inorg_train = inorganic_files[:inorg_split]
    inorg_val = inorganic_files[inorg_split:]
    
    # Copy files
    import shutil
    
    # Training set
    for i, file in enumerate(org_train):
        shutil.copy2(file, os.path.join(organized_dir, 'train', 'organic', f'org_{i:06d}.jpg'))
    
    for i, file in enumerate(inorg_train):
        shutil.copy2(file, os.path.join(organized_dir, 'train', 'inorganic', f'inorg_{i:06d}.jpg'))
    
    # Validation set
    for i, file in enumerate(org_val):
        shutil.copy2(file, os.path.join(organized_dir, 'validation', 'organic', f'org_val_{i:06d}.jpg'))
    
    for i, file in enumerate(inorg_val):
        shutil.copy2(file, os.path.join(organized_dir, 'validation', 'inorganic', f'inorg_val_{i:06d}.jpg'))
    
    print(f"✅ Dataset organized in: {organized_dir}")
    print(f"Training: {len(org_train)} organic, {len(inorg_train)} inorganic")
    print(f"Validation: {len(org_val)} organic, {len(inorg_val)} inorganic")
    
    return organized_dir

# =================== ADVANCED DATA AUGMENTATION ===================

def create_advanced_generators(data_dir):
    """Create data generators with advanced augmentation"""
    
    # Heavy augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=60,
        fill_mode='nearest',
        # Advanced augmentations
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False
    )
    
    # Minimal augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    return train_generator, validation_generator

# =================== ADVANCED MODEL ARCHITECTURE ===================

def create_high_accuracy_model():
    """Create high-accuracy model with advanced architecture"""
    
    if config.USE_EFFICIENTNET:
        # EfficientNetB4 for maximum accuracy
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
        )
        print("🚀 Using EfficientNetB4 for maximum accuracy")
    else:
        # MobileNetV2 for efficiency
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
        )
        print("⚡ Using MobileNetV2 for efficiency")
    
    # Progressive unfreezing - start with frozen base
    base_model.trainable = False
    
    # Advanced custom head
    model = keras.Sequential([
        base_model,
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Batch Normalization
        layers.BatchNormalization(),
        
        # First dense block
        layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(config.DROPOUT_RATE),
        
        # Second dense block
        layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(config.DROPOUT_RATE * 0.7),
        
        # Third dense block
        layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(config.DROPOUT_RATE * 0.5),
        
        # Output layer
        layers.Dense(2, activation='softmax')
    ])
    
    return model

# =================== ADVANCED TRAINING STRATEGY ===================

def create_advanced_callbacks():
    """Create comprehensive callbacks for optimal training"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config.SAVE_DIR, f"{config.MODEL_NAME}_{timestamp}.h5")
    
    callbacks = [
        # Model checkpoint - save best model
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        
        # Early stopping with patience
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-8,
            mode='min',
            verbose=1
        ),
        
        # Learning rate scheduler
        keras.callbacks.LearningRateScheduler(
            lambda epoch: config.INITIAL_LR * 0.95 ** epoch
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            os.path.join(config.SAVE_DIR, f"training_log_{timestamp}.csv")
        ),
        
        # TensorBoard (optional)
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.SAVE_DIR, f"tensorboard_{timestamp}"),
            histogram_freq=1
        )
    ]
    
    return callbacks, model_path

def progressive_training(model, train_gen, val_gen):
    """Progressive training with multiple stages"""
    
    print("🎯 Starting Progressive Training Strategy")
    
    # Stage 1: Train head only
    print("\n" + "="*50)
    print("STAGE 1: Training custom head (base frozen)")
    print("="*50)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    callbacks, model_path = create_advanced_callbacks()
    
    history_stage1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )
    
    # Stage 2: Unfreeze top layers
    print("\n" + "="*50)
    print("STAGE 2: Fine-tuning top layers")
    print("="*50)
    
    # Unfreeze top layers
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze early layers
    if config.USE_EFFICIENTNET:
        freeze_at = len(base_model.layers) - 50
    else:
        freeze_at = len(base_model.layers) - 30
    
    for layer in base_model.layers[:freeze_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.INITIAL_LR * 0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    history_stage2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    # Stage 3: Full fine-tuning
    print("\n" + "="*50)
    print("STAGE 3: Full fine-tuning")
    print("="*50)
    
    # Unfreeze all layers
    for layer in base_model.layers:
        layer.trainable = True
    
    # Compile with very low learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.INITIAL_LR * 0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    history_stage3 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS - 80,
        callbacks=callbacks,
        verbose=1
    )
    
    return [history_stage1, history_stage2, history_stage3], model_path

# =================== EVALUATION ===================

def comprehensive_evaluation(model, val_gen):
    """Comprehensive model evaluation"""
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    class_names = list(val_gen.class_indices.keys())
    
    # Classification report
    print("\n📊 Classification Report:")
    print("-" * 40)
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_names, 
                                 digits=4)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, 'confusion_matrix.png'), dpi=300)
    plt.show()
    
    # Per-class metrics
    print("\n📈 Per-Class Accuracy:")
    print("-" * 30)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, (class_name, acc) in enumerate(zip(class_names, class_accuracy)):
        print(f"{class_name:12}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Overall accuracy
    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    print(f"\n🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Find misclassified samples
    misclassified_indices = np.where(predicted_classes != true_classes)[0]
    print(f"❌ Misclassified samples: {len(misclassified_indices)}")
    
    if len(misclassified_indices) > 0:
        print("\n🔍 Worst predictions (lowest confidence):")
        confidences = np.max(predictions, axis=1)
        worst_indices = misclassified_indices[np.argsort(confidences[misclassified_indices])][:5]
        
        for idx in worst_indices:
            true_class = class_names[true_classes[idx]]
            pred_class = class_names[predicted_classes[idx]]
            confidence = confidences[idx]
            print(f"  True: {true_class}, Predicted: {pred_class}, Confidence: {confidence:.3f}")
    
    return cm, overall_accuracy

def plot_training_history(histories):
    """Plot training history from all stages"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Combine histories
    combined_history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    for history in histories:
        for key in combined_history.keys():
            combined_history[key].extend(history.history[key])
    
    epochs = range(1, len(combined_history['accuracy']) + 1)
    
    # Accuracy
    axes[0, 0].plot(epochs, combined_history['accuracy'], 'b-', label='Training')
    axes[0, 0].plot(epochs, combined_history['val_accuracy'], 'r-', label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(epochs, combined_history['loss'], 'b-', label='Training')
    axes[0, 1].plot(epochs, combined_history['val_loss'], 'r-', label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate (if available)
    if 'lr' in histories[-1].history:
        axes[1, 0].plot(epochs, combined_history.get('lr', []), 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Best accuracy
    best_val_acc = max(combined_history['val_accuracy'])
    best_epoch = combined_history['val_accuracy'].index(best_val_acc) + 1
    
    axes[1, 1].text(0.1, 0.8, f'Best Validation Accuracy:', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'{best_val_acc:.4f} ({best_val_acc*100:.2f}%)', 
                    fontsize=16, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'At Epoch: {best_epoch}', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, 'training_history.png'), dpi=300)
    plt.show()

# =================== MAIN TRAINING FUNCTION ===================

def main_training():
    """Main training function"""
    
    print("🚀 HIGH-ACCURACY WASTE CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Max Epochs: {config.EPOCHS}")
    print(f"Architecture: {'EfficientNetB4' if config.USE_EFFICIENTNET else 'MobileNetV2'}")
    print("=" * 60)
    
    # Step 1: Organize dataset
    print("\n📁 Step 1: Organizing dataset...")
    organized_dir = create_dataset_structure()
    
    # Step 2: Create data generators
    print("\n🔄 Step 2: Creating data generators...")
    train_gen, val_gen = create_advanced_generators(organized_dir)
    
    print(f"✅ Training samples: {train_gen.samples}")
    print(f"✅ Validation samples: {val_gen.samples}")
    print(f"✅ Classes: {list(train_gen.class_indices.keys())}")
    
    # Step 3: Create model
    print("\n🏗️ Step 3: Building model...")
    model = create_high_accuracy_model()
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Step 4: Progressive training
    print("\n🎯 Step 4: Starting progressive training...")
    histories, best_model_path = progressive_training(model, train_gen, val_gen)
    
    # Step 5: Load best model and evaluate
    print(f"\n📊 Step 5: Loading best model from {best_model_path}")
    best_model = keras.models.load_model(best_model_path)
    
    # Step 6: Comprehensive evaluation
    print("\n📈 Step 6: Comprehensive evaluation...")
    cm, accuracy = comprehensive_evaluation(best_model, val_gen)
    
    # Step 7: Plot training history
    print("\n📊 Step 7: Plotting training history...")
    plot_training_history(histories)
    
    # Step 8: Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_path': best_model_path,
        'final_accuracy': float(accuracy),
        'architecture': 'EfficientNetB4' if config.USE_EFFICIENTNET else 'MobileNetV2',
        'image_size': config.IMG_SIZE,
        'batch_size': config.BATCH_SIZE,
        'total_epochs': sum(len(h.history['accuracy']) for h in histories),
        'training_samples': train_gen.samples,
        'validation_samples': val_gen.samples
    }
    
    with open(os.path.join(config.SAVE_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"🏆 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"💾 Best Model: {best_model_path}")
    print(f"📁 All outputs saved in: {config.SAVE_DIR}")
    print("=" * 60)
    
    return best_model, accuracy, best_model_path

# =================== PREDICTION FUNCTION ===================

def predict_single_image(model_path, image_path):
    """Predict a single image"""
    
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    classes = ['inorganic', 'organic']  # Based on alphabetical order
    
    print(f"Image: {image_path}")
    print(f"Prediction: {classes[predicted_class]}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    return classes[predicted_class], confidence

# =================== RUN TRAINING ===================

if __name__ == "__main__":
    # Run training
    model, accuracy, model_path = main_training()
    
    print(f"\n✨ Training completed!")
    print(f"✨ Use predict_single_image('{model_path}', 'your_image.jpg') to test new images")