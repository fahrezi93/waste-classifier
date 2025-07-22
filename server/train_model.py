import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path
import json
from datetime import datetime
import shutil

# --- 1. KONFIGURASI UTAMA ---
# Di sinilah Anda bisa mengatur semua parameter training.

class Config:
    # Path ke dataset Anda (folder yang berisi subfolder 'organik' dan 'anorganik')
    DATASET_DIR = "./dataset"
    
    # Pengaturan Model
    IMG_SIZE = 224      # Ukuran gambar (224x224 untuk MobileNetV2/EfficientNet)
    BATCH_SIZE = 32     # Jumlah gambar yang diproses dalam satu waktu
    
    # Pengaturan Arsitektur
    # Ganti menjadi False jika Anda ingin menggunakan MobileNetV2 (lebih cepat, akurasi sedikit lebih rendah)
    USE_EFFICIENTNET = True 
    DROPOUT_RATE = 0.4  # Seberapa agresif dropout untuk mencegah overfitting
    
    # Pengaturan Training
    INITIAL_EPOCHS = 30 # Jumlah epoch untuk melatih 'head' model saja
    FINE_TUNE_EPOCHS = 50 # Jumlah epoch tambahan untuk fine-tuning (dapat berhenti lebih cepat)
    INITIAL_LR = 1e-3   # Learning rate awal
    
    # Pengaturan Pembagian Data
    VALIDATION_SPLIT = 0.2 # 20% dari data akan digunakan untuk validasi

    # Pengaturan Output
    SAVE_DIR = "./models_output" # Folder untuk menyimpan semua hasil training

config = Config()

# Membuat folder output jika belum ada
os.makedirs(config.SAVE_DIR, exist_ok=True)

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# --- 2. PERSIAPAN DATASET ---

def prepare_dataset():
    """
    Secara otomatis membuat struktur folder train/validation dari dataset sumber.
    Ini adalah praktik terbaik untuk memastikan tidak ada data yang tumpang tindih.
    """
    source_dir = config.DATASET_DIR
    organized_dir = os.path.join(os.path.dirname(source_dir), "dataset_organized")

    # Hapus folder lama jika ada untuk memastikan kebersihan data
    if os.path.exists(organized_dir):
        print(f"Menghapus direktori lama: {organized_dir}")
        shutil.rmtree(organized_dir)

    print(f"Membuat struktur direktori baru di: {organized_dir}")
    
    # Buat ulang struktur folder
    train_dir = os.path.join(organized_dir, 'train')
    val_dir = os.path.join(organized_dir, 'validation')
    
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Pindahkan file
    for cls in classes:
        files = [f for f in os.listdir(os.path.join(source_dir, cls)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        np.random.shuffle(files)
        
        split_index = int(len(files) * (1 - config.VALIDATION_SPLIT))
        train_files = files[:split_index]
        val_files = files[split_index:]

        for f in train_files:
            shutil.copy(os.path.join(source_dir, cls, f), os.path.join(train_dir, cls, f))
        for f in val_files:
            shutil.copy(os.path.join(source_dir, cls, f), os.path.join(val_dir, cls, f))
            
    print("✅ Dataset berhasil diorganisir.")
    return organized_dir

# --- 3. AUGMENTASI DATA & DATA GENERATORS ---

def create_data_generators(data_dir):
    """
    Membuat data generator dengan augmentasi data yang agresif untuk training
    dan augmentasi minimal untuk validasi.
    """
    # Augmentasi data yang 'kaya' untuk membuat model lebih tangguh
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Untuk data validasi, kita hanya melakukan rescale
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary', # Penting untuk binary classification
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator

# --- 4. ARSITEKTUR MODEL ---

def build_model(num_classes=1):
    """
    Membangun model dengan arsitektur pre-trained dan custom head yang canggih.
    """
    if config.USE_EFFICIENTNET:
        base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
        print("🚀 Menggunakan arsitektur EfficientNetB4.")
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
        print("⚡ Menggunakan arsitektur MobileNetV2.")

    # Awalnya, semua layer di base model tidak bisa dilatih (frozen)
    base_model.trainable = False

    # Membuat 'head' atau bagian atas model yang akan kita latih
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(), # Menstabilkan proses learning
        layers.Dense(256, activation='relu'),
        layers.Dropout(config.DROPOUT_RATE), # Mencegah overfitting
        layers.Dense(num_classes, activation='sigmoid') # Output Sigmoid untuk klasifikasi biner
    ])
    
    return model

# --- 5. CALLBACKS & TRAINING STRATEGY ---

def get_callbacks(model_name):
    """
    Mendefinisikan callbacks canggih untuk mengontrol proses training.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config.SAVE_DIR, f"{model_name}_{timestamp}.h5")
    
    return [
        # Menyimpan hanya model terbaik berdasarkan akurasi validasi
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Menghentikan training jika tidak ada peningkatan setelah beberapa epoch
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10, # Berhenti setelah 10 epoch tanpa peningkatan val_loss
            restore_best_weights=True,
            verbose=1
        ),
        # Mengurangi learning rate jika training melambat
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            verbose=1
        )
    ], model_path

# --- 6. FUNGSI UTAMA UNTUK MENJALANKAN SEMUANYA ---

def main():
    print("\n--- Memulai Proses Training Model Akurasi Tinggi ---")
    
    # 1. Siapkan dataset
    organized_data_dir = prepare_dataset()
    
    # 2. Buat data generator
    train_gen, val_gen = create_data_generators(organized_data_dir)
    
    # 3. Bangun model
    model = build_model()
    
    # 4. Compile dan latih 'head' model terlebih dahulu
    print("\n--- TAHAP 1: Melatih Head Model (Feature Extraction) ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.INITIAL_LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks, _ = get_callbacks("waste_model_initial_head")
    
    history = model.fit(
        train_gen,
        epochs=config.INITIAL_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # 5. Lakukan Fine-Tuning
    print("\n--- TAHAP 2: Melatih Sebagian Layer (Fine-Tuning) ---")
    base_model = model.layers[0]
    base_model.trainable = True # 'Cairkan' base model
    
    # Kita hanya akan melatih beberapa layer terakhir saja
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.INITIAL_LR / 10),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_finetune, best_model_path = get_callbacks("high_accuracy_waste_classifier")
    
    history_fine = model.fit(
        train_gen,
        epochs=config.INITIAL_EPOCHS + config.FINE_TUNE_EPOCHS,
        initial_epoch=history.epoch[-1], # Lanjutkan dari epoch terakhir
        validation_data=val_gen,
        callbacks=callbacks_finetune
    )
    
    print("\n" + "="*60 + "\n🎉 TRAINING SELESAI!\n" + "="*60)
    
    # --- EVALUASI AKHIR ---
    print("\n--- Memulai Evaluasi Akhir pada Model Terbaik ---")
    # Memuat model terbaik yang disimpan oleh ModelCheckpoint
    best_model = keras.models.load_model(best_model_path)
    
    val_gen.reset()
    predictions = best_model.predict(val_gen, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = val_gen.classes
    class_names = list(val_gen.class_indices.keys())
    
    print("\n📊 Laporan Klasifikasi:\n" + "-"*40)
    print(classification_report(true_classes, predicted_classes, target_names=class_names, digits=4))
    
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Label Prediksi')
    plt.savefig(os.path.join(config.SAVE_DIR, 'confusion_matrix.png'))
    plt.show()

    print(f"💾 Model terbaik disimpan di: {best_model_path}")

if __name__ == "__main__":
    main()
