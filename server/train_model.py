import tensorflow as tf
import os

# --- 1. Konfigurasi dan Parameter ---

# Path ke dataset Anda
DATASET_DIR = 'dataset'

# Parameter Gambar
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # Jumlah gambar yang diproses dalam satu waktu

# Parameter Training
EPOCHS = 15 # Berapa kali model akan "melihat" keseluruhan dataset
LEARNING_RATE = 0.0001 # Seberapa cepat model belajar

# --- 2. Memuat dan Memproses Data ---

print("Mempersiapkan dataset...")

# Menggunakan image_dataset_from_directory untuk memuat gambar dari folder.
# Ini adalah cara modern dan efisien.
# TensorFlow akan otomatis memberi label berdasarkan nama folder (0 untuk anorganik, 1 untuk organik)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,  # 20% data untuk validasi
    subset="training",
    seed=123, # Seed untuk memastikan split data konsisten
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Menampilkan nama kelas yang ditemukan
class_names = train_dataset.class_names
print(f"Kelas yang ditemukan: {class_names}") # Harusnya ['anorganik', 'organik']

# Optimasi performa dataset
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Augmentasi Data ---
# Membuat variasi gambar (rotasi, zoom, dll) untuk membuat model lebih tangguh

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# --- 4. Membangun Model (Fine-Tuning) ---

print("Membangun model...")

# Memuat MobileNetV2 sebagai model dasar, tanpa layer klasifikasi atasnya.
# Bobotnya sudah dilatih pada dataset ImageNet.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Awalnya, kita "bekukan" semua layer di model dasar agar tidak ikut terlatih.
base_model.trainable = False

# Membuat model baru di atas model dasar
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs) # Terapkan augmentasi
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Pre-processing khusus MobileNetV2
x = base_model(x, training=False) # Jalankan base model dalam mode inferensi
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x) # Dropout untuk mengurangi overfitting
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x) # Layer output untuk klasifikasi biner

model = tf.keras.Model(inputs, outputs)

# --- 5. Compile Model ---

print("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# --- 6. Melatih Model (Tahap Awal) ---

print("\n--- Memulai Training Tahap Awal (Feature Extraction) ---")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# --- 7. Fine-Tuning (Melatih Sebagian Layer Model Dasar) ---

print("\n--- Memulai Training Tahap Lanjut (Fine-Tuning) ---")

# "Cairkan" beberapa layer teratas dari model dasar agar bisa ikut belajar
base_model.trainable = True
fine_tune_at = 100 # Mulai melatih dari layer ke-100

# Bekukan semua layer sebelum `fine_tune_at`
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile ulang model dengan learning rate yang lebih kecil untuk fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# Lanjutkan training untuk beberapa epoch lagi
fine_tune_epochs = 10
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Lanjutkan dari epoch terakhir
    validation_data=validation_dataset
)

# --- 8. Simpan Model ---

print("\nTraining selesai. Menyimpan model...")
model.save('waste_model_trained.h5')
print("Model berhasil disimpan sebagai 'waste_model_trained.h5'")
