import tensorflow as tf

def create_and_save_placeholder_model(path='waste_model.h5'):
    """
    Membuat model klasifikasi biner sederhana menggunakan MobileNetV2 sebagai basis
    dan menyimpannya sebagai file .h5. Model ini adalah placeholder.
    
    Arsitektur:
    1. MobileNetV2 (pre-trained on ImageNet, tanpa layer atas)
    2. GlobalAveragePooling2D untuk meratakan fitur
    3. Dense layer dengan 1 output dan aktivasi sigmoid untuk klasifikasi biner.
    """
    print("Membuat model placeholder...")
    
    # Muat MobileNetV2 sebagai base model, tanpa layer klasifikasi atas (include_top=False)
    # weights='imagenet' berarti kita menggunakan bobot yang sudah dilatih di ImageNet.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # Bekukan (freeze) layer dari base model agar bobotnya tidak berubah saat fine-tuning awal.
    base_model.trainable = False

    # Bangun model sekuensial di atas base model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), # Mengubah output fitur menjadi vektor tunggal
        tf.keras.layers.Dropout(0.2), # Dropout untuk mengurangi overfitting
        tf.keras.layers.Dense(1, activation='sigmoid') # Output tunggal dengan sigmoid untuk klasifikasi biner
    ])

    # Compile model dengan optimizer, loss function, dan metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Simpan model ke path yang ditentukan
    try:
        model.save(path)
        print(f"Model placeholder berhasil dibuat dan disimpan di '{path}'")
        print("\nRingkasan Arsitektur Model:")
        model.summary()
    except Exception as e:
        print(f"Gagal menyimpan model: {e}")

if __name__ == '__main__':
    create_and_save_placeholder_model()
