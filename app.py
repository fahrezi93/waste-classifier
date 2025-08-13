import os
import io
import uuid # Diperlukan untuk membuat nama file unik
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi CORS untuk production
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")
# Tambahkan domain Vercel Anda ke whitelist
ALLOWED_ORIGINS = [
    FRONTEND_ORIGIN,
    "https://waste-classifier.vercel.app",  # Sesuaikan dengan domain Vercel Anda
    "http://localhost:5173",  # Untuk development
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# --- Konfigurasi ---
# Model URLs dari Google Drive
MODEL_URLS = {
    'waste_model.h5': 'https://drive.google.com/uc?export=download&id=1-GxeQlOLMCVp9d9Qj3Cg2K-vRz2eoN-E',
    'waste_model_trained.h5': 'https://drive.google.com/uc?export=download&id=1-8kMqXxfQgMsxJVvOXN2CM53_gTYwNHX',
    'waste_model_initial_head.h5': 'https://drive.google.com/uc?export=download&id=1-3zVJDDnFLReUBEp9IB2qbvRQcybXQrN'
}

# Daftar model untuk dicoba, dalam urutan prioritas
MODELS = [
    'waste_model.h5',
    'waste_model_trained.h5',
    'waste_model_initial_head.h5'
]
CORRECTIONS_DIR = 'corrections' # Folder untuk menyimpan gambar koreksi

model = None
CLASS_NAMES = ['Anorganik', 'Organik']

def download_model():
    """Download model dari Google Drive jika file tidak ada atau hanya pointer Git LFS."""
    import requests
    
    POINTER_FILE_THRESHOLD_BYTES = 2000  # 2 KB, ambang batas untuk pointer

    for model_name, url in MODEL_URLS.items():
        should_download = False
        if not os.path.exists(model_name):
            should_download = True
            print(f"File model '{model_name}' tidak ditemukan.")
        else:
            file_size = os.path.getsize(model_name)
            if file_size < POINTER_FILE_THRESHOLD_BYTES:
                should_download = True
                print(f"File model '{model_name}' ada tapi terlalu kecil ({file_size} bytes). Kemungkinan ini adalah pointer Git LFS.")
            else:
                print(f"File model '{model_name}' sudah ada dengan ukuran {file_size} bytes.")

        if should_download:
            print(f"Mengunduh {model_name} dari Google Drive...")
            try:
                # Gunakan session untuk penanganan koneksi yang lebih baik
                with requests.Session() as s:
                    response = s.get(url, stream=True)
                    response.raise_for_status()  # Cek jika ada error HTTP

                    with open(model_name, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                print(f"Model {model_name} berhasil diunduh. Ukuran baru: {os.path.getsize(model_name)} bytes.")
            except requests.exceptions.RequestException as e:
                print(f"Error saat mengunduh {model_name}: {e}")
    return True

def load_model():
    """Memuat model Keras dari file .h5."""
    global model
    
    # Coba download dulu jika ada URL
    download_model()
    
    # Debug info
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    
    # Coba load dari daftar model yang tersedia
    for model_path in MODELS:
        try:
            print(f"\nMencoba memuat model dari {model_path}...")
            if os.path.exists(model_path):
                print(f"File exists. Size: {os.path.getsize(model_path)} bytes")
                with open(model_path, 'rb') as f:
                    header = f.read(10)
                print(f"File header (first 10 bytes): {header}")
                
                model = tf.keras.models.load_model(model_path, compile=False)
                print(f"Model '{model_path}' berhasil dimuat.")
                
                # Compile model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy', 
                    metrics=['accuracy']
                )
                return
            else:
                print(f"File tidak ditemukan: {model_path}")
                print("Full path:", os.path.abspath(model_path))
        except Exception as e:
            print(f"Error saat memuat {model_path}: {e}")
            print("Full error:", str(e))
            continue
    
    raise Exception("FATAL ERROR: Tidak ada model yang berhasil dimuat")

def preprocess_image(image_bytes, target_size=(224, 224)):
    """Fungsi untuk memproses gambar sebelum dimasukkan ke model."""
    try:
        # Buka gambar dan konversi ke RGB
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize dengan mempertahankan aspek ratio dan interpolasi yang bagus
        width, height = img.size
        new_width = target_size[0]
        new_height = int(height * (new_width / width))
        
        if new_height > target_size[1]:
            new_height = target_size[1]
            new_width = int(width * (new_height / height))
            
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Buat canvas hitam dengan ukuran target (sesuai EfficientNet)
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        
        # Paste gambar yang di-resize ke tengah canvas
        offset = ((target_size[0] - new_width) // 2,
                 (target_size[1] - new_height) // 2)
        new_img.paste(img, offset)
        
        # Konversi ke array dan preprocess untuk EfficientNet
        img_array = np.array(new_img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error saat memproses gambar: {e}")
        return None

load_model()
# Membuat folder koreksi jika belum ada
os.makedirs(os.path.join(CORRECTIONS_DIR, 'organik'), exist_ok=True)
os.makedirs(os.path.join(CORRECTIONS_DIR, 'anorganik'), exist_ok=True)


@app.route('/')
def index():
    return "Server Flask untuk klasifikasi sampah berjalan!"

@app.route('/predict', methods=['POST'])
def predict():
    # ... (kode endpoint ini tidak berubah)
    print("Menerima request di endpoint /predict...")
    if model is None: return jsonify({'error': 'Model tidak berhasil dimuat.'}), 500
    if 'file' not in request.files: return jsonify({'error': 'Request tidak berisi file.'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'Tidak ada file yang dipilih.'}), 400
    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        if processed_image is None: return jsonify({'error': 'Gagal memproses file gambar.'}), 400
        prediction = model.predict(processed_image)
        confidence = float(prediction[0][0])
        if confidence > 0.5:
            predicted_class_name = CLASS_NAMES[1]
            final_confidence = confidence
        else:
            predicted_class_name = CLASS_NAMES[0]
            final_confidence = 1 - confidence
        print(f"Prediksi berhasil: {predicted_class_name} dengan confidence {final_confidence}")
        return jsonify({'prediction': predicted_class_name, 'confidence': round(final_confidence, 4)})
    except Exception as e:
        print(f"Terjadi error saat prediksi: {e}")
        return jsonify({'error': f'Terjadi kesalahan di server: {str(e)}'}), 500

# --- ENDPOINT BARU UNTUK KOREKSI PENGGUNA ---
@app.route('/correct-prediction', methods=['POST'])
def correct_prediction():
    """Menerima gambar dan label koreksi dari pengguna."""
    if 'file' not in request.files or 'correct_label' not in request.form:
        return jsonify({'error': 'Data tidak lengkap.'}), 400

    file = request.files['file']
    correct_label = request.form['correct_label'].lower() # 'organik' atau 'anorganik'

    if correct_label not in ['organik', 'anorganik']:
        return jsonify({'error': 'Label tidak valid.'}), 400

    try:
        # Buat nama file yang unik untuk menghindari tumpang tindih
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Tentukan path penyimpanan
        save_path = os.path.join(CORRECTIONS_DIR, correct_label, unique_filename)
        
        # Simpan file
        file.seek(0) # Kembali ke awal file
        file.save(save_path)
        
        print(f"Koreksi diterima: '{file.filename}' disimpan sebagai '{unique_filename}' di folder '{correct_label}'")
        return jsonify({'message': 'Terima kasih atas masukan Anda!'})

    except Exception as e:
        print(f"Gagal menyimpan koreksi: {e}")
        return jsonify({'error': 'Gagal menyimpan file koreksi.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
