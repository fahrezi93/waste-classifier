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
MODEL_PATH = 'waste_model.h5'  # Model lokal
MODEL_URL = os.environ.get('MODEL_URL', '')  # URL untuk download model
CORRECTIONS_DIR = 'corrections' # Folder untuk menyimpan gambar koreksi

model = None
CLASS_NAMES = ['Anorganik', 'Organik']

def download_model():
    """Download model dari cloud storage jika belum ada."""
    if not os.path.exists(MODEL_PATH) and MODEL_URL:
        print(f"Downloading model from {MODEL_URL}...")
        import requests
        try:
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("Model downloaded successfully")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    return True

def load_model():
    """Memuat model Keras dari file .h5."""
    global model
    try:
        # Download model jika perlu
        if not download_model():
            raise Exception("Failed to download model")
            
        print("Memuat model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model '{MODEL_PATH}' berhasil dimuat.")
            
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return
        
    except Exception as e1:
        print(f"Error saat memuat model utama: {e1}")
        try:
            print("Mencoba memuat model alternatif...")
            model = tf.keras.models.load_model('waste_model.h5')
            print("Model alternatif 'waste_model.h5' berhasil dimuat.")
            return
        except Exception as e2:
            print(f"Gagal memuat kedua model: {e2}")
            raise e2

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
