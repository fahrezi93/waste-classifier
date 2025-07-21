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

# Memberi izin kepada frontend Anda untuk mengakses semua endpoint
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# --- Konfigurasi ---
MODEL_PATH = 'waste_model_trained.h5'
CORRECTIONS_DIR = 'corrections' # Folder untuk menyimpan gambar koreksi

model = None
CLASS_NAMES = ['Anorganik', 'Organik'] 

def load_model():
    """Memuat model Keras dari file .h5."""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Model '{MODEL_PATH}' berhasil dimuat.")
    except Exception as e:
        print(f"Error saat memuat model: {e}")

def preprocess_image(image_bytes, target_size=(224, 224)):
    """Fungsi untuk memproses gambar sebelum dimasukkan ke model."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0
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
