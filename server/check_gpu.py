import tensorflow as tf
import sys

print(f"TF Version: {tf.__version__}")
print(f"Python Version: {sys.version}")

try:
    # Ini adalah cara yang benar untuk memeriksa perangkat GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"🎉 Selamat! TensorFlow dapat mendeteksi {len(gpus)} GPU Anda:")
        # Loop melalui semua GPU yang terdeteksi dan cetak detailnya
        for i, gpu in enumerate(gpus):
            print(f"--- GPU {i} ---")
            tf.config.experimental.set_memory_growth(gpu, True)
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  Nama: {details.get('device_name', 'N/A')}")
            print(f"  Compute Capability: {details.get('compute_capability', 'N/A')}")
    else:
        print("❌ TensorFlow tidak dapat mendeteksi GPU.")
        print("Pastikan Anda telah menginstal driver NVIDIA, CUDA Toolkit, dan cuDNN dengan benar.")
        print("Dan pastikan Anda menjalankan skrip ini di dalam virtual environment yang benar.")

except Exception as e:
    print(f"Terjadi error saat memeriksa GPU: {e}")