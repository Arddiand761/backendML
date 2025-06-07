# Nama file: DeteksiAnomali.py
# Tujuan: Menyediakan fungsi untuk memuat model dan melakukan prediksi anomali.
# File ini akan diimpor oleh app.py

import joblib
import pandas as pd
import os

# --- Konfigurasi Path Aset ---
# Menentukan lokasi file model yang sudah dilatih
MODEL_PATH = "./models/model_anomali.joblib"
ENCODER_PATH = "./models/label_encoder.joblib"

# Variabel global untuk menyimpan model yang sudah dimuat
# Ini untuk efisiensi, agar tidak perlu load dari disk setiap ada request
model_pipeline = None
label_encoder = None

def _muat_aset():
    """Fungsi internal untuk memuat model dan encoder dari file."""
    global model_pipeline, label_encoder
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("Aset deteksi anomali (model dan encoder) berhasil dimuat.")
    except FileNotFoundError:
        print(f"ERROR: File model atau encoder tidak ditemukan di.")
        print("Pastikan Anda sudah menjalankan 'train_model.py' terlebih dahulu.")
        model_pipeline = None
        label_encoder = None

def dapatkan_status_aset():
    """
    Fungsi untuk health check. Mengecek apakah file model sudah ada.
    Ini dipanggil oleh route '/' di app.py.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        return "Model deteksi anomali SIAP."
    else:
        return f"Model deteksi anomali TIDAK DITEMUKAN ."

def proses_deteksi_anomali(data_input):
    """
    Fungsi utama untuk memproses data input dan mengembalikan prediksi.
    Ini dipanggil oleh route '/predict/anomaly' di app.py.
    """
    if not model_pipeline or not label_encoder:
        return {"error": "Model tidak tersedia atau gagal dimuat."}

    # Validasi input
    if not isinstance(data_input, list) or not all(isinstance(item, dict) for item in data_input):
        return {"error": "Input harus dalam format list of dictionaries."}
    
    try:
        # Mengubah input JSON (list of dicts) menjadi DataFrame
        df_input = pd.DataFrame(data_input)
        
        # Ekstrak fitur tanggal dari input
        # Pastikan input memiliki 'Tanggal_Transaksi'
        if 'Tanggal_Transaksi' not in df_input.columns:
            return {"error": "Kolom 'Tanggal_Transaksi' tidak ditemukan dalam input."}
        
        df_input['Tanggal_Transaksi'] = pd.to_datetime(df_input['Tanggal_Transaksi'])
        df_input['Tahun'] = df_input['Tanggal_Transaksi'].dt.year
        df_input['Bulan'] = df_input['Tanggal_Transaksi'].dt.month
        df_input['Hari'] = df_input['Tanggal_Transaksi'].dt.day

    except Exception as e:
        return {"error": f"Gagal memproses data input: {e}"}

    # Melakukan prediksi
    try:
        # Memilih kolom yang sesuai dengan yang digunakan saat training
        kolom_model = ['Jumlah_Pengeluaran', 'Tahun', 'Bulan', 'Hari', 'Deskripsi_Pengeluaran']
        df_prediksi = df_input[kolom_model]

        prediksi_encoded = model_pipeline.predict(df_prediksi)
        
        # Mengubah hasil prediksi dari angka (0/1) kembali ke label ('Normal'/'Anomali')
        prediksi_label = label_encoder.inverse_transform(prediksi_encoded)
        
        # Menambahkan hasil prediksi ke data asli untuk output
        df_input['Prediksi_Label'] = prediksi_label
        
        # Mengubah DataFrame hasil menjadi format JSON (list of dicts)
        hasil_akhir = df_input.to_dict(orient='records')
        
        return hasil_akhir

    except Exception as e:
        return {"error": f"Terjadi kesalahan saat prediksi: {e}"}

# --- PENTING ---
# Memuat model saat modul ini pertama kali diimpor oleh Flask
_muat_aset()

