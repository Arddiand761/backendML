# File: Kategorisasi.py

import os
import numpy as np
import tensorflow as tf
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler # Impor StandardScaler


# --- Variabel Global dan Konfigurasi ---
# Pastikan semua file ini berada di direktori yang sama dengan skrip Anda
# atau gunakan path absolut.
MODEL_PATH = './models/best_transaction_classifier.h5'
TOKENIZER_PATH = './models/tokenizer.pickle'
SCALER_PATH = './models/scaler.pickle'
LABEL_ENCODER_PATH = './models/label_encoder.pickle'
MAX_LEN = 50  # HARUS SAMA dengan yang digunakan saat training

# Cache untuk aset yang sudah dimuat
_model = None
_tokenizer = None
_scaler = None
_label_encoder = None
_indonesian_stopwords = None
_muat_aset_gagal = False

# --- Fungsi Pemuatan Aset ---
def _muat_aset_kategorisasi():
    """
    Memuat semua aset yang dibutuhkan: model, tokenizer, scaler, dan label encoder.
    Dijalankan sekali saat modul diimpor.
    """
    global _model, _tokenizer, _scaler, _label_encoder, _indonesian_stopwords, _muat_aset_gagal

    if _muat_aset_gagal:
        return

    try:
        # Unduh stopwords jika belum ada (diperlukan untuk pra-pemrosesan)
        try:
            _indonesian_stopwords = stopwords.words('indonesian')
        except LookupError:
            print("* [Kategorisasi.py] Mengunduh stopwords NLTK...")
            nltk.download('stopwords')
            nltk.download('punkt')
            _indonesian_stopwords = stopwords.words('indonesian')

        # Memuat model
        print(f"* [Kategorisasi.py] Memuat model dari: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)

        # Memuat tokenizer
        print(f"* [Kategorisasi.py] Memuat tokenizer dari: {TOKENIZER_PATH}")
        with open(TOKENIZER_PATH, 'rb') as handle:
            _tokenizer = pickle.load(handle)

        # Memuat scaler
        print(f"* [Kategorisasi.py] Memuat scaler dari: {SCALER_PATH}")
        with open(SCALER_PATH, 'rb') as handle:
            _scaler = pickle.load(handle)

        # Memuat label encoder
        print(f"* [Kategorisasi.py] Memuat label encoder dari: {LABEL_ENCODER_PATH}")
        with open(LABEL_ENCODER_PATH, 'rb') as handle:
            _label_encoder = pickle.load(handle)

        print("* [Kategorisasi.py] Semua aset untuk kategorisasi berhasil dimuat.")

    except Exception as e:
        print(f"* [Kategorisasi.py] GAGAL memuat aset kategorisasi: {e}")
        _muat_aset_gagal = True

# Panggil fungsi pemuatan saat modul diimpor pertama kali
_muat_aset_kategorisasi()


def dapatkan_status_model_kategorisasi():
    """Mengembalikan status kesiapan model untuk health check."""
    if _muat_aset_gagal:
        return "Gagal Dimuat"
    if _model and _tokenizer and _scaler and _label_encoder:
        return "Siap"
    return "Sedang Memuat"

# --- Fungsi Pra-pemrosesan Teks ---
def _preprocess_text(text):
    """Membersihkan dan memproses teks input."""
    text = str(text).lower()
    text = re.sub(r'\\d+', '', text)
    text = re.sub(r'[^\\w\\s]', '', text)
    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in _indonesian_stopwords]
    return " ".join(tokens)


def kategorisasi_otomatis(data_input_json):
    """
    Fungsi utama untuk memprediksi kategori dari deskripsi dan jumlah transaksi.
    Menggunakan metode yang lebih aman untuk scaler untuk menghindari masalah nama fitur.
    """
    if _muat_aset_gagal:
        raise RuntimeError("Aset untuk kategorisasi gagal dimuat. Periksa log server.")

    # 1. Validasi input
    try:
        description = data_input_json['description']
        amount = float(data_input_json['amount'])
    except KeyError:
        raise KeyError("Input JSON harus memiliki key 'description' dan 'amount'.")
    except (ValueError, TypeError):
        raise ValueError("'amount' harus berupa angka.")

    # 2. Pra-pemrosesan input teks
    clean_desc = _preprocess_text(description)
    sequence = _tokenizer.texts_to_sequences([clean_desc])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    # 3. Pra-pemrosesan input numerik (PERBAIKAN PALING AMAN)
    # Membuat instance scaler baru dan menyalin parameter (mean dan scale) dari
    # scaler yang dimuat. Ini menghindari peringatan/error nama fitur dari scikit-learn.
    temp_scaler = StandardScaler()
    temp_scaler.mean_ = _scaler.mean_
    temp_scaler.scale_ = _scaler.scale_
    scaled_amount = temp_scaler.transform(np.array([[amount]]))

    # 4. Prediksi menggunakan model
    prediction_probabilities = _model.predict([padded_sequence, scaled_amount])

    # 5. Mendapatkan kelas prediksi dan kepercayaannya
    predicted_class_index = np.argmax(prediction_probabilities, axis=1)[0]
    confidence = float(prediction_probabilities[0][predicted_class_index])

    # 6. Mengubah indeks kelas kembali ke nama kategori asli
    predicted_category = _label_encoder.inverse_transform([predicted_class_index])[0]

    return {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 4)
    }

# Contoh penggunaan modul jika dijalankan langsung
if __name__ == '__main__':
    if not _muat_aset_gagal:
        print("\\n--- Tes Modul Kategorisasi.py ---")
        test_data = {"description": "Makan siang nasi padang sederhana", "amount": 22000}
        hasil = kategorisasi_otomatis(test_data)
        print(f"Input: {test_data}")
        print(f"Output: {hasil}")
