import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv 
import Prediksi  # Modul Anda yang sudah ada
import Kategorisasi # Modul baru untuk kategorisasi
import Anomali

load_dotenv()
app = Flask(__name__)

EXPECTED_API_KEY = os.environ.get("PYTHON_API_KEY")

if not EXPECTED_API_KEY:
    print("PERINGATAN: Environment variable PYTHON_API_KEY tidak di-set. API akan berjalan TANPA otentikasi API Key.")

@app.before_request
def check_api_key():
    if request.endpoint == 'static': 
        return

    if EXPECTED_API_KEY: 
        api_key_header = request.headers.get("X-API-Key")
        if not api_key_header or api_key_header != EXPECTED_API_KEY:
            return jsonify({"error": "API Key tidak valid atau tidak disertakan."}), 401

@app.route('/')
def home():
    # Health check untuk kedua model
    status_model_prediksi = Prediksi.dapatkan_status_aset() if hasattr(Prediksi, 'dapatkan_status_aset') else "Modul tidak ditemukan"
    status_model_kategorisasi = Kategorisasi.dapatkan_status_model_kategorisasi()

    return jsonify({
        "message": "Selamat Datang di API Keuangan",
        "status_model_prediksi_anomali": status_model_prediksi,
        "status_model_kategorisasi": status_model_kategorisasi
    })

@app.route('/predict/keuangan', methods=['POST'])

def predict_keuangan_route():

    if not request.is_json:

        return jsonify({"error": "Request harus dalam format JSON"}), 400
    try:
        data_input = request.get_json()
        hasil_prediksi = Prediksi.proses_prediksi(data_input)
        return jsonify({"prediksi_keuangan": hasil_prediksi})
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Input data error: {str(e)}"}), 400
    except Exception as e:
        print(f"Error di /predict/keuangan: {str(e)}") # Logging error ke konsol server
        return jsonify({"error": f"Inputan salah,Input harus berbentuk json."}), 500 
    
    
@app.route('/predict/anomaly', methods=['POST']) # Mengubah nama rute menjadi lebih spesifik
def predict_anomaly_route():
    if not request.is_json:
        return jsonify({"error": "Request harus dalam format JSON"}), 400
    
    try:
        data_input = request.get_json()
        hasil_prediksi = Anomali.proses_deteksi_anomali(data_input) 
        if "error" in hasil_prediksi:
            return jsonify(hasil_prediksi), 500
        return jsonify({"anomaly_detection_result": hasil_prediksi})
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Input data error: {str(e)}"}), 400
    except Exception as e:
        print(f"Error di /predict/anomaly: {str(e)}")
        return jsonify({"error": "Terjadi kesalahan internal saat prediksi anomali."}), 500


# --- Rute BARU untuk Kategorisasi ---
@app.route('/categorize/transaction', methods=['POST'])
def categorize_transaction_route():
    if not request.is_json:
        return jsonify({"error": "Request harus dalam format JSON"}), 400
    
    try:
        data_input = request.get_json()
        # Panggil fungsi dari modul Kategorisasi
        hasil_kategori = Kategorisasi.kategorisasi_otomatis(data_input)
        return jsonify(hasil_kategori)
    except (KeyError, TypeError, ValueError) as e:
        # Menangkap error validasi dari modul Kategorisasi
        return jsonify({"error": f"Input data tidak valid: {str(e)}"}), 400
    except Exception as e:
        print(f"Error di /categorize/transaction: {str(e)}")
        return jsonify({"error": "Terjadi kesalahan internal saat melakukan kategorisasi."}), 500

if __name__ == '__main__':
      # Port diambil dari environment variable PORT yang disediakan Render
    port = int(os.environ.get("PORT", 5001))
    # Host harus 0.0.0.0 agar bisa diakses dari luar
    app.run(host="0.0.0.0", port=port)

