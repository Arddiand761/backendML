
import numpy as np
import tensorflow as tf
import joblib # Untuk memuat scaler


MODEL_PATH = './models/model_pengeluaran.h5' 
SCALER_PATH = './models/scaler_pengeluaran.pkl' 
LOOK_BACK = 3 

model = None
scaler = None


def muat_model_dan_scaler():
    global model, scaler 
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("* [Prediksi.py] Model dan scaler berhasil dimuat.")
    except Exception as e:
        print(f"* [Prediksi.py] GAGAL memuat model atau scaler: {e}")
        model = None # Set ke None jika gagal
        scaler = None # Set ke None jika gagal

muat_model_dan_scaler()


def proses_prediksi(data_input_json):
    """
    Fungsi ini menerima data input JSON (sebuah dictionary),
    melakukan validasi dasar, pra-pemrosesan, prediksi, dan pasca-pemrosesan.
    """
    if model is None or scaler is None:
        raise RuntimeError("Model atau scaler tidak berhasil dimuat. Periksa log startup Prediksi.py.")

    try:
        previous_expenses = data_input_json['previous_expenses']
        if not isinstance(previous_expenses, list) or len(previous_expenses) != LOOK_BACK:
            raise ValueError(f"Input 'previous_expenses' harus berupa list dengan {LOOK_BACK} angka.")
        if not all(isinstance(num, (int, float)) for num in previous_expenses):
            raise ValueError("Semua item dalam 'previous_expenses' harus berupa angka.")
    except KeyError:

        raise KeyError("Key 'previous_expenses' tidak ditemukan dalam input JSON.")
    except ValueError as ve:
 
        raise ve

    input_array = np.array(previous_expenses, dtype=np.float32).reshape(-1, 1)


    scaled_input = scaler.transform(input_array)


    reshaped_input = scaled_input.reshape((1, LOOK_BACK, 1))


    scaled_prediction = model.predict(reshaped_input)


    actual_prediction = scaler.inverse_transform(scaled_prediction)
    predicted_value = float(actual_prediction[0, 0])

    return round(predicted_value, 2)