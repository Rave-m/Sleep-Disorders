import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

st.title("Prediksi Gangguan Tidur")

st.write("Masukkan data berikut untuk memprediksi apakah seseorang mengalami gangguan tidur.")

# Input sesuai fitur model
bmi_category = st.selectbox("Kategori BMI",['Overweight', 'Normal', 'Obese', 'Normal Weight'], format_func=lambda x: f"Kategori {x}")
blood_pressure = st.slider("Tekanan Darah (encode)", 0, 24, 20)
occupation = st.selectbox("Pekerjaan", ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager'], format_func=lambda x: f"Kode {x}")
age = st.slider("Usia", 27, 67, 47)
physical_activity = st.slider("Aktivitas Fisik Harian", 30, 90, 60)
heart_rate = st.slider("Detak Jantung", 65, 86, 70)

# Load model dan scaler
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Prediksi
if st.button("Prediksi"):
    # Buat DataFrame dengan input terbaru
    input_data = pd.DataFrame([[
        bmi_category,
        blood_pressure,
        occupation,
        age,
        physical_activity,
        heart_rate
    ]], columns=[
        'BMI Category', 'Blood Pressure', 'Occupation', 'Age', 'Physical Activity Level', 'Heart Rate'
    ])
    
    # Definisikan mapping yang konsisten untuk kategori BMI
    bmi_mapping = {'Normal': 0, 'Normal Weight': 1, 'Obese': 2, 'Overweight': 3}
    input_data['BMI Category'] = input_data['BMI Category'].map(bmi_mapping)
    
    # Untuk occupation, kita gunakan dictionary mapping
    occupation_mapping = {
        'Software Engineer': 0, 'Doctor': 1, 'Sales Representative': 2,
        'Teacher': 3, 'Nurse': 4, 'Engineer': 5, 'Accountant': 6,
        'Scientist': 7, 'Lawyer': 8, 'Salesperson': 9, 'Manager': 10
    }
    input_data['Occupation'] = input_data['Occupation'].map(occupation_mapping)
    
    # Debug information
    st.write("Data sebelum scaling:")
    st.dataframe(input_data)
    try:
        # Pastikan semua fitur adalah numerik sebelum scaling
        numeric_input = input_data.astype(float)
        
        # Transform input data
        input_scaled = scaler.transform(numeric_input)
        
        # Tampilkan data sesudah scaling untuk debugging
        st.write("Data setelah scaling:")
        st.dataframe(pd.DataFrame(input_scaled, columns=input_data.columns))
        
        # Prediksi
        prediction = model.predict(input_scaled)
        
        # Mapping nilai prediksi ke jenis gangguan tidur
        disorder_mapping = {
            0: 'Insomnia',
            1: 'Sleep Apnea',
            2: 'None'  # Tidak ada gangguan tidur
        }
        
        # Mendapatkan jenis gangguan tidur berdasarkan hasil prediksi
        predicted_disorder = disorder_mapping.get(prediction[0], "Tidak diketahui")
        
        if predicted_disorder == 'None':
            st.success(f"Hasil prediksi: Tidak mengalami gangguan tidur")
        else:
            st.warning(f"Hasil prediksi: Mengalami gangguan tidur - {predicted_disorder}")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
