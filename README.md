# 🏥 WDBC Breast Cancer Prediction App

Aplikasi web untuk prediksi kanker payudara menggunakan model machine learning berdasarkan dataset Wisconsin Diagnostic Breast Cancer (WDBC).

## 🚀 Quick Start

### Deploy ke Streamlit Cloud (Paling Mudah)

1. **Fork atau clone repository ini**
2. **Upload ke GitHub**
3. **Buka [share.streamlit.io](https://share.streamlit.io)**
4. **Connect dengan repository GitHub Anda**
5. **Deploy otomatis**

### Deploy Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run streamlit_app.py
```

## 📋 Requirements

File yang diperlukan:
- `streamlit_app.py` - Aplikasi utama
- `requirements.txt` - Dependencies Python
- `models/` - Folder berisi model yang sudah dilatih
  - `best_model_support_vector_machine.pkl`
  - `scaler.pkl`
  - `label_encoder.pkl`
  - `model_info.json`
- `wdbc_clean_no_outliers.csv` - Data untuk analisis (opsional)

## 🔧 Troubleshooting

### Error: psycopg2 build failed
**Solusi:** Requirements.txt sudah diperbaiki dengan menghapus package yang tidak diperlukan.

### Error: Model not loaded
**Solusi:** Pastikan semua file model ada di folder `models/`

### Error: Data file not found
**Solusi:** File `wdbc_clean_no_outliers.csv` opsional, tab Data Analysis mungkin tidak berfungsi.

## 📊 Fitur Aplikasi

1. **🔮 Prediction Tab**
   - Input form untuk 30 fitur sel nucleus
   - Prediksi diagnosis (Benign/Malignant)
   - Visualisasi probabilitas
   - Analisis fitur penting

2. **📈 Data Analysis Tab**
   - Overview dataset
   - Distribusi diagnosis
   - Statistik deskriptif
   - Heatmap korelasi

3. **📊 Model Performance Tab**
   - Metrik performa model
   - Visualisasi performa
   - Informasi model

4. **📁 Upload Data Tab**
   - Upload file CSV untuk batch prediction
   - Download hasil prediksi
   - Statistik batch prediction

## 🌐 Deployment Options

### 1. Streamlit Cloud (Gratis)
- Upload ke GitHub
- Deploy otomatis di share.streamlit.io
- URL publik otomatis

### 2. Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
git add .
git commit -m "Initial commit"
git push heroku main
```

### 3. Google Cloud Platform
- Buat project di GCP
- Enable Cloud Run
- Deploy dengan Docker

### 4. AWS
- Buat EC2 instance
- Install dependencies
- Setup nginx sebagai reverse proxy

## 📱 Mobile Responsive
Aplikasi sudah dioptimasi untuk mobile dengan:
- Layout responsive
- Touch-friendly interface
- Optimized for small screens

## 🔒 Security
- Medical disclaimer included
- Input validation
- Error handling
- Secure file upload

## 📄 License
Aplikasi ini dibuat untuk tujuan edukasi. Untuk penggunaan medis, konsultasikan dengan profesional kesehatan.

## 🆘 Support
Jika ada masalah:
1. Check error logs
2. Verify file structure
3. Test dengan data sample
4. Check dependencies versions
