# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Penyakit Hati",
    page_icon="🩺",
    layout="wide"
)

# --- FUNGSI-FUNGSI UTAMA ---

@st.cache_data
def load_and_preprocess_data():
    """Memuat dan melakukan pra-pemrosesan awal pada dataset dari file lokal."""
    data_path = "Indian Liver Patient Dataset (ILPD).csv"
    try:
        df = pd.read_csv(data_path, header=None)
        df.columns = [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", 
            "Alkaline_Phosphotase", "Alamine_Aminotransferase", 
            "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", 
            "Albumin_and_Globulin_Ratio", "Selector"
        ]
        # Mengisi missing values di kolom A/G Ratio dengan median
        df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)
        # Mengubah label target (Selector): 1 -> 1 (sakit), 2 -> 0 (tidak sakit)
        df['Selector'] = df['Selector'].apply(lambda x: 1 if x == 1 else 0)
        # Label Encoding untuk kolom Gender
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari file lokal. Error: {e}")
        return None

def show_introduction():
    """Menampilkan halaman pendahuluan."""
    st.title("UAS Penambangan Data: Analisis Penyakit Hati")
    st.markdown("""
    **Nama:** Intan Aulia Majid  
    **NIM:** 230411100001
    """)
    st.header("Dataset Pasien Hati India (Indian Liver Patient Dataset - ILPD)")
    st.write("""
    Dataset ini digunakan untuk membangun model klasifikasi yang mampu memprediksi apakah seorang pasien menderita penyakit hati (liver disease) atau tidak, berdasarkan serangkaian parameter medis. Tujuannya adalah untuk membantu diagnosis dini dan mendukung pengembangan sistem cerdas di bidang kesehatan.
    """)
    st.info("Sumber Data: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)")

def show_eda(df):
    """Menampilkan halaman Exploratory Data Analysis (EDA)."""
    st.title("📊 Analisis & Visualisasi Data")
    st.subheader("Tampilan Awal Data")
    st.dataframe(df.head())
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

    st.subheader("Visualisasi Data")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='Selector', palette='pastel', ax=ax1)
        ax1.set_title('Distribusi Kelas (0: Tidak Sakit, 1: Sakit Hati)')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Tidak Sakit', 'Sakit Hati'])
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax2)
        ax2.set_title("Heatmap Korelasi Fitur")
        st.pyplot(fig2)

def show_modeling_and_evaluation(X_train_scaled, X_test_scaled, y_train, y_test, X_columns):
    """Menampilkan halaman pemodelan dan evaluasi."""
    st.title("🧠 Pemodelan & Evaluasi")
    st.sidebar.header("Opsi Model")
    model_choice = st.sidebar.selectbox(
        "Pilih model untuk dievaluasi:",
        ("K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest + SMOTE")
    )
    st.header(f"Hasil Evaluasi: {model_choice}")

    model = None
    if model_choice == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif model_choice == "Random Forest + SMOTE":
        st.write("**Catatan**: Model ini menggunakan teknik SMOTE untuk menyeimbangkan data latih.")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        st.write(f"Distribusi kelas sebelum SMOTE: {Counter(y_train)}")
        st.write(f"Distribusi kelas setelah SMOTE: {Counter(y_train_resampled)}")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)

    if model:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Tidak Sakit', 'Sakit Hati'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Akurasi Model")
            st.metric("Akurasi", f"{accuracy:.2%}")
            st.subheader("Classification Report")
            st.table(pd.DataFrame(report).transpose())
        with col2:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Prediksi Tidak Sakit', 'Prediksi Sakit'],
                        yticklabels=['Aktual Tidak Sakit', 'Aktual Sakit'])
            ax.set_ylabel('Aktual')
            ax.set_xlabel('Prediksi')
            st.pyplot(fig)

def show_conclusion(X_train_scaled, X_test_scaled, y_train, y_test):
    """Menampilkan halaman kesimpulan."""
    st.title("📊 Perbandingan & Kesimpulan")

    with st.spinner("Menghitung ulang akurasi model untuk perbandingan..."):
        # 1. KNN
        knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
        acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
        # 2. Decision Tree
        dt = DecisionTreeClassifier(random_state=42).fit(X_train_scaled, y_train)
        acc_dt = accuracy_score(y_test, dt.predict(X_test_scaled))
        # 3. Random Forest + SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        rf_smote = RandomForestClassifier(random_state=42).fit(X_train_res, y_train_res)
        acc_rf_smote = accuracy_score(y_test, rf_smote.predict(X_test_scaled))

    df_akurasi = pd.DataFrame({
        'Model': ['KNN', 'Decision Tree', 'Random Forest + SMOTE'],
        'Akurasi': [acc_knn, acc_dt, acc_rf_smote]
    }).sort_values(by='Akurasi', ascending=False)
    
    st.subheader("Tabel Perbandingan Akurasi")
    st.table(df_akurasi)
    
    st.subheader("Visualisasi Perbandingan")
    fig, ax = plt.subplots()
    sns.barplot(data=df_akurasi, x='Akurasi', y='Model', palette='viridis', ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title('Perbandingan Akurasi Model')
    st.pyplot(fig)
    
    st.header("Kesimpulan")
    st.success(f"**Model Terbaik: {df_akurasi.iloc[0]['Model']}** dengan akurasi **{df_akurasi.iloc[0]['Akurasi']:.2%}**.")
    st.write("""
    Setelah dilakukan serangkaian percobaan, Random Forest yang dikombinasikan dengan SMOTE memberikan hasil terbaik. 
    Meskipun perbedaan akurasinya tidak terlalu besar dibandingkan model lain, teknik SMOTE membantu model dalam mengenali kelas minoritas (pasien sakit hati) dengan lebih baik, yang sangat penting dalam konteks medis. 
    Dengan hasil ini, model ini dapat dijadikan dasar untuk pengembangan lebih lanjut.
    """)

# --- MAIN APP LOGIC ---

def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""
    st.sidebar.title("Navigasi Aplikasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ("Pendahuluan Proyek", "Analisis & Visualisasi Data", "Pra-Pemrosesan & Pemodelan", "Kesimpulan")
    )

    # Memuat data secara otomatis
    df = load_and_preprocess_data()
    
    if df is not None:
        # Persiapan data dilakukan di sini agar variabel tersedia untuk semua halaman
        X = df.drop('Selector', axis=1)
        y = df['Selector']
        X_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Perutean halaman
        if page == "Pendahuluan Proyek":
            show_introduction()
        elif page == "Analisis & Visualisasi Data":
            show_eda(df)
        elif page == "Pra-Pemrosesan & Pemodelan":
            show_modeling_and_evaluation(X_train_scaled, X_test_scaled, y_train, y_test, X_columns)
        elif page == "Kesimpulan":
            show_conclusion(X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        st.error("Gagal memuat data. Mohon periksa kembali koneksi atau URL data.")

if __name__ == "__main__":
    main()
