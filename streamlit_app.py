# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Kanker Payudara",
    page_icon="🔬",
    layout="wide"
)

st.set_option('deprecation.showPyplotGlobalUse', False)

# --- FUNGSI-FUNGSI ---

@st.cache_data
def load_data():
    """Memuat dan memberi nama kolom pada dataset WDBC."""
    column_names = [
        'ID', 'Diagnosis',
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    # Anda perlu memastikan file 'wdbc.data' berada di direktori yang sama dengan streamlit_app.py
    # atau menyediakan path lengkap ke file tersebut.
    try:
        df = pd.read_csv('wdbc.data', header=None, names=column_names)
        return df
    except FileNotFoundError:
        st.error("File 'wdbc.data' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama.")
        return None

@st.cache_data
def remove_outliers(_df):
    """Mendeteksi dan menghapus outlier menggunakan metode IQR."""
    df_clean = _df.copy()
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.drop('ID')
    
    for col in numeric_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

# --- Navigasi Sidebar ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ("🔬 Pendahuluan", "📊 Eksplorasi Data", "⚙️ Pra-Pemrosesan Data", "🧠 Pemodelan & Evaluasi", "💡 Prediksi Interaktif")
)

# Muat data awal
df_raw = load_data()
if df_raw is None:
    st.stop()
    
# Proses data (outlier removal)
df_clean = remove_outliers(df_raw)

# Pra-pemrosesan umum untuk model
X = df_clean.drop(['ID', 'Diagnosis'], axis=1)
y = df_clean['Diagnosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- KONTEN HALAMAN ---

if page == "🔬 Pendahuluan":
    st.title("Analisis Data Mining: Klasifikasi Kanker Payudara")
    st.markdown("### Menggunakan Dataset Wisconsin Diagnostic Breast Cancer (WDBC)")
    st.image("https://www.verywellhealth.com/thmb/392z2y0aWp3F54yv4vAF7A-Y330=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-signs-of-breast-cancer-430225-3000-a29241a2743a41b5a034226d97c55c74.jpg",
             caption="Sumber Gambar: verywellhealth.com", width=600)

    st.header("Apa itu Dataset WDBC?")
    st.write(
        """
        WDBC adalah dataset yang berisi data diagnosis kanker payudara berdasarkan hasil pemeriksaan mikroskopis 
        sel-sel yang diambil dari jaringan payudara. Fitur-fitur dalam dataset ini dihitung dari gambar digital 
        aspirasi jarum halus (Fine Needle Aspirate - FNA) dari massa payudara. Mereka menggambarkan karakteristik 
        inti sel yang ada dalam gambar.
        """
    )
    
    st.header("Sumber Data")
    st.markdown(
        """
        - **Donor**: Dr. William H. Wolberg (University of Wisconsin)
        - **Tanggal**: November 1995
        - **Sumber**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
        """
    )
    
    st.header("Tujuan Proyek")
    st.write(
        """
        Tujuan utama dari proyek ini adalah membangun model machine learning yang dapat secara akurat 
        mengklasifikasikan tumor payudara sebagai **'ganas' (Malignant)** atau **'jinak' (Benign)** berdasarkan fitur-fitur seluler yang diukur. Aplikasi ini akan memandu Anda melalui setiap 
        langkah, mulai dari eksplorasi data hingga prediksi interaktif.
        """
    )

elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi dan Analisis Data (EDA)")
    
    st.header("1. Tampilan Awal Dataset")
    st.write("Berikut adalah 5 baris pertama dari dataset mentah WDBC. Dataset ini memiliki 32 kolom.")
    st.dataframe(df_raw.head())
    
    st.header("2. Statistik Deskriptif")
    st.write("Statistik ini memberikan gambaran umum tentang sebaran setiap fitur numerik dalam data mentah.")
    st.dataframe(df_raw.describe())
    
    st.header("3. Distribusi Diagnosis")
    st.write("Diagram di bawah ini menunjukkan distribusi jumlah kasus Jinak (Benign - B) dan Ganas (Malignant - M) pada data mentah.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Diagnosis', data=df_raw, ax=ax1, palette=['skyblue', 'salmon'])
        ax1.set_title('Distribusi Diagnosis (B: Jinak, M: Ganas)')
        ax1.set_xlabel('Diagnosis')
        ax1.set_ylabel('Jumlah')
        st.pyplot(fig1)

    with col2:
        st.write("Jumlah data berdasarkan diagnosis:")
        diagnosis_counts = df_raw['Diagnosis'].value_counts()
        st.table(diagnosis_counts)
        st.write(f"Total data: {df_raw.shape[0]} sampel")
        st.write(f"Proporsi Ganas (M): **{diagnosis_counts['M']/df_raw.shape[0]:.2%}**")
        st.write(f"Proporsi Jinak (B): **{diagnosis_counts['B']/df_raw.shape[0]:.2%}**")

elif page == "⚙️ Pra-Pemrosesan Data":
    st.title("⚙️ Langkah-Langkah Pra-Pemrosesan Data")
    st.write("Sebelum data dapat digunakan untuk melatih model, beberapa langkah pembersihan dan persiapan perlu dilakukan.")

    st.header("1. Pengecekan Missing Values")
    st.write("Langkah pertama adalah memastikan tidak ada data yang hilang (kosong) di dalam dataset.")
    missing_values = df_raw.isnull().sum().sum()
    if missing_values == 0:
        st.success("✓ Dataset tidak memiliki missing values.")
    else:
        st.warning(f"⚠️ Ditemukan {missing_values} missing values yang perlu ditangani.")

    st.header("2. Penanganan Outlier")
    st.write("Outlier (data pencilan) dapat memengaruhi performa model. Di sini, kita menggunakan metode Interquartile Range (IQR) untuk mendeteksi dan menghapus outlier.")
    
    st.info(f"""
    - **Ukuran data sebelum penghapusan outlier**: {df_raw.shape[0]} baris
    - **Ukuran data setelah penghapusan outlier**: {df_clean.shape[0]} baris
    - **Jumlah data yang dihapus**: {df_raw.shape[0] - df_clean.shape[0]} baris
    - **Persentase data yang dihapus**: {(df_raw.shape[0] - df_clean.shape[0]) / df_raw.shape[0]:.2%}
    
    Penghapusan outlier yang signifikan ini dilakukan untuk mendapatkan data inti yang lebih homogen, dengan harapan meningkatkan performa generalisasi model pada data yang tidak ekstrem.
    """)
    if st.checkbox("Tampilkan data setelah penghapusan outlier"):
        st.dataframe(df_clean.head())

    st.header("3. Encoding Target")
    st.write("Model machine learning memerlukan input numerik. Oleh karena itu, kolom target 'Diagnosis' yang bersifat kategorikal ('B' dan 'M') diubah menjadi angka.")
    st.markdown(
        """
        - **B (Benign/Jinak)** diubah menjadi **0**
        - **M (Malignant/Ganas)** diubah menjadi **1**
        """
    )

    st.header("4. Pembagian Data (Split Data)")
    st.write("Data dibagi menjadi dua set: data latih (training set) untuk melatih model, dan data uji (test set) untuk menguji seberapa baik performa model pada data baru.")
    st.code(
        f"""
        - Data Latih (80%): {X_train.shape[0]} sampel
        - Data Uji (20%): {X_test.shape[0]} sampel
        """
    )
    
    st.header("5. Penskalaan Fitur (Feature Scaling)")
    st.write("Fitur-fitur numerik memiliki rentang nilai yang berbeda-beda. Feature scaling (menggunakan `StandardScaler`) dilakukan untuk menyamakan skala semua fitur (rata-rata=0, standar deviasi=1). Ini penting agar fitur dengan skala besar tidak mendominasi proses pembelajaran model.")
    if st.checkbox("Tampilkan contoh data setelah di-scaling"):
        st.dataframe(pd.DataFrame(X_train_scaled, columns=X.columns).head())


elif page == "🧠 Pemodelan & Evaluasi":
    st.title("🧠 Pembuatan, Pelatihan, dan Evaluasi Model")
    
    model_choice = st.selectbox(
        "Pilih model untuk dievaluasi:",
        ("Random Forest", "Logistic Regression", "Support Vector Machine (SVM)")
    )
    
    if model_choice == "Random Forest":
        st.header("Evaluasi Model: Random Forest")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Plot feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15), palette='viridis', ax=ax)
        ax.set_title('Top 15 Fitur Penting - Random Forest')
        ax.set_xlabel('Tingkat Kepentingan (Importance)')
        ax.set_ylabel('Fitur')
        st.pyplot(fig)

    elif model_choice == "Logistic Regression":
        st.header("Evaluasi Model: Logistic Regression")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Plot coefficients
        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_[0]
        })
        coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='coefficient', y='feature', data=coef_df.head(15), palette='coolwarm', ax=ax)
        ax.set_title('Top 15 Koefisien Fitur - Logistic Regression')
        ax.set_xlabel('Nilai Koefisien')
        ax.set_ylabel('Fitur')
        st.pyplot(fig)
        
    elif model_choice == "Support Vector Machine (SVM)":
        st.header("Evaluasi Model: Support Vector Machine (SVM)")
        st.write("Untuk SVM dengan kernel non-linear, feature importance tidak dapat dihitung secara langsung seperti pada model linear atau tree-based.")
        model = SVC(random_state=42, probability=True)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    st.subheader("Hasil Evaluasi")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Metrik Kinerja:**")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.metric(label="Akurasi", value=f"{accuracy:.4f}")
        st.metric(label="Presisi", value=f"{precision:.4f}")
        st.metric(label="Recall", value=f"{recall:.4f}")
        st.metric(label="F1-Score", value=f"{f1:.4f}")
    
    with col2:
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Prediksi Jinak', 'Prediksi Ganas'],
                    yticklabels=['Aktual Jinak', 'Aktual Ganas'])
        ax_cm.set_xlabel('Prediksi')
        ax_cm.set_ylabel('Aktual')
        st.pyplot(fig_cm)
    
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Jinak (0)', 'Ganas (1)'], output_dict=True)
    st.table(pd.DataFrame(report).transpose())

elif page == "💡 Prediksi Interaktif":
    st.title("💡 Prediksi Kanker Payudara Interaktif")
    st.write("Gunakan slider di bawah untuk memasukkan nilai fitur dari sampel tumor dan dapatkan prediksi dari model terbaik.")

    # Melatih kembali model terbaik (SVM berdasarkan notebook) untuk digunakan di sini
    best_model = SVC(random_state=42, probability=True)
    best_model.fit(X_train_scaled, y_train)

    st.sidebar.header("Input Fitur Pengguna")
    
    input_data = {}
    top_features = [
        'concave_points_worst', 'area_worst', 'perimeter_worst', 'radius_worst',
        'concave_points_mean', 'area_mean', 'perimeter_mean', 'radius_mean',
        'concavity_mean', 'area_se'
    ] # Fitur paling penting berdasarkan analisis RF

    st.info("Hanya 10 fitur terpenting yang ditampilkan untuk input. Fitur lainnya akan menggunakan nilai rata-ratanya.")

    for feature in X.columns:
        if feature in top_features:
            min_val = float(df_clean[feature].min())
            max_val = float(df_clean[feature].max())
            mean_val = float(df_clean[feature].mean())
            input_data[feature] = st.sidebar.slider(
                f'{feature}', 
                min_val, 
                max_val, 
                mean_val
            )
        else:
            input_data[feature] = float(df_clean[feature].mean())

    input_df = pd.DataFrame([input_data])
    
    if st.button("Prediksi Sekarang"):
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Lakukan prediksi
        prediction = best_model.predict(input_scaled)
        prediction_proba = best_model.predict_proba(input_scaled)
        
        st.subheader("Hasil Prediksi")
        
        diagnosis = le.inverse_transform(prediction)[0]
        
        if diagnosis == 'M':
            st.error("Hasil Prediksi: Ganas (Malignant)")
            st.write(f"**Probabilitas Ganas:** {prediction_proba[0][1]:.2%}")
            st.write(f"**Probabilitas Jinak:** {prediction_proba[0][0]:.2%}")
            st.warning("""
            **Peringatan:** Model memprediksi tumor ini bersifat **ganas**. Disarankan untuk segera melakukan konsultasi lebih lanjut dengan profesional medis untuk diagnosis dan penanganan yang tepat.
            """)
        else:
            st.success("Hasil Prediksi: Jinak (Benign)")
            st.write(f"**Probabilitas Jinak:** {prediction_proba[0][0]:.2%}")
            st.write(f"**Probabilitas Ganas:** {prediction_proba[0][1]:.2%}")
            st.info("""
            **Catatan:** Model memprediksi tumor ini bersifat **jinak**. Meskipun demikian, tetap penting untuk memantau kondisi dan berkonsultasi dengan dokter untuk memastikan kesehatan Anda.
            """)
        
        st.markdown("---")
        st.write("Input Fitur yang Anda Masukkan:")
        st.dataframe(input_df[top_features].T.rename(columns={0: 'Nilai'}))
