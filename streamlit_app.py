import streamlit as st
#!/usr/bin/env python
# coding: utf-8

# # Analisis Data Mining: Klasifikasi Kanker Payudara dengan Dataset WDBC

# Apa itu Dataset WDBC?
# WDBC adalah dataset yang berisi data diagnosis kanker payudara berdasarkan hasil pemeriksaan mikroskopis sel-sel yang diambil dari jaringan payudara.

#   Sumber Data  
# - Donor: Dr. William H. Wolberg (University of Wisconsin)
# - Tanggal: November 1995
# - Sumber: UCI Machine Learning Repository
# - Link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

# Cara Pengambilan Data  
# Data diperoleh dari Fine Needle Aspirate (FNA) - yaitu:  
# - Mengambil sampel jaringan payudara dengan jarum halus
# - Sampel kemudian diperiksa di bawah mikroskop
# - Gambar digital dari sel-sel tersebut dianalisis
# - Karakteristik sel diukur dan dihitung secara otomatis

# ## 1. Data Understanding
# 
# Pada tahap ini, kita akan:
# - Membaca dataset WDBC ke dalam DataFrame.
# - Melihat struktur data (jumlah baris, kolom, tipe data).
# - Menampilkan beberapa baris pertama data.
# - Melihat statistik deskriptif untuk fitur numerik.
# - Mengecek distribusi label diagnosis (benign/malignant).
# - Mengecek apakah ada data yang hilang (missing values).
# 
# Langkah ini penting untuk memastikan data yang akan dianalisis sudah benar dan siap untuk tahap selanjutnya.

# In[34]:


# Import library yang diperlukan
import pandas as pd

# Membaca data
column_names = [
    'ID', 'Diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]
df = pd.read_csv('wdbc.data', header=None, names=column_names)

# Melihat 5 baris pertama
st.write(df.head())

# Melihat info struktur data
df.info()

# Statistik deskriptif fitur numerik
df.describe()

# Distribusi label diagnosis
print(df['Diagnosis'].value_counts())
print(df['Diagnosis'].value_counts(normalize=True) * 100)

# Mengecek missing values
print('Jumlah missing values per kolom:')
print(df.isnull().sum())


# Distribusi

# In[35]:


import matplotlib.pyplot as plt

df['Diagnosis'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribusi Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Jumlah')
plt.show()


# ## 2. Pre Processing Data

# Langkah Preprocessing untuk Dataset WDBC

# ### 2.1 Cek Missing Values

# In[36]:


# ========================================
# CEK MISSING VALUES
# ========================================
print("=== CEK MISSING VALUES ===")

# Cek missing values per kolom
missing_values = df.isnull().sum()
print("Missing values per kolom:")
print(missing_values)

# Total missing values
total_missing = missing_values.sum()
print(f"\nTotal missing values: {total_missing}")

if total_missing == 0:
    print("✓ Dataset tidak memiliki missing values")
else:
    print("⚠️ Dataset memiliki missing values yang perlu ditangani")

    # Visualisasi missing values
    plt.figure(figsize=(12, 6))
    missing_values[missing_values > 0].plot(kind='bar', color='red')
    plt.title('Missing Values per Kolom')
    plt.xlabel('Kolom')
    plt.ylabel('Jumlah Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ### 2.2 Deteksi Outlier

# In[37]:


# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


# In[38]:


# ========================================
# STATISTIK DESKRIPTIF UNTUK DETEKSI OUTLIER
# ========================================
print("=== STATISTIK DESKRIPTIF ===")

# Ambil kolom numerik (exclude ID)
numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_columns = numeric_columns.drop('ID')

print("Statistik deskriptif untuk fitur numerik:")
print(df[numeric_columns].describe())

# Cek range nilai untuk setiap fitur
print("\nRange nilai per fitur:")
for col in numeric_columns:
    min_val = df[col].min()
    max_val = df[col].max()
    range_val = max_val - min_val
    print(f"{col}: {min_val:.4f} - {max_val:.4f} (range: {range_val:.4f})")


# In[39]:


# ========================================
# DETEKSI DAN HAPUS OUTLIERS
# ========================================
print("=== DETEKSI DAN HAPUS OUTLIERS ===")

def detect_and_remove_outliers(df, columns, method='iqr'):
    """
    Deteksi dan hapus outliers menggunakan IQR method
    """
    df_clean = df.copy()
    total_removed = 0

    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Hitung jumlah outliers
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            outlier_count = len(outliers)

            if outlier_count > 0:
                print(f"{col}: {outlier_count} outliers")
                # Hapus outliers
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                total_removed += outlier_count
            else:
                print(f"{col}: ✓ Tidak ada outliers")

    return df_clean, total_removed

# Deteksi dan hapus outliers
numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_columns = numeric_columns.drop('ID')

print("Deteksi outliers per kolom:")
df_clean, total_removed = detect_and_remove_outliers(df, numeric_columns, method='iqr')

print(f"\nTotal outliers yang dihapus: {total_removed}")
print(f"Data sebelum: {len(df)} baris")
print(f"Data setelah: {len(df_clean)} baris")
print(f"Persentase data yang dihapus: {(total_removed/len(df)*100):.2f}%")

# Simpan data bersih
df_clean.to_csv('wdbc_clean_no_outliers.csv', index=False)
print("✓ Data bersih disimpan sebagai 'wdbc_clean_no_outliers.csv'")


# ### 2.4 Encoding Target
# - Mengubah label kategorikal (B/M) menjadi numerik (0/1)
# - B (Benign) = 0
# - M (Malignant) = 1

# In[40]:


df = df_clean

# 1. Persiapan Data
X = df.drop(['ID', 'Diagnosis'], axis=1)
y = df['Diagnosis']


# In[41]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Target encoded: {le.classes_} -> {le.transform(le.classes_)}")
print(f"B (Benign) -> {le.transform(['B'])[0]}")
print(f"M (Malignant) -> {le.transform(['M'])[0]}")


# ### 2.5 Split Data
# - Membagi data menjadi training set (80%) dan test set (20%)
# - Menggunakan stratify untuk menjaga proporsi kelas
# - random_state=42 untuk hasil yang konsisten

# In[42]:


from sklearn.model_selection import train_test_split

# Split data 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Distribusi target training: {np.bincount(y_train)}")
print(f"Distribusi target test: {np.bincount(y_test)}")


# ### 2.6 Feature Scaling
# - Menstandarisasi fitur numerik menggunakan StandardScaler
# - Mengubah semua fitur ke skala yang sama (mean=0, std=1)
# - Penting untuk algoritma yang sensitif terhadap skala data

# In[43]:


from sklearn.preprocessing import StandardScaler

# Scaling fitur numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data telah di-scale menggunakan StandardScaler")
print(f"Training set scaled shape: {X_train_scaled.shape}")
print(f"Test set scaled shape: {X_test_scaled.shape}")


# ## 3. Data Modeling

# In[44]:


# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("✓ Library berhasil diimport")


# Load Data

# In[45]:


# Load data bersih yang sudah dihapus outliers
try:
    df = pd.read_csv('wdbc_clean_no_outliers.csv')
    print("✓ Data bersih berhasil dimuat")
except FileNotFoundError:
    print("⚠️ File 'wdbc_clean_no_outliers.csv' tidak ditemukan")
    print("Menggunakan data asli...")
    # Load data asli jika file bersih tidak ada
    column_names = [
        'ID', 'Diagnosis',
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    df = pd.read_csv('wdbc.data', header=None, names=column_names)

print(f"Shape data: {df.shape}")
print(f"Jumlah sampel: {len(df)}")


# In[46]:


# ========================================
# PREPROCESSING DATA
# ========================================
print("=== PREPROCESSING DATA ===")

# 1. Persiapan Data
X = df.drop(['ID', 'Diagnosis'], axis=1)
y = df['Diagnosis']

print(f"Shape X (features): {X.shape}")
print(f"Shape y (target): {y.shape}")

# 2. Encoding Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Target encoded: {le.classes_} -> {le.transform(le.classes_)}")
print(f"B (Benign) -> {le.transform(['B'])[0]}")
print(f"M (Malignant) -> {le.transform(['M'])[0]}")

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Distribusi target training: {np.bincount(y_train)}")
print(f"Distribusi target test: {np.bincount(y_test)}")

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Data telah di-scale menggunakan StandardScaler")
print(f"Training set scaled shape: {X_train_scaled.shape}")
print(f"Test set scaled shape: {X_test_scaled.shape}")

# Verifikasi scaling
print(f"\nStatistik training set setelah scaling:")
print(f"Mean: {X_train_scaled.mean():.6f}")
print(f"Std: {X_train_scaled.std():.6f}")


# ### Random Forest

# In[47]:


# ========================================
# MODEL 1: RANDOM FOREST
# ========================================
print("=== MODEL 1: RANDOM FOREST ===")

# Inisialisasi model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training model
print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)

# Prediksi
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluasi
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Cross validation
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_mean_rf = cv_scores_rf.mean()
cv_std_rf = cv_scores_rf.std()

# Tampilkan hasil
print(f"\n HASIL EVALUASI RANDOM FOREST:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
print(f"CV Accuracy: {cv_mean_rf:.4f} (+/- {cv_std_rf*2:.4f})")

# Classification report
print(f"\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'Malignant']))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"\n CONFUSION MATRIX:")
print(cm_rf)

# Feature Importance
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n TOP 10 FEATURE IMPORTANCE:")
print(feature_importance_rf.head(10))

# Visualisasi feature importance
plt.figure(figsize=(12, 8))
top_features_rf = feature_importance_rf.head(15)
plt.barh(range(len(top_features_rf)), top_features_rf['importance'], color='skyblue')
plt.yticks(range(len(top_features_rf)), top_features_rf['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("✓ Random Forest selesai!")


# ### Logistic Regression

# In[48]:


# ========================================
# MODEL 2: LOGISTIC REGRESSION
# ========================================
print("=== MODEL 2: LOGISTIC REGRESSION ===")

# Inisialisasi model
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Training model
print("Training Logistic Regression...")
lr_model.fit(X_train_scaled, y_train)

# Prediksi
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluasi
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

# Cross validation
cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_mean_lr = cv_scores_lr.mean()
cv_std_lr = cv_scores_lr.std()

# Tampilkan hasil
print(f"\n HASIL EVALUASI LOGISTIC REGRESSION:")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-Score: {f1_lr:.4f}")
print(f"CV Accuracy: {cv_mean_lr:.4f} (+/- {cv_std_lr*2:.4f})")

# Classification report
print(f"\n CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_lr, target_names=['Benign', 'Malignant']))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"\n CONFUSION MATRIX:")
print(cm_lr)

# Coefficients (feature importance untuk logistic regression)
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\n TOP 10 FEATURE COEFFICIENTS:")
print(coef_df.head(10))

# Visualisasi coefficients
plt.figure(figsize=(12, 8))
top_coef = coef_df.head(15)
colors = ['red' if x < 0 else 'blue' for x in top_coef['coefficient']]
plt.barh(range(len(top_coef)), top_coef['coefficient'], color=colors)
plt.yticks(range(len(top_coef)), top_coef['feature'])
plt.xlabel('Coefficient Value')
plt.title('Top 15 Feature Coefficients - Logistic Regression')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("✓ Logistic Regression selesai!")


# ### Support Vector Machine

# In[49]:


# ========================================
# MODEL 3: SUPPORT VECTOR MACHINE (SVM)
# ========================================
print("=== MODEL 3: SUPPORT VECTOR MACHINE ===")

# Inisialisasi model
svm_model = SVC(random_state=42, probability=True)

# Training model
print("Training Support Vector Machine...")
svm_model.fit(X_train_scaled, y_train)

# Prediksi
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# Evaluasi
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

# Cross validation
cv_scores_svm = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_mean_svm = cv_scores_svm.mean()
cv_std_svm = cv_scores_svm.std()

# Tampilkan hasil
print(f"\n HASIL EVALUASI SUPPORT VECTOR MACHINE:")
print(f"Accuracy: {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall: {recall_svm:.4f}")
print(f"F1-Score: {f1_svm:.4f}")
print(f"CV Accuracy: {cv_mean_svm:.4f} (+/- {cv_std_svm*2:.4f})")

# Classification report
print(f"\n CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_svm, target_names=['Benign', 'Malignant']))

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(f"\n CONFUSION MATRIX:")
print(cm_svm)

print("✓ Support Vector Machine selesai!")


# ### Perbandingan Model

# In[50]:


# ========================================
# SIMPAN BEST MODEL
# ========================================
print("=== SIMPAN BEST MODEL ===")

import joblib
import os

# Buat folder models jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')
    print("✓ Folder 'models' berhasil dibuat")

# Tentukan model terbaik berdasarkan F1-Score
best_model_idx = comparison_df['F1-Score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']

# Pilih model yang akan disimpan
if best_model_name == 'Random Forest':
    best_model = rf_model
    print("✓ Random Forest dipilih sebagai model terbaik")
elif best_model_name == 'Logistic Regression':
    best_model = lr_model
    print("✓ Logistic Regression dipilih sebagai model terbaik")
else:
    best_model = svm_model
    print("✓ Support Vector Machine dipilih sebagai model terbaik")

# Simpan model
model_filename = f'models/best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
joblib.dump(best_model, model_filename)
print(f"✓ Model tersimpan sebagai: {model_filename}")

# Simpan scaler
scaler_filename = 'models/scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✓ Scaler tersimpan sebagai: {scaler_filename}")

# Simpan label encoder
encoder_filename = 'models/label_encoder.pkl'
joblib.dump(le, encoder_filename)
print(f"✓ Label encoder tersimpan sebagai: {encoder_filename}")

# Simpan informasi model
model_info = {
    'best_model_name': best_model_name,
    'best_f1_score': best_f1_score,
    'best_accuracy': best_accuracy,
    'best_precision': best_precision,
    'best_recall': best_recall,
    'feature_names': list(X.columns),
    'model_filename': model_filename,
    'scaler_filename': scaler_filename,
    'encoder_filename': encoder_filename
}

import json
with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)
print("✓ Informasi model tersimpan sebagai: models/model_info.json")

# Verifikasi file tersimpan
print(f"\n FILES YANG TERSIMPAN:")
print(f"1. {model_filename}")
print(f"2. {scaler_filename}")
print(f"3. {encoder_filename}")
print(f"4. models/model_info.json")

# Cek ukuran file
import os
for filename in [model_filename, scaler_filename, encoder_filename]:
    size = os.path.getsize(filename) / 1024  # dalam KB
    print(f"   {os.path.basename(filename)}: {size:.2f} KB")

print(f"\n BEST MODEL BERHASIL DISIMPAN!")
print(f"Model {best_model_name} siap digunakan untuk deployment.")
