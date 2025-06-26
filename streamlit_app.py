# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

# =================================================================================
# KONFIGURASI HALAMAN & STYLE
# =================================================================================
st.set_page_config(
    page_title="Perbandingan Model Klasifikasi",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style untuk plot
plt.style.use('seaborn-v0_8-darkgrid')

# =================================================================================
# FUNGSI UNTUK MEMUAT DAN MEMPROSES DATA (dengan caching)
# =================================================================================
@st.cache_data
def load_and_preprocess_data():
    """Memuat dataset Kanker Payudara Wisconsin, membaginya, dan menskalakannya."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Skala fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Konversi kembali ke DataFrame untuk menjaga nama kolom
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, data.target_names

# =================================================================================
# FUNGSI UNTUK MELATIH & EVALUASI MODEL
# =================================================================================
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """Melatih model, membuat prediksi, dan mengembalikan semua metrik evaluasi."""
    # Training
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Classification Report & Confusion Matrix
    class_report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "classification_report": class_report,
        "confusion_matrix": cm
    }
    return results

# =================================================================================
# UTAMA: LAYOUT APLIKASI STREAMLIT
# =================================================================================

# --- Muat Data ---
X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names = load_and_preprocess_data()

# --- SIDEBAR: Kontrol Pengguna ---
st.sidebar.title("⚙️ Pengaturan Model")
st.sidebar.markdown("Pilih model klasifikasi dan atur hyperparameternya.")

model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    ("Random Forest", "Logistic Regression", "Support Vector Machine (SVM)")
)

# Hyperparameter dinamis berdasarkan pilihan model
params = {}
if model_choice == "Random Forest":
    st.sidebar.header("Hyperparameter Random Forest")
    params['n_estimators'] = st.sidebar.slider("Jumlah Pohon (n_estimators)", 50, 500, 100, 10)
    params['max_depth'] = st.sidebar.slider("Kedalaman Maksimum (max_depth)", 2, 32, 10, 1)
    
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )

elif model_choice == "Logistic Regression":
    st.sidebar.header("Hyperparameter Logistic Regression")
    params['C'] = st.sidebar.slider("Regularisasi (C)", 0.01, 10.0, 1.0, 0.01)
    params['max_iter'] = st.sidebar.slider("Iterasi Maksimum (max_iter)", 100, 2000, 1000, 100)
    
    model = LogisticRegression(
        C=params['C'],
        max_iter=params['max_iter'],
        random_state=42
    )

elif model_choice == "Support Vector Machine (SVM)":
    st.sidebar.header("Hyperparameter SVM")
    params['C'] = st.sidebar.slider("Regularisasi (C)", 0.01, 10.0, 1.0, 0.01)
    params['kernel'] = st.sidebar.selectbox("Kernel", ('rbf', 'linear', 'poly', 'sigmoid'))
    
    model = SVC(
        C=params['C'],
        kernel=params['kernel'],
        probability=True,  # Diperlukan untuk predict_proba
        random_state=42
    )

st.sidebar.info("Data yang digunakan adalah 'Wisconsin Breast Cancer Dataset' dari Scikit-learn.")
st.sidebar.markdown(f"Ukuran data training: `{X_train_scaled.shape}`")
st.sidebar.markdown(f"Ukuran data testing: `{X_test_scaled.shape}`")

# --- PANEL UTAMA: Tampilan Hasil ---
st.title("📊 Perbandingan Model Klasifikasi Machine Learning")
st.markdown(f"Menampilkan hasil evaluasi untuk model **{model_choice}**.")

# --- Latih dan Evaluasi Model Pilihan ---
with st.spinner(f"Melatih model {model_choice}, mohon tunggu..."):
    results = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

# --- Tampilkan Hasil Evaluasi ---
st.header("📈 Hasil Evaluasi")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{results['accuracy']:.4f}")
col2.metric("Precision", f"{results['precision']:.4f}")
col3.metric("Recall", f"{results['recall']:.4f}")
col4.metric("F1-Score", f"{results['f1_score']:.4f}")

st.metric(
    "Cross-Validation Accuracy",
    f"{results['cv_accuracy_mean']:.4f}",
    f"± {results['cv_accuracy_std'] * 2:.4f}"
)

# --- Tampilkan Laporan Klasifikasi dan Confusion Matrix ---
st.header("📋 Laporan Detail")

col1_report, col2_report = st.columns(2)

with col1_report:
    st.subheader("Classification Report")
    st.text(results['classification_report'])

with col2_report:
    st.subheader("Confusion Matrix")
    cm = results['confusion_matrix']
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=target_names, yticklabels=target_names)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    st.pyplot(fig_cm)

# --- Tampilkan Feature Importance / Coefficients ---
st.header("🚀 Feature Importance")

if model_choice == "Random Forest":
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': results['model'].feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    st.dataframe(feature_importance_df, use_container_width=True)
    
    # Visualisasi
    fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
    top_features = feature_importance_df.head(15)
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis', ax=ax_fi)
    ax_fi.set_title('Top 15 Feature Importance - Random Forest')
    ax_fi.invert_yaxis()
    st.pyplot(fig_fi)

elif model_choice == "Logistic Regression":
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': results['model'].coef_[0]
    })
    coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False).drop('abs_coefficient', axis=1).reset_index(drop=True)
    
    st.dataframe(coef_df, use_container_width=True)

    # Visualisasi
    fig_coef, ax_coef = plt.subplots(figsize=(10, 8))
    top_coef = coef_df.head(15)
    colors = ['red' if x < 0 else 'blue' for x in top_coef['coefficient']]
    sns.barplot(x='coefficient', y='feature', data=top_coef, palette=colors, ax=ax_coef)
    ax_coef.set_title('Top 15 Feature Coefficients - Logistic Regression')
    ax_coef.axvline(x=0, color='black', linewidth=0.8)
    ax_coef.invert_yaxis()
    st.pyplot(fig_coef)

elif model_choice == "Support Vector Machine (SVM)":
    if params['kernel'] == 'linear':
        # Feature importance hanya bisa langsung didapat dari koefisien kernel linear
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': results['model'].coef_[0]
        })
        coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False).drop('abs_coefficient', axis=1).reset_index(drop=True)

        st.dataframe(coef_df, use_container_width=True)

        # Visualisasi
        fig_coef_svm, ax_coef_svm = plt.subplots(figsize=(10, 8))
        top_coef_svm = coef_df.head(15)
        colors = ['red' if x < 0 else 'blue' for x in top_coef_svm['coefficient']]
        sns.barplot(x='coefficient', y='feature', data=top_coef_svm, palette=colors, ax=ax_coef_svm)
        ax_coef_svm.set_title('Top 15 Feature Coefficients - SVM (Linear Kernel)')
        ax_coef_svm.axvline(x=0, color='black', linewidth=0.8)
        ax_coef_svm.invert_yaxis()
        st.pyplot(fig_coef_svm)
    else:
        st.info(f"Feature importance tidak dapat divisualisasikan secara langsung untuk kernel SVM '{params['kernel']}'. Koefisien hanya tersedia untuk kernel 'linear'.")

st.success(f"✓ Analisis model {model_choice} selesai!")
