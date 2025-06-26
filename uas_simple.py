# Analisis Data Wisconsin Diagnostic Breast Cancer (WDBC)
# Tahapan: Data Understanding, Preprocessing, Modelling, Evaluasi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=== ANALISIS DATA WDBC ===")
print("Tahapan: Data Understanding -> Preprocessing -> Modelling -> Evaluasi\n")

# ========================================
# 1. DATA UNDERSTANDING
# ========================================
print("1. DATA UNDERSTANDING")
print("=" * 50)

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

print(f"Shape data: {df.shape}")
print(f"Jumlah sampel: {df.shape[0]}")
print(f"Jumlah fitur: {df.shape[1]}")
print(f"Kolom: {list(df.columns)}")

# Informasi dataset
print("\nInformasi dataset:")
print(df.info())

# Statistik deskriptif
print("\nStatistik deskriptif:")
print(df.describe())

# Distribusi target
print("\nDistribusi Diagnosis:")
diagnosis_counts = df['Diagnosis'].value_counts()
print(diagnosis_counts)
print(f"Persentase:")
print(diagnosis_counts / len(df) * 100)

# Cek missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ========================================
# 2. PREPROCESSING
# ========================================
print("\n\n2. PREPROCESSING")
print("=" * 50)

# Persiapan data
X = df.drop(['ID', 'Diagnosis'], axis=1)
y = df['Diagnosis']

print(f"Shape X (features): {X.shape}")
print(f"Shape y (target): {y.shape}")

# Encoding target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Target encoded: {le.classes_} -> {le.transform(le.classes_)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Distribusi target training: {np.bincount(y_train)}")
print(f"Distribusi target test: {np.bincount(y_test)}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data telah di-scale menggunakan StandardScaler")

# ========================================
# 3. MODELLING
# ========================================
print("\n\n3. MODELLING")
print("=" * 50)

# Inisialisasi model
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Training model
trained_models = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Training
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    
    # Probabilities
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        probabilities[name] = y_prob
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

print("\nSemua model telah dilatih!")

# ========================================
# 4. EVALUASI
# ========================================
print("\n\n4. EVALUASI")
print("=" * 50)

# Evaluasi setiap model
for name in trained_models.keys():
    print(f"\n{'='*60}")
    print(f"EVALUASI {name.upper()}")
    print(f"{'='*60}")
    
    y_pred = predictions[name]
    y_prob = probabilities.get(name)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # ROC AUC Score
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(trained_models[name], X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature Importance (Random Forest)
print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (Random Forest)")
print(f"{'='*60}")

rf_model = trained_models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importance:")
print(feature_importance.head(10))

# Perbandingan model
print(f"\n{'='*60}")
print("PERBANDINGAN MODEL")
print(f"{'='*60}")

comparison_data = []
for name in trained_models.keys():
    accuracy = accuracy_score(y_test, predictions[name])
    roc_auc = roc_auc_score(y_test, probabilities[name]) if probabilities.get(name) is not None else None
    comparison_data.append({
        'Model': name,
        'Accuracy': accuracy,
        'ROC AUC': roc_auc
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

print("\nPerbandingan Performa Model:")
print(comparison_df)

# Kesimpulan
print(f"\n{'='*60}")
print("KESIMPULAN")
print(f"{'='*60}")

best_model = comparison_df.iloc[0]['Model']
best_accuracy = comparison_df.iloc[0]['Accuracy']

print(f"\nModel terbaik: {best_model}")
print(f"Accuracy: {best_accuracy:.4f}")

# Analisis confusion matrix model terbaik
best_cm = confusion_matrix(y_test, predictions[best_model])
tn, fp, fn, tp = best_cm.ravel()

print(f"\nAnalisis Confusion Matrix ({best_model}):")
print(f"True Negatives (Benign correctly classified): {tn}")
print(f"False Positives (Benign misclassified as Malignant): {fp}")
print(f"False Negatives (Malignant misclassified as Benign): {fn}")
print(f"True Positives (Malignant correctly classified): {tp}")

sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate

print(f"\nSensitivity (True Positive Rate): {sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")

print(f"\nRekomendasi:")
print(f"1. Model {best_model} memberikan performa terbaik dengan accuracy {best_accuracy:.1%}")
print(f"2. Model dapat mengidentifikasi {sensitivity:.1%} kasus malignant dengan benar")
print(f"3. Model dapat mengidentifikasi {specificity:.1%} kasus benign dengan benar")
print(f"4. Fitur-fitur yang paling penting untuk diagnosis adalah:")
for i, (feature, importance) in enumerate(feature_importance.head(5).values, 1):
    print(f"   {i}. {feature} (importance: {importance:.4f})")

print(f"\nKesimpulan:")
print(f"Dataset WDBC berhasil dianalisis dengan tahapan yang lengkap.")
print(f"Model machine learning dapat membantu dalam diagnosis kanker payudara")
print(f"dengan tingkat akurasi yang tinggi ({best_accuracy:.1%}).")

print(f"\n{'='*60}")
print("ANALISIS SELESAI")
print(f"{'='*60}") 