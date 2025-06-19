# Import library yang dibutuhkan
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle # Library untuk menyimpan objek Python

print("Memulai proses training model final...")

# 1. Muat Dataset yang Sudah Dibersihkan
try:
    df_clean = pd.read_csv('bank-full_no_outliers.csv', sep=';')
    print("Dataset bersih berhasil dimuat.")
except FileNotFoundError:
    print("GAGAL: File 'bank-full_no_outliers.csv' tidak ditemukan.")
    exit()

# 2. Pra-pemrosesan Variabel Target (y)
df_clean['y'] = df_clean['y'].map({'yes': 1, 'no': 0})

# 3. Memisahkan Fitur (X) dan Target (y)
X = df_clean.drop('y', axis=1)
y = df_clean['y']

# 4. Mengidentifikasi Tipe Kolom
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 5. Membuat dan Melatih Preprocessor
# Preprocessor ini akan disave untuk digunakan di aplikasi Streamlit
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Latih preprocessor pada seluruh data
print("Melatih preprocessor pada seluruh data...")
X_processed = preprocessor.fit_transform(X)

# 6. Melatih Model Decision Tree Final
# Kita gunakan seluruh data yang sudah bersih untuk melatih model final
print("Melatih model Decision Tree final...")
final_dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', min_samples_leaf=10, random_state=42)
final_dt_model.fit(X_processed, y)
print("Model final berhasil dilatih.")

# 7. Menyimpan Model dan Preprocessor
# Kita simpan kedua objek ini agar bisa dimuat di aplikasi Streamlit
with open('model.pkl', 'wb') as model_file:
    pickle.dump(final_dt_model, model_file)

with open('preprocessor.pkl', 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

print("\nPROSES SELESAI!")
print("Model telah disimpan sebagai 'model.pkl'")
print("Preprocessor telah disimpan sebagai 'preprocessor.pkl'")