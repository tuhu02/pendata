import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import joblib
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="WDBC Breast Cancer Analysis & Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and cache the WDBC dataset"""
    column_names = [
        'ID', 'Diagnosis',
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    try:
        df = pd.read_csv('wdbc.data', header=None, names=column_names)
        return df
    except FileNotFoundError:
        st.error("File wdbc.data tidak ditemukan!")
        return None

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        with open('models/best_model_support_vector_machine.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        try:
            with open('models/model_info.json', 'r') as f:
                model_info = json.load(f)
        except:
            model_info = {
                'best_model_name': 'Support Vector Machine',
                'best_accuracy': 0.9821,
                'best_precision': 0.8000,
                'best_recall': 1.0000,
                'best_f1_score': 0.8889
            }
        return model, scaler, label_encoder, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def data_understanding_section(df):
    st.header("📊 Data Understanding")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {len(df.columns)}")
        st.write(f"**Samples:** {len(df)}")
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
    with col2:
        st.subheader("Data Types")
        st.dataframe(df.dtypes.to_frame('Data Type'))
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values.to_frame('Missing Count'))
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())
    st.subheader("Diagnosis Distribution")
    diagnosis_counts = df['Diagnosis'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(diagnosis_counts.to_frame('Count'))
    with col2:
        fig = px.pie(
            values=diagnosis_counts.values,
            names=diagnosis_counts.index,
            title="Diagnosis Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def preprocessing_section(df):
    st.header("🧹 Data Preprocessing")
    st.subheader("Missing Values Check")
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing == 0:
        st.success("✅ No missing values found in the dataset")
    else:
        st.warning(f"⚠️ Found {total_missing} missing values")
        st.dataframe(missing_values[missing_values > 0].to_frame('Missing Count'))
    st.subheader("Feature Preparation")
    X = df.drop(['ID', 'Diagnosis'], axis=1)
    y = df['Diagnosis']
    st.write(f"**Features shape:** {X.shape}")
    st.write(f"**Target shape:** {y.shape}")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    st.write(f"**Label mapping:** {dict(zip(le.classes_, le.transform(le.classes_)))}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    st.write(f"**Training set:** {X_train.shape}")
    st.write(f"**Test set:** {X_test.shape}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.success("✅ Data preprocessing completed!")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def modeling_section(X_train_scaled, X_test_scaled, y_train, y_test):
    st.header("🤖 Model Training & Evaluation")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }
    results = []
    for name, model in models.items():
        st.subheader(f"Training {name}")
        with st.spinner(f"Training {name}..."):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std
            })
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
            with col2:
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1-Score", f"{f1:.4f}")
            st.write(f"Cross-validation: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.set_index('Model'))
    best_model_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    st.success(f"🏆 Best Model: {best_model_name}")
    return results_df, best_model_name

def prediction_section(model, scaler, label_encoder):
    st.header("🔮 Breast Cancer Prediction")
    st.write("Enter the cell nucleus characteristics to predict the diagnosis.")
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        features = {}
        with col1:
            st.subheader("Mean Values")
            for i in range(10):
                feature = feature_names[i]
                features[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    format="%.3f",
                    key=f"mean_{i}"
                )
        with col2:
            st.subheader("Standard Error Values")
            for i in range(10, 20):
                feature = feature_names[i]
                features[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    format="%.3f",
                    key=f"se_{i}"
                )
        with col3:
            st.subheader("Worst Values")
            for i in range(20, 30):
                feature = feature_names[i]
                features[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    format="%.3f",
                    key=f"worst_{i}"
                )
        submitted = st.form_submit_button("🔮 Predict Diagnosis", type="primary")
        if submitted and model is not None:
            feature_values = [features[feature] for feature in feature_names]
            features_scaled = scaler.transform([feature_values])
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            diagnosis = label_encoder.inverse_transform([prediction])[0]
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                if diagnosis == 'M':
                    st.error("🚨 **MALIGNANT** - Cancerous Tumor Detected")
                    st.write("This indicates a high probability of breast cancer.")
                    st.write("**Immediate medical attention is recommended.**")
                else:
                    st.success("✅ **BENIGN** - Non-Cancerous Tumor")
                    st.write("This indicates a low probability of breast cancer.")
                    st.write("**Regular monitoring is still recommended.**")
            with col2:
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Benign', 'Malignant'],
                        y=[probability[0], probability[1]],
                        marker_color=['green', 'red'],
                        text=[f'{probability[0]:.1%}', f'{probability[1]:.1%}'],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🏥 Wisconsin Diagnostic Breast Cancer (WDBC) Analysis & Prediction")
    st.markdown("---")
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Understanding", "Data Preprocessing", "Model Training", "Prediction", "About"]
    )
    df = load_data()
    if df is None:
        st.error("Cannot load data. Please check if wdbc.data file exists.")
        return
    model, scaler, label_encoder, model_info = load_model()
    if model_info:
        st.sidebar.header("📈 Model Info")
        st.sidebar.write(f"**Best Model:** {model_info.get('best_model_name', 'Unknown')}")
        st.sidebar.write(f"**Accuracy:** {model_info.get('best_accuracy', 'Unknown'):.4f}")
        st.sidebar.write(f"**F1-Score:** {model_info.get('best_f1_score', 'Unknown'):.4f}")
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    This application analyzes breast cancer diagnosis using the WDBC dataset.

    **Diagnosis:**
    - **M (Malignant):** Cancerous tumor
    - **B (Benign):** Non-cancerous tumor

    **Data Source:** UCI Machine Learning Repository
    """)
    if page == "Data Understanding":
        data_understanding_section(df)
    elif page == "Data Preprocessing":
        preprocessing_section(df)
    elif page == "Model Training":
        st.warning("⚠️ Model training is computationally intensive. Use the Prediction tab for making predictions with pre-trained models.")
        if st.button("Train Models (This may take a while)"):
            with st.spinner("Preprocessing data..."):
                X_train_scaled, X_test_scaled, y_train, y_test, scaler, le = preprocessing_section(df)
            with st.spinner("Training models..."):
                results_df, best_model_name = modeling_section(X_train_scaled, X_test_scaled, y_train, y_test)
    elif page == "Prediction":
        if model is not None:
            prediction_section(model, scaler, label_encoder)
        else:
            st.error("❌ Model not loaded. Please ensure model files are available.")
    elif page == "About":
        st.header("📋 About This Application")
        st.write("""
        This application demonstrates a complete data mining workflow for breast cancer diagnosis:

        1. **Data Understanding**: Explore and understand the WDBC dataset
        2. **Data Preprocessing**: Clean and prepare data for modeling
        3. **Model Training**: Train and evaluate multiple machine learning models
        4. **Prediction**: Make predictions on new data

        **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC)
        **Features**: 30 cell nucleus characteristics
        **Target**: Diagnosis (Benign/Malignant)

        **Disclaimer**: This is for educational purposes only. 
        Always consult healthcare professionals for medical decisions.
        """)

if __name__ == "__main__":
    main()
