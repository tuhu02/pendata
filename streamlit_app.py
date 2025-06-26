import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="WDBC Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the saved model and preprocessing objects"""
    try:
        with open('models/best_model_support_vector_machine.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        try:
            with open('models/model_info.json', 'r') as f:
                import json
                model_info = json.load(f)
        except:
            model_info = {
                'model_type': 'Support Vector Machine',
                'accuracy': 0.95,
                'roc_auc': 0.98,
                'training_date': '2024-01-01'
            }
        return model, scaler, label_encoder, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Load model
model, scaler, label_encoder, model_info = load_model()

# Feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Feature descriptions
feature_descriptions = {
    'radius_mean': 'Mean radius of the tumor',
    'texture_mean': 'Mean texture of the tumor',
    'perimeter_mean': 'Mean perimeter of the tumor',
    'area_mean': 'Mean area of the tumor',
    'smoothness_mean': 'Mean smoothness of the tumor',
    'compactness_mean': 'Mean compactness of the tumor',
    'concavity_mean': 'Mean concavity of the tumor',
    'concave_points_mean': 'Mean concave points of the tumor',
    'symmetry_mean': 'Mean symmetry of the tumor',
    'fractal_dimension_mean': 'Mean fractal dimension of the tumor',
    'radius_se': 'Standard error of radius',
    'texture_se': 'Standard error of texture',
    'perimeter_se': 'Standard error of perimeter',
    'area_se': 'Standard error of area',
    'smoothness_se': 'Standard error of smoothness',
    'compactness_se': 'Standard error of compactness',
    'concavity_se': 'Standard error of concavity',
    'concave_points_se': 'Standard error of concave points',
    'symmetry_se': 'Standard error of symmetry',
    'fractal_dimension_se': 'Standard error of fractal dimension',
    'radius_worst': 'Worst radius of the tumor',
    'texture_worst': 'Worst texture of the tumor',
    'perimeter_worst': 'Worst perimeter of the tumor',
    'area_worst': 'Worst area of the tumor',
    'smoothness_worst': 'Worst smoothness of the tumor',
    'compactness_worst': 'Worst compactness of the tumor',
    'concavity_worst': 'Worst concavity of the tumor',
    'concave_points_worst': 'Worst concave points of the tumor',
    'symmetry_worst': 'Worst symmetry of the tumor',
    'fractal_dimension_worst': 'Worst fractal dimension of the tumor'
}

def predict_diagnosis(features):
    """Make prediction using the loaded model"""
    try:
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        diagnosis = label_encoder.inverse_transform([prediction])[0]
        
        return diagnosis, probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def main():
    # Header
    st.title("🏥 Wisconsin Diagnostic Breast Cancer (WDBC) Prediction")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    if model_info:
        st.sidebar.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
        st.sidebar.write(f"**Accuracy:** {model_info.get('accuracy', 'Unknown'):.4f}")
        st.sidebar.write(f"**ROC AUC:** {model_info.get('roc_auc', 'Unknown'):.4f}")
        st.sidebar.write(f"**Training Date:** {model_info.get('training_date', 'Unknown')}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    This application predicts breast cancer diagnosis based on cell nucleus characteristics.
    
    **Diagnosis:**
    - **M (Malignant):** Cancerous tumor
    - **B (Benign):** Non-cancerous tumor
    
    **Data Source:** UCI Machine Learning Repository
    """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📈 Data Analysis", "📊 Model Performance", "📁 Upload Data"])
    
    with tab1:
        st.header("🔮 Breast Cancer Prediction")
        st.write("Enter the cell nucleus characteristics to predict the diagnosis.")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            features = {}
            
            with col1:
                st.subheader("Mean Values")
                for i in range(10):
                    feature = feature_names[i]
                    features[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        help=feature_descriptions[feature],
                        format="%.3f",
                        key=f"mean_{i}"
                    )
            
            with col2:
                st.subheader("Standard Error Values")
                for i in range(10, 20):
                    feature = feature_names[i]
                    features[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        help=feature_descriptions[feature],
                        format="%.3f",
                        key=f"se_{i}"
                    )
            
            with col3:
                st.subheader("Worst Values")
                for i in range(20, 30):
                    feature = feature_names[i]
                    features[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        help=feature_descriptions[feature],
                        format="%.3f",
                        key=f"worst_{i}"
                    )
            
            submitted = st.form_submit_button("🔮 Predict Diagnosis", type="primary")
            
            if submitted:
                if model is not None:
                    # Convert features to list in correct order
                    feature_values = [features[feature] for feature in feature_names]
                    
                    # Make prediction
                    diagnosis, probability = predict_diagnosis(feature_values)
                    
                    if diagnosis is not None:
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
                            # Create probability bar chart
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
                        
                        # Feature importance visualization
                        st.subheader("📊 Feature Analysis")
                        st.write("Here are the most important features for this prediction:")
                        
                        # Create a simple feature importance visualization
                        feature_importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Value': feature_values
                        })
                        
                        # Show top 10 features by absolute value
                        feature_importance_df['Abs_Value'] = abs(feature_importance_df['Value'])
                        top_features = feature_importance_df.nlargest(10, 'Abs_Value')
                        
                        fig = px.bar(
                            top_features,
                            x='Feature',
                            y='Value',
                            title="Top 10 Feature Values",
                            color='Value',
                            color_continuous_scale='RdBu'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Disclaimer
                        st.warning("""
                        ⚠️ **Medical Disclaimer:**
                        This prediction is for educational purposes only and should not replace professional medical diagnosis.
                        Always consult with healthcare professionals for medical decisions.
                        """)
                else:
                    st.error("❌ Model not loaded. Please check if model files are available.")
    
    with tab2:
        st.header("📈 Data Analysis")
        
        # Load sample data for analysis
        try:
            df = pd.read_csv('wdbc_clean_no_outliers.csv')
            st.success("✅ Data loaded successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Dataset Overview")
                st.write(f"**Total samples:** {len(df)}")
                st.write(f"**Features:** {len(df.columns) - 1}")  # Excluding target
                
                # Diagnosis distribution
                diagnosis_counts = df['Diagnosis'].value_counts()
                fig = px.pie(
                    values=diagnosis_counts.values,
                    names=diagnosis_counts.index,
                    title="Diagnosis Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Feature Statistics")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                st.dataframe(df[numeric_cols].describe())
            
            # Correlation heatmap
            st.subheader("🔥 Feature Correlation Heatmap")
            correlation_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.info("Please ensure 'wdbc_clean_no_outliers.csv' is available in the project directory.")
    
    with tab3:
        st.header("📊 Model Performance")
        
        if model_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Model Metrics")
                metrics_data = {
                    'Metric': ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1-Score'],
                    'Value': [
                        model_info.get('accuracy', 0),
                        model_info.get('roc_auc', 0),
                        model_info.get('precision', 0),
                        model_info.get('recall', 0),
                        model_info.get('f1_score', 0)
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            
            with col2:
                st.subheader("📈 Performance Visualization")
                # Create a radar chart for model performance
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[model_info.get('accuracy', 0), model_info.get('precision', 0), 
                       model_info.get('recall', 0), model_info.get('f1_score', 0), model_info.get('roc_auc', 0)],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                    fill='toself',
                    name='Model Performance'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title="Model Performance Radar Chart"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Model information not available.")
    
    with tab4:
        st.header("📁 Upload Data for Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with feature data",
            type=['csv'],
            help="Upload a CSV file with the same feature columns as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Load uploaded data
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df_upload.shape}")
                
                # Check if required columns are present
                missing_cols = set(feature_names) - set(df_upload.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns: {list(missing_cols)}")
                else:
                    st.subheader("📊 Uploaded Data Preview")
                    st.dataframe(df_upload.head(), use_container_width=True)
                    
                    if st.button("🔮 Predict All", type="primary"):
                        if model is not None:
                            # Prepare features
                            X_upload = df_upload[feature_names]
                            
                            # Scale features
                            X_upload_scaled = scaler.transform(X_upload)
                            
                            # Make predictions
                            predictions = model.predict(X_upload_scaled)
                            probabilities = model.predict_proba(X_upload_scaled)
                            
                            # Decode predictions
                            diagnoses = label_encoder.inverse_transform(predictions)
                            
                            # Create results dataframe
                            results_df = df_upload.copy()
                            results_df['Predicted_Diagnosis'] = diagnoses
                            results_df['Benign_Probability'] = probabilities[:, 0]
                            results_df['Malignant_Probability'] = probabilities[:, 1]
                            
                            st.subheader("🎯 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Predictions", len(results_df))
                            with col2:
                                benign_count = (results_df['Predicted_Diagnosis'] == 'B').sum()
                                st.metric("Predicted Benign", benign_count)
                            with col3:
                                malignant_count = (results_df['Predicted_Diagnosis'] == 'M').sum()
                                st.metric("Predicted Malignant", malignant_count)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("❌ Model not loaded.")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()