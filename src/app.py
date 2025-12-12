import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# Set page config
st.set_page_config(page_title="Churn Prediction System (Dev Mode)", layout="wide")

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join("src", "model", "model.pkl")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Header
st.title("⚡ Customer Churn Prediction System")
st.markdown("**Developer Edition**: Advanced insights and model transparency.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Customer Profile")
    
    # Input Form
    with st.form("churn_form"):
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Services")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.subheader("Account")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=50.0)
        
        submitted = st.form_submit_button("Predict Churn Risk")

if submitted and model:
    # Prepare DataFrame
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    df = pd.DataFrame([input_data])
    
    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    with col2:
        st.header("📊 Prediction Analysis")
        
        # 1. Gauge Chart
        st.subheader("Churn Probability")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Result Badge
        if prediction == 1:
            st.error(f"🔴 PREDICTION: CHURN RISKY (Probability: {probability:.2%})")
        else:
            st.success(f"🟢 PREDICTION: LOYAL (Probability: {probability:.2%})")

        
        # 2. Global Feature Importance (Developer Insight)
        st.subheader("🔍 Global Feature Importance (Model Level)")
        
        try:
            # Extract Feature Names and Importances
            # Step 1: Get Preprocessor
            preprocessor = model.named_steps['preprocessor']
            # Step 2: Get Classifier
            classifier = model.named_steps['classifier']
            
            # Helper to get feature names from ColumnTransformer
            def get_feature_names(column_transformer):
                output_features = []
                
                for name, pipe, features in column_transformer.transformers_:
                    if name == 'remainder':
                        continue
                    if hasattr(pipe, 'get_feature_names_out'):
                        # For OneHotEncoder
                        f_names = pipe.get_feature_names_out(features)
                        output_features.extend(f_names)
                    else:
                        # For StandardScaler/SimpleImputer (Numeric)
                        output_features.extend(features)
                return output_features

            feature_names = get_feature_names(preprocessor)
            importances = classifier.feature_importances_
            
            feat_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(10)
            
            fig_bar = px.bar(
                feat_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Top 10 Influential Features"
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info("💡 Note: These are global feature importances derived from the Random Forest model training, indicating which features generally drive decisions.")
            
        except Exception as e:
            st.warning(f"Could not extract feature importance: {e}")
            
        # 3. Raw Data View
        with st.expander("Show Raw Input Data"):
            st.json(input_data)
            
elif not model:
    st.warning("Model file not found. Please run training script first: `python src/model/train.py`")
