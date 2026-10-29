import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import joblib
import os

# ==========================================
# 1. PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="Health Analytics & Risk Assessor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-header {
        font-size:2.5rem;
        font-weight:700;
        color: #1E3A8A;
        margin-bottom:0px;
    }
    .sub-header {
        font-size:1.1rem;
        color: #4B5563;
        margin-bottom:25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. AUTOMATIC MODEL TRAINING & LOADING
# ==========================================
MODEL_FILE = 'diabetes_model.pkl'

@st.cache_resource
def load_or_train_model():
    """Trains a Random Forest classifier on diabetes data if model doesn't exist."""
    if not os.path.exists(MODEL_FILE):
        # Load sample diabetes dataset
        raw_data = load_diabetes(as_frame=True)
        df = raw_data.frame
        
        # Rename columns to human-readable names
        df = df.rename(columns={
            'age': 'Age',
            'sex': 'Sex',
            'bmi': 'BMI',
            'bp': 'Blood_Pressure',
            's1': 'Total_Cholesterol',
            's2': 'LDL',
            's3': 'HDL',
            's4': 'TCH',
            's5': 'Glucose',
            's6': 'Insulin'
        })
        
        # Binary target: Above median risk vs Below median risk
        df['Target'] = (df['target'] > df['target'].median()).astype(int)
        
        feature_cols = ['Age', 'BMI', 'Blood_Pressure', 'Glucose', 'Insulin']
        X = df[feature_cols]
        y = df['Target']
        
        # Train Random Forest Classifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Save model and feature list
        joblib.dump({'model': model, 'features': feature_cols}, MODEL_FILE)
    
    saved_data = joblib.load(MODEL_FILE)
    return saved_data['model'], saved_data['features']

model, feature_names = load_or_train_model()


# ==========================================
# 3. SIDEBAR - USER INPUTS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
st.sidebar.title("Patient Information")
st.sidebar.write("Adjust vitals to test risk assessment:")

age = st.sidebar.slider("Age (Years)", min_value=18, max_value=90, value=45)
gender = st.sidebar.radio("Sex", options=["Male", "Female"])
bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=50.0, value=24.5, step=0.1)
bp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", min_value=70, max_value=300, value=100)
insulin = st.sidebar.slider("Insulin Level (μU/mL)", min_value=15, max_value=276, value=80)


# ==========================================
# 4. MAIN DASHBOARD CONTENT
# ==========================================
st.markdown('<p class="main-header">🩺 AI Health Risk Analytics Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predictive machine learning evaluation based on clinical biomarkers.</p>', unsafe_allow_html=True)

# Layout Columns
col1, col2 = st.columns([1, 1])

# Convert inputs into model-compatible format (normalizing roughly to dataset distribution)
scaled_age = (age - 45) / 20.0
scaled_bmi = (bmi - 25) / 5.0
scaled_bp = (bp - 120) / 15.0
scaled_glucose = (glucose - 100) / 30.0
scaled_insulin = (insulin - 80) / 40.0

input_df = pd.DataFrame([[scaled_age, scaled_bmi, scaled_bp, scaled_glucose, scaled_insulin]], columns=feature_names)

# Make Prediction
prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]
risk_score = probabilities[1] * 100  # High risk percentage

with col1:
    st.subheader("📊 Patient Risk Profile")
    
    # Gauge Chart for Risk Percentage
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={'suffix': "%"},
        title={'text': "Metabolic Risk Index"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2563EB"},
            'steps': [
                {'range': [0, 40], 'color': "#D1FAE5"},
                {'range': [40, 70], 'color': "#FEF3C7"},
                {'range': [70, 100], 'color': "#FEE2E2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Risk Assessment Box
    if risk_score >= 70:
        st.error("⚠️ **High Risk Profile Detected**\nThe model indicates elevated risk based on current glucose, BMI, and blood pressure levels. Clinical consultation recommended.")
    elif risk_score >= 40:
        st.warning("⚡ **Moderate Risk Profile**\nSome health metrics are slightly elevated. Monitoring lifestyle, nutrition, and physical activity is suggested.")
    else:
        st.success("✅ **Low Risk Profile**\nBiomarkers are within healthy ranges. Continue regular annual wellness screenings.")

with col2:
    st.subheader("🔬 AI Model Explanation")
    st.write("Feature Importance according to the trained Random Forest Classifier:")
    
    # Feature Importance Plot
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'Biomarker': feature_names, 'Importance': importance}).sort_values('Importance', ascending=True)
    
    fig_bar = px.bar(
        importance_df, 
        x='Importance', 
        y='Biomarker', 
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        title="Key Biomarkers Influencing Assessment"
    )
    fig_bar.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# Footer / Disclaimer
st.caption("ℹ️ **Disclaimer:** This tool is designed strictly for educational and portfolio demonstration purposes using machine learning algorithms. It is not intended for official clinical diagnoses.")