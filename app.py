import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load Model
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

st.set_page_config(page_title="Breast Cancer Diagnosis", layout="wide")
st.title("🩺 Breast Cancer Prediction App")

# History Storage Initialization
if 'history' not in st.session_state:
    st.session_state.history = []

st.subheader("Patient Input Features")
col1, col2, col3 = st.columns(3)

with col1:
    f1 = st.number_input("Clump Thickness", 1, 10, 5)
    f2 = st.number_input("Uniformity of Cell Size", 1, 10, 5)
    f3 = st.number_input("Uniformity of Cell Shape", 1, 10, 5)

with col2:
    f4 = st.number_input("Marginal Adhesion", 1, 10, 5)
    f5 = st.number_input("Single Epithelial Cell Size", 1, 10, 5)
    f6 = st.number_input("Bare Nuclei", 1, 10, 5)

with col3:
    f7 = st.number_input("Bland Chromatin", 1, 10, 5)
    f8 = st.number_input("Normal Nucleoli", 1, 10, 5)
    f9 = st.number_input("Mitoses", 1, 10, 5)

if st.button("Predict"):
    features = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])
    prediction = model.predict(features)[0]
    
    result = "Malignant (Cancerous)" if prediction == 4 else "Benign (Non-Cancerous)"
    
    if prediction == 4:
        st.error(f"Prediction Result: *{result}*")
    else:
        st.success(f"Prediction Result: *{result}*")
        
    # Save to Session History
    st.session_state.history.append({
        "Clump Thickness": f1,
        "Cell Size": f2,
        "Prediction": result
    })

# Display History Section
if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 Prediction History")
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
