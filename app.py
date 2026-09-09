import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# Page Setup
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")

# Database Connection
conn = sqlite3.connect('history.db', check_same_thread=False)
c = conn.cursor()

# Create History Table if not exists
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        prediction TEXT
    )
''')
conn.commit()

# Load Trained Model
@st.cache_resource
def load_model():
    with open('breast_cancer_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

st.title("🩺 Breast Cancer Prediction App")
st.write("Enter the patient's clinical parameters below to predict tumor diagnosis.")

# Input Form
st.subheader("Patient Input Features")
col1, col2, col3 = st.columns(3)

with col1:
    clump_thickness = st.number_input("Clump Thickness", 1, 10, 5)
    uniformity_cell_size = st.number_input("Uniformity of Cell Size", 1, 10, 5)
    uniformity_cell_shape = st.number_input("Uniformity of Cell Shape", 1, 10, 5)

with col2:
    marginal_adhesion = st.number_input("Marginal Adhesion", 1, 10, 5)
    single_epithelial_size = st.number_input("Single Epithelial Cell Size", 1, 10, 5)
    bare_nuclei = st.number_input("Bare Nuclei", 1, 10, 5)

with col3:
    bland_chromatin = st.number_input("Bland Chromatin", 1, 10, 5)
    normal_nucleoli = st.number_input("Normal Nucleoli", 1, 10, 5)
    mitoses = st.number_input("Mitoses", 1, 10, 5)

# Prediction Logic
if st.button("Predict Diagnosis", type="primary"):
    features = np.array([[clump_thickness, uniformity_cell_size, uniformity_cell_shape,
                          marginal_adhesion, single_epithelial_size, bare_nuclei,
                          bland_chromatin, normal_nucleoli, mitoses]])
    
    prediction = model.predict(features)[0]
    result_text = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
    
    # Save to SQLite Database with Timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO predictions (timestamp, prediction) VALUES (?, ?)", (current_time, result_text))
    conn.commit()
    
    if prediction == 1:
        st.error(f"*Prediction Result:* {result_text}")
    else:
        st.success(f"*Prediction Result:* {result_text}")

# History Section (Past 7 Days Data Only)
st.markdown("---")
st.subheader("📋 Past 1 Week Prediction History")

# Query last 7 days records
one_week_ago = datetime.now() - timedelta(days=7)
df_history = pd.read_sql_query(
   "SELECT strftime('%Y-%m-%d %H:%M:%S', timestamp) AS 'Date & Time', prediction AS 'Prediction Result' FROM predictions WHERE timestamp >= ? ORDER BY id DESC"
    conn, 
    params=(one_week_ago,)
)

if not df_history.empty:
    st.dataframe(df_history, use_container_width=True)
else:
    st.info("No prediction history recorded in the past 7 days.")
