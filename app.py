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

# Create Tables
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        timestamp DATETIME,
        prediction TEXT
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
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

# Initialize Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# Centered Auth View (If Not Logged In)
if not st.session_state['logged_in']:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3 columns layout to center the login box
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tab1:
            st.subheader("Login to your account")
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", type="primary", use_container_width=True):
                c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (login_user, login_pass))
                user = c.fetchone()
                if user:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = login_user
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

        with tab2:
            st.subheader("Create a new account")
            new_user = st.text_input("Choose Username", key="new_user")
            new_pass = st.text_input("Choose Password", type="password", key="new_pass")
            
            if st.button("Sign Up", use_container_width=True):
                if new_user and new_pass:
                    try:
                        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, new_pass))
                        conn.commit()
                        st.success("Account created! Go to Login tab.")
                    except sqlite3.IntegrityError:
                        st.error("Username already taken.")
                else:
                    st.warning("Please fill in both fields.")

# Dashboard View (If Logged In)
else:
    col_user, col_logout = st.columns([8, 2])
    with col_user:
        st.success(f"Welcome, **{st.session_state['username']}**! You are logged in.")
    with col_logout:
        if st.button("Logout", type="secondary"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = ""
            st.rerun()

    # Clinical Input Form
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
        
        current_time = datetime.now()
        c.execute("INSERT INTO predictions (username, timestamp, prediction) VALUES (?, ?, ?)", 
                  (st.session_state['username'], current_time, result_text))
        conn.commit()
        
        if prediction == 1:
            st.error(f"**Prediction Result:** {result_text}")
        else:
            st.success(f"**Prediction Result:** {result_text}")

    # History Section (Past 7 Days Data Only)
    st.markdown("---")
    st.subheader(f"📋 Past 1 Week History for {st.session_state['username']}")

    one_week_ago = datetime.now() - timedelta(days=7)

    query = """
    SELECT strftime('%Y-%m-%d %H:%M:%S', timestamp) AS "Date & Time", 
           prediction AS "Prediction Result" 
    FROM predictions 
    WHERE username = ? AND timestamp >= ? 
    ORDER BY id DESC
    """

    df_history = pd.read_sql_query(query, conn, params=(st.session_state['username'], one_week_ago))

    if not df_history.empty:
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No prediction history recorded in the past 7 days.")
