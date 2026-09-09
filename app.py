import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import streamlit_authenticator as stauth

# Page Setup
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")

# Database Connection
conn = sqlite3.connect('history.db', check_same_thread=False)
c = conn.cursor()

# Create Tables for Predictions and Users
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

# User Authentication Management
st.title("🩺 Breast Cancer Prediction App")

menu = ["Login", "Sign Up"]
choice = st.sidebar.selectbox("Account Menu", menu)

if choice == "Sign Up":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')
    
    if st.button("Sign Up"):
        if new_user and new_password:
            try:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, new_password))
                conn.commit()
                st.success("Account created successfully! Please go to the Login menu.")
            except sqlite3.IntegrityError:
                st.error("Username already exists. Try another one.")
        else:
            st.warning("Please enter both username and password.")

elif choice == "Login":
    st.sidebar.subheader("Login Section")
    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password", type='password')
    
    if st.sidebar.button("Login"):
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        
        if user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.sidebar.success(f"Logged in as {username}")
        else:
            st.sidebar.error("Incorrect Username/Password")

    if st.session_state.get('logged_in'):
        st.write(f"Welcome **{st.session_state['username']}**! Enter clinical parameters below:")
        
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.experimental_rerun()

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
            
            # Save Prediction with Logged-in Username
            current_time = datetime.now()
            c.execute("INSERT INTO predictions (username, timestamp, prediction) VALUES (?, ?, ?)", 
                      (st.session_state['username'], current_time, result_text))
            conn.commit()
            
            if prediction == 1:
                st.error(f"**Prediction Result:** {result_text}")
            else:
                st.success(f"**Prediction Result:** {result_text}")

        # Individual Prediction History (Past 1 Week)
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
    else:
        st.info("Please login or sign up from the sidebar to use the prediction tool.")
