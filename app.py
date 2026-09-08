import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

st.title("Breast Cancer Prediction App")

# Input fields for the 9 features
clump = st.number_input("Clump Thickness", 1, 10)
size = st.number_input("Uniformity of Cell Size", 1, 10)
shape = st.number_input("Uniformity of Cell Shape", 1, 10)
m_adh = st.number_input("Marginal Adhesion", 1, 10)
s_size = st.number_input("Single Epithelial Cell Size", 1, 10)
b_nuc = st.number_input("Bare Nuclei", 1, 10)
b_chrom = st.number_input("Bland Chromatin", 1, 10)
n_nucl = st.number_input("Normal Nucleoli", 1, 10)
mitoses = st.number_input("Mitoses", 1, 10)

if st.button("Predict"):
    features = np.array([[clump, size, shape, m_adh, s_size, b_nuc, b_chrom, n_nucl, mitoses]])
    prediction = model.predict(features)
    if prediction[0] == 4:
        st.error("Malignant (Cancerous)")
    else:
        st.success("Benign (Safe)")
