import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.write(""" 
# WELCOME TO PREDICT HEPATITIS C SYSTEM
""")

st.write("========================================================================================")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Deskripsi Data", "Dataset (Import Data)", "Preprocessing", "Modelling", "Implementasi"])

with tab1:
    st.write("HEPATITIS C DATASET")
    st.write("Dataset ini berisi nilai laboratorium donor darah dan pasien Hepatitis C serta nilai demografi seperti usia")
    st.write("Data ini digunakan untuk memodelkan penyakit Hepatitis C")
    st.write("Fitur - Fitur dalam dataset ini yang akan digunakan ialah sebagai berikut.\n
            1. ALB (Albumin) \n
            2. ALP (Alkaline Phosphatase) \n
            3. ALT (Alanine Transaminase) \n
            4. AST (Aspartate aminotransferase) \n
            5. BIL (Bilirubin) \n
            6. CHE (Cholinesterase) \n
            7. CHOL (Cholesterol) \n
            8. CREA (Creatin) \n
            9. GGT (Gamma-glutamyl transferase) \n
            10. PROT (Protein)")
    st.write("Sumber Data : https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset")

with tab2:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
        
with tab3 : 
    st.write("""# Preprocessing""")
    df.drop('Unnamed: 0',axis=1,inplace=True) 
    df.replace(to_replace=['0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis',
       '2=Fibrosis', '3=Cirrhosis'],
             value=['0','1','2','3','4'],inplace=True)
    df['Category'].unique()
    df.replace(to_replace=['m','f'],
             value=['0','1'],inplace=True)
    st.dataframe(df)
