import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.write(""" 
# WELCOME TO PREDICT HEPATITIS C SYSTEM
""")

st.write("========================================================================================")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Deskripsi Data", "Dataset (Import Data)", "Preprocessing", "Modelling", "Implementasi"])

with tab1:
    st.write("HEPATITIS C DATASET")
    st.write("Dataset ini berisi nilai laboratorium donor darah dan pasien Hepatitis C serta nilai demografi seperti usia")
    st.write("Data ini digunakan untuk memodelkan penyakit Hepatitis C")
    st.write("Fitur - Fitur dalam dataset ini yang akan digunakan ialah sebagai berikut.")
    st.write("1. ALB (Albumin) : Numerik ")
    st.write("2. ALP (Alkaline Phosphatase) : Numerik ")
    st.write("3. ALT (Alanine Transaminase) : Numerik ")
    st.write("4. AST (Aspartate aminotransferase) : Numerik ")
    st.write("5. BIL (Bilirubin) : Numerik ")
    st.write("6. CHE (Cholinesterase) : Numerik ")
    st.write("7. CHOL (Cholesterol) : Numerik ")
    st.write("8. CREA (Creatin) : Numerik ")
    st.write("9. GGT (Gamma-glutamyl transferase) : Numerik ")
    st.write("10. PROT (Protein) : Numerik ")
    st.write("11. Sex : Kategorikal ")
    st.write("12. Age  : Numerik ")
    st.write("13. Category : Kategorikal (0 = Blood Donor, 1 = Suspect Blood Donor, 2 = Hepatitis, 3 = Fibrosis, 4 = Cirrhosis ")
    st.write("Sumber Data : https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset")
    st.write("Sumber Dataset Cleaning : https://raw.githubusercontent.com/Aisyahmsp/datamining/main/hepatitis.csv")
    

with tab2:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
        
with tab3 : 
    st.write("""# Preprocessing""")
    df[["Age", "Sex", "AST", "BIL", "CHE", "CREA", "GGT"]].agg(['min','max'])

    df.Category.value_counts()
    df = df.drop(columns=["Unnamed: 0","ALB","ALP","ALT","CHOL","PROT"])

    X = df.drop(columns="Category")
    y = df.Category
    "### Membuang fitur yang tidak diperlukan"
    df

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.Category).columns.values.tolist()

    "### Label"
    labels

    st.markdown("# Normalize")

    "### Normalize data"

    dataubah=df.drop(columns="Sex")
    dataubah

    "### Normalize data sex"
    data_sex=df[['Sex']]
    sex = pd.get_dummies(data_sex)
    sex

    dataOlah = pd.concat([sex], axis=1)
    dataHasil = pd.concat([df,dataOlah], axis = 1)

    X = dataHasil.drop(columns=["Sex","Category"])
    y = dataHasil.Category
    "### Normalize data hasil"
    X

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(dataHasil.Category).columns.values.tolist()
    
    "### Label"
    labels

    # """## Normalisasi MinMax Scaler"""


    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape

