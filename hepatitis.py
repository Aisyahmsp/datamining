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
    st.write("Fitur - Fitur dalam dataset ini yang akan digunakan ialah...")
    st.write("Sumber Data : https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset")

with tab2:
    uploaded_file = st.file_uploader("Choose a file csv")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)

