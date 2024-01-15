import streamlit as st
import pandas as pd
import pickle
from utils import read_config

st.set_page_config(
    page_title="Home",
    page_icon="📊",
    layout='centered'
)

st.write("# Welcome to use this extra ML tool")
# st.sidebar.success("Select the step")

st.markdown("""
    ### Flexible interface with SAP IBP CI-DS
    
    We only provide upload data manually now
            """)

st.image("https://github.com/tamakiseda/streamlit-/blob/main/pic3.png?raw=true")
csv_url = "https://raw.githubusercontent.com/tamakiseda/kmeans-/main/test_data.csv"
if st.button("Download Example"):
    # Generate a link for downloading the CSV file
    csv_link = f'<a href="{csv_url}" download="test_data.csv">Click here to download the CSV file</a>'
    st.markdown(csv_link, unsafe_allow_html=True)
# File uploader with restriction to Excel and CSV files

uploaded_file = st.file_uploader(
    'File uploader', type=['xlsx', 'xls', 'csv'],
    accept_multiple_files=False)
if uploaded_file is not None:
    st.success('File successfully uploaded!')



st.markdown(
    """
        Features of our tiny platform

    **👈 Try to explore from the sidebar**

    # Main Steps

    ### Export data from IBP
    - Upload data from Home page manually
    - Get data form IBP directly 

    ### Pre-processing 
    - Data Standardization
    - Categorical Variable
    - Dimensionality Reduction

    ### Select Algorithms
    - Cluster
    - Regression
    - Classification


    ### Select Parameters 
    - Number of clusters
    - Iteration times
    


    ### Modeling
    -To run the model


    ### Evalutate
    - Silhouette Score
    - Davies-Bouldin Index	
    - Calinski-Harabasz Index	

"""
)

# Use Pandas to read the file into a DataFrame
if uploaded_file.name.endswith(('.xls', '.xlsx')):
    df_ori = pd.read_excel(uploaded_file)
else:
    df_ori = pd.read_csv(uploaded_file)


st.session_state['data'] = df_ori

# 读取之前的配置参数
if 'configs' not in st.session_state:
    print('load classification confings')
    st.session_state['configs'] = read_config('configs.json')
if 'ori_config' not in st.session_state:
    st.session_state['ori_config'] = {}

st.markdown("""
    ### Do you want to learn more❔
    - Check [Machine Learing](https://scikit-learn.org/stable/index.html)
    - Check [Doc of Streamlit](https://docs.streamlit.io)
            """)



