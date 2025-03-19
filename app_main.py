import streamlit as st
import Classification 
import Clustering_Algorithms 
import Linear_Regression
import Semi_supervised
import NN
import PCA_t_sne 

st.set_page_config(page_title="Machine Learning App", layout="wide")

menu = ["Semi-supervised","Classification", "Clustering", "Linear Regression", "PCA/t-SNE", "Neural network"]
choice = st.sidebar.selectbox("Chọn chức năng", menu)


if choice == "Semi-supervised":
    Semi_supervised.run()
elif choice == "Classification":
    Classification.run()
elif choice == "Clustering":
    Clustering_Algorithms.run()
elif choice == "Linear Regression":
    Linear_Regression.run()
elif choice == "PCA/t-SNE":
    PCA_t_sne.run()
elif choice == "Neural network":
    NN.run()
