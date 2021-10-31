import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from model_page import show_model_page
from deep_page import show_deep_page
from PIL import Image

image = Image.open("aqara.png")
st.sidebar.image(image,width = 200)
page=st.sidebar.selectbox("Explore or Predict using Aqara THP",("Predict","Explore","Model Evaluation","Deep Learning"))

if page == "Predict":
    show_predict_page()
elif page == "Explore":
    show_explore_page()
elif page =="Model Evaluation":
    show_model_page()
else:
    show_deep_page()
    


    