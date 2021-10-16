import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from PIL import Image

image = Image.open("aqara.png")
st.sidebar.image(image,width = 200)
page=st.sidebar.selectbox("Explore or Predict using Aqara THP",("Predict","Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
    


    