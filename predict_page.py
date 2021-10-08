import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('discomfort_index.sav','rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

def show_predict_page():
    st.title("Discomfort Index Prediction")
    st.title("Using Aqara THP Sensor Data")
    
    st.write("""### Trained From Aqara Temperature and Humidity Sensor""")

    temperature = st.slider("Temperature", min_value=-20, max_value=50,value=10,step=1)
    humidity = st.slider("Humidity", min_value=1, max_value=99,value=30, step=1)
    air_pressure = st.slider("Air Pressure", min_value=900, max_value=1200, value=1000,step=10)

    ok = st.button("Predict Discomfort Index")
    

    if ok:
        #X = np.arrary([[temperature,humidity,air_pressure]])

        prediction = model.predict([[temperature,humidity,air_pressure]])
        if prediction == "HIGH":
            prediction = "Uncomfortable"
        elif prediction == "MIDDLE":
            prediction = "So-So"
        elif prediction == "LOW":
            prediction = "Comfortable"
        else:
            return "error"
        #st.subheader("The estimated discomfort index is ${discomfort_index[0]}")
        st.write("Discomfort Index Below")
        st.subheader(prediction)


        
        
    

     