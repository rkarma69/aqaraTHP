#Aqara 온습도 센서 불쾌지수 구하기

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mysql.connector
import seaborn as sns
import streamlit as st
# Load dataset

names = ['Temperature', 'Humidity', 'AirPressure', 'Class']

connection = mysql.connector.connect(host='192.168.219.102',user='iotuser',password="iot12345", database='iot')
mycursor = connection.cursor()

sample_size = 1000

mycursor.execute("select temperature, humidity,pressure,discomfort from discomfortTable order by id desc limit {}".format(sample_size))
dataset=pd.DataFrame(list(mycursor))
connection.commit()
dataset.columns = names

dataset = dataset.rename(columns=lambda x: x.strip().lower())

g = sns.PairGrid(dataset, hue="class")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
st.pyplot(g)

x = dataset.drop(['class'],axis=1)
y = dataset['class']

# model fit

log_model = LogisticRegression(C=1)
log_model.fit(x,y)

import pickle
pickle.dump(log_model, open("/home/iotuser/mlapp/discomfort_index.sav","wb"))




