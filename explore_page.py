import streamlit as st
import pandas as pd
import matplotlib 
import seaborn as sns
import pickle
import mysql.connector

# Load dataset
@st.cache
def load_data():
    names = ['Temperature', 'Humidity', 'AirPressure', 'Class']
    #connection = mysql.connector.connect(host='192.168.219.102',user='iotuser',password="iot12345", database='iot')
    connection = mysql.connector.connect(host='112.157.171.74',user='iotuser',password="iot12345", database='iot')

    mycursor = connection.cursor()
    
    sample_size = 1000
    
    mycursor.execute("select temperature, humidity,pressure,discomfort from discomfortTable order by id desc limit {}".format(sample_size))
    dataset=pd.DataFrame(list(mycursor))
    connection.commit()
    
    dataset.columns = names
    dataset = dataset.rename(columns=lambda x: x.strip().lower())
    return dataset


dataset = load_data()
subdata = dataset[["temperature","humidity","airpressure"]]


def show_explore_page():
    st.title("Visualize Data")
    st.write("""### Visualize Aqara Temperature-Humidity Sensor Data""")
    choosePlot=st.selectbox("Correlation chart, bar chart or data frame",("Correlation","Bar Chart", "Data Frame"))
    if choosePlot == "Correlation":
        st.write("""### Pair Grid Chart for Correlation""")
        g = sns.PairGrid(dataset, hue="class")
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot)
        g.add_legend()
        g.savefig("output.png")
        st.pyplot(g)
    elif choosePlot == "Bar Chart":
        st.write("""### Bar Chart""")
        st.bar_chart(subdata)
    else:
        test=st.text_area("Dataset Head",pd.DataFrame.to_string(dataset),height=150)

    # 데이터 
# box plot
 #   boxplot = dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
 #   st.pyplot(boxplot)

# histogram
 #   histogram=dataset.hist()
 #   st.pyplot(histogram)

# scatter plot matrix
  #  scatter= sns.scatterplot(x="temperature",y="humidity",data=dataset)
   # st.pyplot(scatter)
