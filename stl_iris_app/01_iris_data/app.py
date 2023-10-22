import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Defile & Select Columns in Iris Dataset
st.title('Iris Flower')
st.markdown('Create `Scatter Plot` to Display Iris Flower Dataset')
col_list = ["sepal.length","sepal.width","petal.length","petal.width"]

#Create Selectbox Feature to Select X and Y
selected_x_var = st.selectbox('Select for X axis',col_list)
selected_y_var = st.selectbox('Select for Y axis',col_list)

#Create File Uploader
iris_file = st.file_uploader("Choose a file", type =['csv'])

#Read File
if iris_file is not None:
    iris_file_df = pd.read_csv(iris_file)
else:
    st.stop()

#Create Table
st.subheader('Simple Data')
st.write(iris_file_df)

#Create Scatter Plot
st.subheader('Scatter Plot')
sns.set_style('darkgrid')
markers = {"Setosa": "v", "Versicolor": "s", "Virginica": 'o'}
fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_file_df,
                     x=selected_x_var, y=selected_y_var,
                     hue='variety', markers=markers, style='variety')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("Iris Flower Data")
st.pyplot(fig)
#%%
