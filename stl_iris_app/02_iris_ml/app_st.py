import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import  pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Iris Flower Classification')
st.write('This app uses 4 inputs to predict the variety of iris flower using a model that build on iris flower dataset. Use the form below to get started')

iris_file = st.file_uploader('Upload Iris Flower Dataset')

if iris_file is None:
    rfc_pickle = open('random_forest_iris.pickle', 'rb')
    map_pickle = open('output_iris.pickle', 'rb')
    rfc_model = pickle.load(rfc_pickle)
    unique_iris_mapping = pickle.load(map_pickle)
    rfc_pickle.close()
else:
    iris_df = pd.read_csv(iris_file)
    iris_df = iris_df.dropna()

    output = iris_df['variety']
    features = iris_df[["sepal.length","sepal.width","petal.length","petal.width"]]

    features = pd.get_dummies(features)

    output, unique_iris_mapping = pd.factorize(output)
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size= 0.8)

    rfc_model = RandomForestClassifier(random_state=15)
    rfc_model.fit(X_train,y_train)

    y_pred = rfc_model.predict(X_test)

    score = round(accuracy_score(y_pred,y_test),2)

    st.write(f'We trained a Random Forest model on iris dataset, It has score of {score} ! Use the inputs below to try out this medel.')

with st.form('user_input'):
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=8.0,step=.1,format="%.1f")
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=5.0,step=.1,format="%.1f")
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=7.0,step=.1,format="%.1f")
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=3.0,step=.1,format="%.1f")
    st.form_submit_button()

new_prediction = rfc_model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
prediction_variety = unique_iris_mapping[new_prediction][0]
st.write(f'We predict your data is {prediction_variety} veriety !!')

if prediction_variety == 'Setosa':
    image = Image.open('setosa.jpg')
    st.image(image, caption= ' Setosa')
elif prediction_variety == 'Versicolor':
    image = Image.open('versicolor.jpg')
    st.image(image, caption= 'Versicolor')
else:
    image = Image.open('virginica.jpg')
    st.image(image, caption= 'Virginica')