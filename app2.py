from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('bayesian_ridge')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


def run():

    from PIL import Image
    image = Image.open('prim8_2.png')
    image_buildings = Image.open('buildings.jpg')

    st.image(image, use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict endprice of houses around Gothenburg')

    st.sidebar.image(image_buildings)

    st.title("House Endprice Prediction App")

    if add_selectbox == 'Online':
        kommun = st.selectbox('Area', ['undefined', 'Ale kommun', 'Stenungsunds kommun', 'Göteborgs kommun', 'Härryda kommun', 'Kungälvs kommun', 'Kungsbacka kommun', 'Lerums kommun', 'Mölndals kommun', 'Öckerö kommun', 'Partille kommun'])
        price = st.number_input('Price (kr)', min_value=1, value=1000000)
        rooms = st.number_input('Rooms', min_value=1, max_value=100, value=1)
        livingarea = st.number_input('Livingarea (m2)', min_value=1)

        output = ""

        input_dict = {'kommun': kommun, 'price': price, 'rooms': rooms, 'livingarea': livingarea}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = round(output)
            output = str(output) + ' kr'

        st.success('Prediction of endprice: {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)


if __name__ == '__main__':
    run()
