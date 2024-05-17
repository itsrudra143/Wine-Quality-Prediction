## Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

# Load the dataset
wine = pd.read_csv("WineQT.csv")

# Prepare the data for modeling
X = wine.drop("quality", axis=1)
y = wine["quality"].apply(lambda value: 1 if value >= 7 else 0)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Streamlit UI
st.set_page_config(
    page_title="Wine Quality Prediction", page_icon=":wine_glass:", layout="wide"
)

# CSS styling
css = """
<style>
    .stHeader {
        background-color: #8C1AFF;
        color: white;
        padding: 10px;
        text-align: center;
    }
    .stButton > button {
        background-color: #8C1AFF;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
    }
    .st-emotion-cache-sh2krr p {
        font-size: 22px;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="stHeader"><h1>Wine Quality Prediction</h1></div>',
    unsafe_allow_html=True,
)

# Columns for input sliders
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider(
        "Fixed Acidity",
        float(X["fixed acidity"].min()),
        float(X["fixed acidity"].max()),
        float(X["fixed acidity"].mean()),
        0.01,
        label_visibility="visible",
        key="fixed_acidity",
    )
    volatile_acidity = st.slider(
        "Volatile Acidity",
        float(X["volatile acidity"].min()),
        float(X["volatile acidity"].max()),
        float(X["volatile acidity"].mean()),
        0.01,
        label_visibility="visible",
        key="volatile_acidity",
    )
    citric_acid = st.slider(
        "Citric Acid",
        float(X["citric acid"].min()),
        float(X["citric acid"].max()),
        float(X["citric acid"].mean()),
        0.01,
        label_visibility="visible",
        key="citric_acid",
    )
    residual_sugar = st.slider(
        "Residual Sugar",
        float(X["residual sugar"].min()),
        float(X["residual sugar"].max()),
        float(X["residual sugar"].mean()),
        0.01,
        label_visibility="visible",
        key="residual_sugar",
    )   
    chlorides = st.slider(
        "Chlorides",
        float(X["chlorides"].min()),
        float(X["chlorides"].max()),
        float(X["chlorides"].mean()),
        0.01,
        label_visibility="visible",
        key="chlorides",
    )
    free_sulfur_dioxide = st.slider(
        "Free Sulfur Dioxide",
        float(X["free sulfur dioxide"].min()),
        float(X["free sulfur dioxide"].max()),
        float(X["free sulfur dioxide"].mean()),
        0.01,
        label_visibility="visible",
        key="free_sulfur_dioxide",
    )

with col2:
    total_sulfur_dioxide = st.slider(
        "Total Sulfur Dioxide",
        float(X["total sulfur dioxide"].min()),
        float(X["total sulfur dioxide"].max()),
        float(X["total sulfur dioxide"].mean()),
        0.01,
        label_visibility="visible",
        key="total_sulfur_dioxide",
    )
    density = st.slider(
        "Density",
        float(X["density"].min()),
        float(X["density"].max()),
        float(X["density"].mean()),
        0.0001,
        label_visibility="visible",
        key="density",
    )
    pH = st.slider(
        "pH",
        float(X["pH"].min()),
        float(X["pH"].max()),
        float(X["pH"].mean()),
        0.01,
        label_visibility="visible",
        key="pH",
    )
    sulphates = st.slider(
        "Sulphates",
        float(X["sulphates"].min()),
        float(X["sulphates"].max()),
        float(X["sulphates"].mean()),
        0.01,
        label_visibility="visible",
        key="sulphates",
    )
    alcohol = st.slider(
        "Alcohol",
        float(X["alcohol"].min()),
        float(X["alcohol"].max()),
        float(X["alcohol"].mean()),
        0.01,
        label_visibility="visible",
        key="alcohol",
    )

# Predict button
predict = st.button("Predict")

# Display prediction result
if predict:
    try:
        features = np.array(
            [
                fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                pH,
                sulphates,
                alcohol,
            ]
        ).reshape(1, -1)
        prediction = model.predict(features)

        if prediction == 1:
            st.success("The wine is good!")
        else:
            st.error("The wine is not good.")
    except ValueError:
        st.error("Please adjust the sliders to valid values.")
