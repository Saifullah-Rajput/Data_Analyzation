
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Math Score Predictor", page_icon="📐", layout="centered")

# Header!

st.markdown("<h1 style='text-align: center;'>📐 Predict Math Scores from Reading & Writing scores</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter reading and writing scores to estimate a student's math score.</p>", unsafe_allow_html=True)

# Bar for selection!

st.sidebar.header("📊 Input Student Scores")
reading = st.sidebar.slider("Reading Score", 0, 100, 70)
writing = st.sidebar.slider("Writing Score", 0, 100, 70)

# loading data 
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    df = df[['math score', 'reading score', 'writing score']]
    return df

df = load_data()
X = df[['reading score', 'writing score']]
y = df['math score']

# Training Of Model
model = LinearRegression()
model.fit(X, y)

# Predicting and evaluating 
input_data = np.array([[reading, writing]])
predicted_math = model.predict(input_data)[0]


st.markdown("---")
st.subheader("📈 Prediction Result")
col1, col2, col3 = st.columns(3)

col1.metric("📖 Reading Score", f"{reading}")
col2.metric("✍️ Writing Score", f"{writing}")
col3.metric("🧮 Predicted Math Score", f"{predicted_math:.2f}")


# Footer!
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit | By Saifullah Rajput</p>",
    unsafe_allow_html=True
)
