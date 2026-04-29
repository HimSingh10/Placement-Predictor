import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("best_model.pkl", "rb"))

st.title("Engineering Placement Predictor")

st.write("Enter student details:")

# Inputs
gender = st.selectbox("Gender", ["M", "F"])
ssc_p = st.slider("10th Percentage", 40, 100, 60)
hsc_p = st.slider("12th Percentage", 40, 100, 60)
degree_p = st.slider("Degree Percentage", 40, 100, 60)
workex = st.selectbox("Work Experience", ["Yes", "No"])
etest_p = st.slider("Aptitude Score", 40, 100, 60)
specialisation = st.selectbox("Specialization Type", ["Tech", "Non-Tech"])
mba_p = st.slider("Final Score (%)", 40, 100, 60)

# Prediction
if st.button("Predict Placement"):
    input_data = pd.DataFrame([{
        "gender": gender,
        "ssc_p": ssc_p,
        "hsc_p": hsc_p,
        "degree_p": degree_p,
        "workex": workex,
        "etest_p": etest_p,
        "specialisation": specialisation,
        "mba_p": mba_p
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Student is likely to be Placed!")
    else:
        st.error("Student is Not Likely to be Placed")

# Show Metrics
st.subheader("Model Performance")

try:
    with open("metrics.txt") as f:
        st.text(f.read())
except:
    st.warning("Run training first to generate metrics")

try:
    st.image("confusion_matrix.png", caption="Confusion Matrix")
except:
    st.warning("Confusion matrix not found")

try:
    with open("metrics.txt") as f:
        st.text(f.read())
except:
    st.warning("Run training first to generate metrics")

# Show Confusion Matrix
try:
    st.image("confusion_matrix.png", caption="Confusion Matrix")
except:
    st.warning("Confusion matrix not found")

# Feature Importance Section
st.subheader("Feature Importance")

try:
    st.image("feature_importance.png", caption="Top Important Features")
except:
    st.warning("Feature importance not found. Run training first.")