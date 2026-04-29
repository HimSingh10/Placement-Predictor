import streamlit as st
import pandas as pd
import pickle
import os

# Page config
st.set_page_config(page_title="Placement Predictor", layout="centered")

# Load model safely
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("best_model.pkl"):
            return None
        return pickle.load(open("best_model.pkl", "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Title
st.title("🎓 Engineering Placement Predictor")

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
    if model is None:
        st.error("Model not found. Please upload best_model.pkl")
    else:
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

        try:
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.success(" Student is likely to be Placed!")
            else:
                st.error(" Student is Not Likely to be Placed")

        except Exception as e:
            st.error(f"Prediction error: {e}")


#  MODEL PERFORMANCE

st.subheader("Model Performance")

# Metrics
if os.path.exists("metrics.txt"):
    with open("metrics.txt") as f:
        st.text(f.read())
else:
    st.warning("⚠️ Metrics not available. Run training first.")

# Confusion Matrix
if os.path.exists("confusion_matrix.png"):
    st.image("confusion_matrix.png", caption="Confusion Matrix")
else:
    st.warning("Confusion matrix not available.")

# FEATURE IMPORTANCE

st.subheader("Feature Importance")

if os.path.exists("feature_importance.png"):
    st.image("feature_importance.png", caption="Top Important Features")
else:
    st.warning("Feature importance not available.")