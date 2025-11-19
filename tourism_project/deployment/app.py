import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="neuralDash05/tourism-package-prediction", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism package sales prediction app
st.title("Tourism Package Sales Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package.
Please enter the customer and interaction details data below to get a prediction.
""")

# User input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
DurationOfPitch = st.number_input("Duration of Pitch(in minutes)", min_value=0, max_value=120, value=5)
Occupation = st.selectbox("Customer's occupation (e.g., Salaried, Freelancer)", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
Gender = st.selectbox("Gender of the customer (Male, Female)", ["Female", "Male"])
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch", min_value=0, max_value=10, value=2)
ProductPitched = st.selectbox("The type of product pitched to the customer", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
PreferredPropertyStar = st.number_input("Preferred hotel rating by the customer", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital status of the customer (Single, Married, Divorced)", ["Divorced", "Married", "Single", "Unmarried"])
NumberOfTrips = st.number_input("Average number of trips the customer takes annually", min_value=1, max_value=500, value=2)
Passport = st.selectbox("Whether the customer holds a valid passport (0: No, 1: Yes)", ["0", "1"])
PitchSatisfactionScore = st.number_input("Score indicating the customer's satisfaction with the sales pitch", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Whether the customer owns a car (0: No, 1: Yes)", ["0", "1"])
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer", min_value=0, max_value=10, value=2)
Designation = st.selectbox("Customer's designation in their current organization", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Gross monthly income of the customer", min_value=0, max_value=1000000, value=50000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Product Sale"):
    prediction = model.predict(input_data)[0]
    result = "Will Buy" if prediction == 1 else "Will Not Buy"
    st.subheader("Product Sale Prediction Result:")
    st.success(f"The model predicts: Customer **{result}**")
