import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon="🧑‍⚕️")

# Load pre-trained models & scalers
model_path = r"D:\shadow_knight\Prediction of Disease Outbreak\saved_models"

try:
    diabetes_model = pickle.load(open(os.path.join(model_path, "diabetes_model.sav"), 'rb'))
    diabetes_scaler = pickle.load(open(os.path.join(model_path, "diabetes_scaler.sav"), 'rb'))

    heart_disease_model = pickle.load(open(os.path.join(model_path, "heart_disease_model.sav"), 'rb'))
    heart_disease_scaler = pickle.load(open(os.path.join(model_path, "heart_disease_scaler.sav"), 'rb'))

    parkinsons_model = pickle.load(open(os.path.join(model_path, "parkinsons_model.sav"), 'rb'))
    parkinsons_scaler = pickle.load(open(os.path.join(model_path, "parkinsons_scaler.sav"), 'rb'))

except Exception as e:
    st.error(f"Error loading models: {e}")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Prediction of Disease Outbreaks',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson’s Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ---------------------- Diabetes Prediction ---------------------- #
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', '0')
    with col2:
        Glucose = st.text_input('Glucose Level', '0')
    with col3:
        BloodPressure = st.text_input('Blood Pressure', '0')
    with col1:
        SkinThickness = st.text_input('Skin Thickness', '0')
    with col2:
        Insulin = st.text_input('Insulin Level', '0')
    with col3:
        BMI = st.text_input('BMI', '0')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', '0')
    with col2:
        Age = st.text_input('Age', '0')

    if st.button('Diabetes Test Result'):
        try:
            user_input = np.array([
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]).reshape(1, -1)

            user_input_scaled = diabetes_scaler.transform(user_input)
            diab_prediction = diabetes_model.predict(user_input_scaled)

            st.success('The person is DIABETIC' if diab_prediction[0] == 1 else 'The person is NOT DIABETIC')

        except ValueError:
            st.error("Invalid input! Please enter numeric values.")

# ---------------------- Heart Disease Prediction (Fixed Feature Count) ---------------------- #
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age', '0')
        RestingBP = st.text_input('Resting Blood Pressure', '0')
        Cholesterol = st.text_input('Cholesterol Level', '0')
        MaxHR = st.text_input('Maximum Heart Rate', '0')
        OldPeak = st.text_input('ST Depression (OldPeak)', '0')

    with col2:
        Sex = st.selectbox('Sex (0: Female, 1: Male)', [0, 1])
        FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dL (1: Yes, 0: No)', [0, 1])
        ExerciseAngina = st.selectbox('Exercise Induced Angina (1: Yes, 0: No)', [0, 1])
        Slope = st.selectbox('Slope of ST Segment (0, 1, 2)', [0, 1, 2])

    with col3:
        ChestPainType = st.selectbox('Chest Pain Type (0, 1, 2, 3)', [0, 1, 2, 3])
        RestingECG = st.selectbox('Resting ECG Results (0, 1, 2)', [0, 1, 2])
        CA = st.selectbox('Major Vessels Colored by Fluoroscopy (0-4)', [0, 1, 2, 3, 4])
        Thal = st.selectbox('Thalassemia (0, 1, 2, 3)', [0, 1, 2, 3])

    if st.button('Heart Disease Test Result'):
        try:
            user_input = np.array([
                float(Age.strip()), int(Sex), int(ChestPainType), float(RestingBP.strip()),
                float(Cholesterol.strip()), int(FastingBS), int(RestingECG),
                float(MaxHR.strip()), int(ExerciseAngina), float(OldPeak.strip()),
                int(Slope), int(CA), int(Thal)
            ])

            # Append 10 missing features (placeholder values)
            missing_features = np.zeros(10)  
            user_input = np.concatenate((user_input, missing_features)).reshape(1, -1)

            # Check feature count
            if user_input.shape[1] != 23:
                st.error(f"Feature mismatch! Expected 23 features, but got {user_input.shape[1]}")
            else:
                user_input_scaled = heart_disease_scaler.transform(user_input)
                heart_prediction = heart_disease_model.predict(user_input_scaled)

                st.success('The person has HEART DISEASE' if heart_prediction[0] == 1 else 'The person does NOT have heart disease')

        except ValueError as e:
            st.error(f"Invalid input! Please enter numeric values. Error: {e}")

# ---------------------- Parkinson’s Prediction ---------------------- #
elif selected == 'Parkinson’s Prediction':
    st.title('Parkinson’s Disease Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)

    with col1:
        MDVP_Fo = st.text_input('MDVP: Fo (Hz)', '0')
        MDVP_Fhi = st.text_input('MDVP: Fhi (Hz)', '0')
        MDVP_Flo = st.text_input('MDVP: Flo (Hz)', '0')
    with col2:
        Jitter_Percent = st.text_input('Jitter (%)', '0')
        Shimmer = st.text_input('Shimmer', '0')
        HNR = st.text_input('HNR (Harmonics-to-Noise Ratio)', '0')
    with col3:
        RPDE = st.text_input('RPDE (Recurrence Period Density Entropy)', '0')
        DFA = st.text_input('DFA (Detrended Fluctuation Analysis)', '0')
        PPE = st.text_input('PPE (Pitch Period Entropy)', '0')

    if st.button("Parkinson’s Test Result"):
        try:
            # Convert user input to float safely
            user_input = np.array([
                float(MDVP_Fo.strip()), float(MDVP_Fhi.strip()), float(MDVP_Flo.strip()),
                float(Jitter_Percent.strip()), float(Shimmer.strip()), float(HNR.strip()),
                float(RPDE.strip()), float(DFA.strip()), float(PPE.strip())
            ])

            # Append 14 missing features as placeholder values (zeros)
            missing_features = np.zeros(14)  
            user_input = np.concatenate((user_input, missing_features)).reshape(1, -1)

            # Check feature count before scaling
            if user_input.shape[1] != parkinsons_scaler.n_features_in_:
                st.error(f"Feature mismatch! Model expects {parkinsons_scaler.n_features_in_} features, but received {user_input.shape[1]}")
            else:
                # Scale user input
                user_input_scaled = parkinsons_scaler.transform(user_input)

                # Model prediction
                parkinsons_prediction = parkinsons_model.predict(user_input_scaled)

                # Display result
                st.success('The person has PARKINSON’S DISEASE' if parkinsons_prediction[0] == 1 else 'The person does NOT have Parkinson’s disease')

        except ValueError as e:
            st.error(f"Invalid input! Please enter numeric values. Error: {e}")