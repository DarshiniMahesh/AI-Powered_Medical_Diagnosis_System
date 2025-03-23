# diseasePrediction.py
import pickle
import os
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# Define the path to the saved models
MODEL_DIR = 'savedModels'

# Ensure that the 'savedModels' directory exists in the project
if not os.path.exists(MODEL_DIR):
    st.error("Models directory not found!")
    st.stop()

# Loading saved models using relative paths
try:
    diabetesModel = pickle.load(open(os.path.join(MODEL_DIR, 'trainedModel_RF.sav'), 'rb'))
    heartDiseaseModel = pickle.load(open(os.path.join(MODEL_DIR, 'heartTrainedModel.sav'), 'rb'))
    parkinsonModel = pickle.load(open(os.path.join(MODEL_DIR, 'parkinsonTrainedModel.sav'), 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           icons=['house', 'activity', 'heart', 'person'], default_index=0)



# Function to plot the comparison chart
def plot_comparison(healthy_data, user_data, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(len(feature_names))
    bar_width = 0.35
    
    ax.plot(index, healthy_data, label='Healthy Data', color='green', marker='o', linestyle='-', linewidth=2)
    ax.plot(index + bar_width, user_data, label='User Input', color='red', marker='o', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_title('Comparison between Healthy Data and User Input')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(feature_names, rotation=45)
    ax.legend()

    # Show the plot
    st.pyplot(fig)



# Function to check if any input field is empty
def validate_inputs(inputs):
    for value in inputs:
        if isinstance(value, str) and value.strip() == '':
            return False
        elif isinstance(value, (int, float)):
            if value is None or value == '':
                return False
    return True

# Healthy person data (placeholder or pre-filled values)
healthy_data = {
    'diabetes': {
        'pregnancies': 0,
        'glucose': 110,
        'bp': 70,
        'skinThickness': 20,
        'insulin': 85,
        'bmi': 20.0,
        'diabetesFunc': 0.5,
        'age': 25
    },
    'heart': {
        'age': 30,
        'sex': 1,
        'cp': 0,
        'trestbps': 120,
        'chol': 200,
        'fbs': 0,
        'ecg': 0,
        'thalach': 100,
        'exang': 0,
        'oldpeak': 0,
        'slope': 1,
        'ca': 0,
        'thal': 1,
        'target': 0
    },
    'parkinson': {
        'fo': 0.5,
        'flo': 0.5,
        'jitter': 0.01,
        'jitter1': 0.01,
        'rap': 0.02,
        'ppq': 0.03,
        'ddpJitter': 0.02,
        'shimmer': 0.03,
        'shimmer1': 0.02,
        'shimmer2': 0.01,
        'shimmer3': 0.01,
        'shimmer4': 0.02,
        'shimmer5': 0.02,
        'dda': 0.02,
        'nhr': 0.02,
        'hnr': 0.1,
        'status': 1,  # Healthy
        'rpde': 0.02,
        'dfa': 0.02,
        'spread1': 0.02,
        'spread2': 0.02,
        'd2': 0.01,
        'age': 35,
        'ppe': 0.02
    }
}

# Home Page
if selected == 'Home':
    st.title("Welcome to the Multiple Disease Prediction System")
    st.markdown("""
    This system helps you predict the likelihood of the following diseases based on input parameters:
    - **Diabetes**
    - **Heart Disease**
    - **Parkinson's Disease**

    Explore each section to learn more about the diseases, their consequences, causes, effects, precautions, and consultation resources.
    """)

    st.subheader("Disease Information")

    with st.expander("1. Diabetes"):
        st.markdown("""
        **Diabetes** is a chronic disease that affects how your body turns food into energy.  
        - **Consequences:** High blood sugar can lead to serious complications like heart disease, vision loss, and kidney disease.  
        - **Causes:** Insulin resistance, lifestyle choices, and genetics.  
        - **Precautions:** Maintain a healthy diet, exercise regularly, and monitor blood sugar levels.  
        - **Consultation:** Visit an endocrinologist for diagnosis and management.
        """)

    with st.expander("2. Heart Disease"):
        st.markdown("""
        **Heart Disease** refers to several types of heart conditions, including coronary artery disease and heart attack.  
        - **Consequences:** Can lead to heart failure, stroke, or sudden cardiac arrest.  
        - **Causes:** High blood pressure, cholesterol, smoking, and unhealthy diet.  
        - **Precautions:** Avoid smoking, eat a balanced diet, exercise, and manage stress.  
        - **Consultation:** Seek advice from a cardiologist for heart health assessments.
        """)

    with st.expander("3. Parkinson's Disease"):
        st.markdown("""
        **Parkinson's Disease** is a progressive nervous system disorder that affects movement.  
        - **Consequences:** Tremors, stiffness, and difficulty with balance and coordination.  
        - **Causes:** Loss of dopamine-producing brain cells, often linked to genetics and environmental factors.  
        - **Precautions:** Regular exercise, healthy diet, and therapies like physiotherapy.  
        - **Consultation:** Neurologists specialize in diagnosing and treating Parkinson’s disease.
        """)

    st.info("Use the sidebar to navigate to the prediction sections for each disease.")

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction Model')
    st.text("=> It Requires: Blood Tests(Fasting Blood Sugar, Oral Glucose Tolerance Test, Hemoglobin A1c Test), BMI, BP, Family history")
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.text_input('Number of pregnancies', value=healthy_data['diabetes']['pregnancies'])
    with col2:
        glucose = st.text_input('Glucose Level', value=healthy_data['diabetes']['glucose'])
    with col3:
        bp = st.text_input('Blood Pressure (BP)', value=healthy_data['diabetes']['bp'])
    with col1:
        skinThickness = st.text_input('Skin Thickness', value=healthy_data['diabetes']['skinThickness'])
    with col2:
        insulin = st.text_input('Insulin Level', value=healthy_data['diabetes']['insulin'])
    with col3:
        bmi = st.text_input('Body-Mass Index (BMI)', value=healthy_data['diabetes']['bmi'])
    with col1:
        diabetesFunc = st.text_input('Diabetes Pedigree Function value', value=healthy_data['diabetes']['diabetesFunc'])
    with col2:
        age = st.text_input('Age', value=healthy_data['diabetes']['age'])


    # Code for prediction
    dataContainer = [pregnancies, glucose, bp, skinThickness, insulin, bmi, diabetesFunc, age]
    diabetesDiagnosis = ""
    suggestions = ""
    if st.button('Predict the Diabetes Result'):
        if validate_inputs(dataContainer):
            diabetesPrediction = diabetesModel.predict([dataContainer])
            if diabetesPrediction[0] == 1:
                diabetesDiagnosis = 'The person has high chances of having diabetes. Please consult a doctor for further evaluation.'
                suggestions = """
    **Dietary Suggestions to Boost Immunity for Diabetes:**
    - **Leafy Greens:** Include spinach, kale, and other dark leafy greens for high fiber and antioxidants.
    - **Nuts and Seeds:** Almonds, walnuts, and chia seeds provide healthy fats and help regulate blood sugar levels.
    - **Berries:** Blueberries, strawberries, and raspberries are rich in antioxidants and have a low glycemic index.
    - **Whole Grains:** Choose oats, quinoa, and barley for better blood sugar control.
    - **Citrus Fruits:** Oranges, lemons, and grapefruits are rich in Vitamin C, boosting immunity and controlling sugar levels.
    - **Green Tea:** Contains catechins that help improve insulin sensitivity.
    - **Garlic & Ginger:** Known for their anti-inflammatory and immune-boosting properties.
    """
            else:
                diabetesDiagnosis = 'The person is not diabetic'
                suggestions = """
    **General Tips for Immunity and Healthy Living:**
    - Stay hydrated and maintain a balanced diet.
    - Exercise regularly to improve overall health and immunity.
    - Manage stress levels to reduce inflammation.
    """
        st.success(diabetesDiagnosis)
        user_data = [int(pregnancies), int(glucose), int(bp), int(skinThickness), int(insulin), float(bmi), float(diabetesFunc), int(age)]
        healthy_values = list(healthy_data['diabetes'].values())
        feature_names = ['Pregnancies', 'Glucose', 'BP', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Function', 'Age']
        plot_comparison(healthy_values, user_data, feature_names)
        st.markdown(suggestions)

        

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction Model')
    st.text("=> It Requires: Blood Tests, ECG (Electrocardiogram), Stress Test (Exercise Stress Test), Cardiac Imaging, Blood Pressure and Heart Rate Measurements")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', value=healthy_data['heart']['age'])
    with col2:
        sex = st.text_input('Gender', value=healthy_data['heart']['sex'])
    with col3:
        cp = st.text_input('Cerebral Palsy', value=healthy_data['heart']['cp'])
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', value=healthy_data['heart']['trestbps'])
    with col2:
        chol = st.text_input('Cholesterol Level', value=healthy_data['heart']['chol'])
    with col3:
        fbs = st.text_input('Fasting Blood Sugar', value=healthy_data['heart']['fbs'])
    with col1:
        ecg = st.text_input('Electrocardiogram', value=healthy_data['heart']['ecg'])
    with col2:
        thalach = st.text_input('Maximum heart rate achieved', value=healthy_data['heart']['thalach'])
    with col3:
        exang = st.text_input('Exang', value=healthy_data['heart']['exang'])
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise', value=healthy_data['heart']['oldpeak'])
    with col2:
        slope = st.text_input('Slope of Recovery', value=healthy_data['heart']['slope'])
    with col3:
        ca = st.text_input('Cardiac Arrest', value=healthy_data['heart']['ca'])
    with col1:
        thal = st.text_input('Thalassemia', value=healthy_data['heart']['thal'])
    with col2:
        target = st.text_input('Target', value=healthy_data['heart']['target'])

    try:
        age = int(age)
        sex = int(sex)
        cp = int(cp)
        trestbps = int(trestbps)
        chol = int(chol)
        fbs = int(fbs)
        ecg = int(ecg)
        thalach = int(thalach)
        exang = int(exang)
        oldpeak = float(oldpeak)
        slope = int(slope)
        ca = int(ca)
        thal = int(thal)
        target = int(target)
    except ValueError:
        st.error("Please ensure all inputs are valid numeric values.")
        st.stop()




    # Code for prediction
    dataContainer2 = [age, sex, cp, trestbps, chol, fbs, ecg, thalach, exang, oldpeak, slope, ca, thal]
    heartDiagnosis = ""
    suggestions = ""
    if st.button('Predict the Heart Disease Result'):
        if validate_inputs(dataContainer2):
            heartPrediction = heartDiseaseModel.predict([dataContainer2])
            if heartPrediction[0] == 1:
                heartDiagnosis = 'The person has high chances of having heart disease. Please consult a doctor for further evaluation.'
                suggestions = """
    **Dietary Suggestions to Boost Immunity for Heart Disease:**
    - **Oats and Barley:** Whole grains like oats and barley can lower cholesterol levels and support heart health.
    - **Fatty Fish:** Salmon, mackerel, and sardines are rich in Omega-3 fatty acids, which help reduce inflammation and improve heart health.
    - **Leafy Greens:** Kale, spinach, and swiss chard are rich in vitamins and antioxidants, which help improve cardiovascular health.
    - **Nuts:** Walnuts, almonds, and flaxseeds help lower bad cholesterol levels.
    - **Berries:** Blueberries, strawberries, and raspberries are packed with antioxidants that help reduce inflammation and oxidative stress.
    - **Legumes:** Beans, lentils, and peas provide protein and fiber, essential for heart health.
    - **Olive Oil:** Rich in monounsaturated fats, which help lower cholesterol and inflammation.
    """
            else:
                heartDiagnosis = 'The person does not have heart disease'
                suggestions = """
    **General Tips for Immunity and Healthy Living:**
    - Consume more fiber from fruits, vegetables, and whole grains.
    - Regular physical activity helps to strengthen your heart.
    - Avoid smoking and excessive alcohol consumption.
    """
        st.success(heartDiagnosis)
        user_data = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(ecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]
        healthy_values = list(healthy_data['heart'].values())[:-1]
        feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar', 'ECG', 'Max Heart Rate', 'Exercise Induced Angina', 'Oldpeak', 'Slope', 'Number of Major Vessels', 'Thalassemia']
        plot_comparison(healthy_values, user_data, feature_names)
        st.markdown(suggestions)
       

elif selected == 'Parkinsons Prediction':
    st.title('Parkinsons Prediction Model')
    st.text('=> It Requires: Voice and Motor Skills Tests, MRI/CT Scan, Neurological Exam, Blood Tests')
    col1, col2, col3 = st.columns(3)
    with col1:
        fo = st.text_input('Foramen Ovale', value=healthy_data['parkinson']['fo'])
    with col2:
        flo = st.text_input('Fluorouracil-leucovorin-Oxaliplatin (FLO)', value=healthy_data['parkinson']['flo'])
    with col3:
        jitter = st.text_input('MDVP-Jitter(%)', value=healthy_data['parkinson']['jitter'])
    with col1:
        jitter1 = st.text_input('MDVP-Jitter(abs)', value=healthy_data['parkinson']['jitter1'])
    with col2:
        rap = st.text_input('MDVP-RAP', value=healthy_data['parkinson']['rap'])
    with col3:
        ppq = st.text_input('MDVP-PPQ', value=healthy_data['parkinson']['ppq'])
    with col1:
        ddpJitter = st.text_input('DDP', value=healthy_data['parkinson']['ddpJitter'])
    with col2:
        shimmer = st.text_input('MDVP-Shimmer', value=healthy_data['parkinson']['shimmer'])
    with col3:
        shimmer1 = st.text_input('MDVP-Shimmer(db)', value=healthy_data['parkinson']['shimmer1'])
    with col1:
        shimmer2 = st.text_input('APQ3', value=healthy_data['parkinson']['shimmer2'])
    with col2:
        shimmer3 = st.text_input('APQ5', value=healthy_data['parkinson']['shimmer3'])
    with col3:
        shimmer4 = st.text_input('MDVP-APQ', value=healthy_data['parkinson']['shimmer4'])
    with col1:
        shimmer5 = st.text_input('MDVP-Shimmer2', value=healthy_data['parkinson']['shimmer5'])
    with col2:
        dda = st.text_input('DDA', value=healthy_data['parkinson']['dda'])
    with col3:
        nhr = st.text_input('NHR', value=healthy_data['parkinson']['nhr'])
    with col1:
        hnr = st.text_input('HNR', value=healthy_data['parkinson']['hnr'])
    with col2:
        status = st.text_input('Status', value=healthy_data['parkinson']['status'])
    with col3:
        rpde = st.text_input('RPDE', value=healthy_data['parkinson']['rpde'])
    with col1:
        dfa = st.text_input('DFA', value=healthy_data['parkinson']['dfa'])
    with col2:
        spread1 = st.text_input('Spread1', value=healthy_data['parkinson']['spread1'])
    with col3:
        spread2 = st.text_input('Spread2', value=healthy_data['parkinson']['spread2'])
    with col1:
        d2 = st.text_input('D2', value=healthy_data['parkinson']['d2'])
    with col2:
        age = st.text_input('Age', value=healthy_data['parkinson']['age'])
    with col3:
        ppe = st.text_input('PPE', value=healthy_data['parkinson']['ppe'])

    # Code for prediction
    dataContainer3 = [fo, flo, jitter, jitter1, rap, ppq, ddpJitter, shimmer, shimmer1, shimmer2, shimmer3, shimmer4, shimmer5, dda, nhr, hnr, status, rpde, dfa, spread1, spread2, d2]
    parkinsonDiagnosis = ""
    suggestions = ""
    if st.button('Predict the Parkinson Disease Result'):
        if validate_inputs(dataContainer3):
            parkinsonPrediction = parkinsonModel.predict([dataContainer3])
            if parkinsonPrediction[0] == 1:
                parkinsonDiagnosis = "The person has high chances of having Parkinson's disease. Please consult a doctor for further evaluation."
                suggestions = """
    **Dietary Suggestions to Boost Immunity for Parkinson's Disease:**
    - **Leafy Greens and Cruciferous Vegetables:** Vegetables like broccoli, kale, and cabbage are rich in antioxidants and promote brain health.
    - **Omega-3 Fatty Acids:** Fatty fish like salmon and sardines, and plant-based sources like flaxseeds, chia seeds, and walnuts, help reduce inflammation and improve brain function.
    - **Berries:** Blueberries and strawberries are rich in antioxidants that help combat oxidative stress in the brain.
    - **Curcumin:** Found in turmeric, curcumin has anti-inflammatory properties that may reduce brain inflammation and slow Parkinson’s progression.
    - **Caffeine:** Moderate amounts of caffeine from coffee or tea may enhance brain function and alertness.
    - **Vitamin E and C:** Nuts, seeds, and citrus fruits are good sources of Vitamin E and C, which protect brain cells from oxidative damage.
    - **Whole Grains:** Oats, quinoa, and brown rice provide essential nutrients and fiber that support overall health.
    """
            else:
                parkinsonDiagnosis = 'The person does not have Parkinson’s disease'
                suggestions = """
    **General Tips for Brain Health and Immunity:**
    - Stay mentally active with puzzles, learning, and problem-solving.
    - Eat foods rich in antioxidants and healthy fats to support brain health.
    - Engage in regular physical and mental exercises to strengthen neural connections.
    """
        st.success(parkinsonDiagnosis)
        user_data = [float(fo), float(flo), float(jitter), float(jitter1), float(rap), float(ppq), float(ddpJitter), float(shimmer), float(shimmer1), float(shimmer2), float(shimmer3), float(shimmer4), float(shimmer5), float(dda), float(nhr), float(hnr), int(status), float(rpde), float(dfa), float(spread1), float(spread2), float(d2)]
        healthy_values = list(healthy_data['parkinson'].values())[:-2]
        feature_names = ['FO', 'FLO', 'Jitter', 'Jitter1', 'RAP', 'PPQ', 'DDP Jitter', 'Shimmer', 'Shimmer1', 'Shimmer2', 'Shimmer3', 'Shimmer4', 'Shimmer5', 'DDA', 'NHR', 'HNR', 'Status', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2']
        plot_comparison(healthy_values, user_data, feature_names)
        st.markdown(suggestions)
