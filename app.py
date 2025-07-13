import streamlit as st
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load ML Models
MODEL_DIR = 'models'
def load_model(filename):
    try:
        with open(os.path.join(MODEL_DIR, filename), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model '{filename}' not found in models folder.")
        st.stop()

diabetes_model = load_model('diabetes_model.pkl')
hypertension_model = load_model('hypertension_model.pkl')
heart_model = load_model('heart_model.pkl')

# Apply custom dark theme CSS
def add_bg():
    st.markdown(
        """
        <style>
        html, body, .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .main {
            background-color: #2c2f33;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(255,255,255,0.1);
        }
        label, .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
            color: white !important;
            background-color: #2c2f33 !important;
        }
        .stButton>button {
            background-color: #7289da;
            color: white;
            border-radius: 8px;
            padding: 0.4rem 0.8rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #99aab5;
        }
        .stSidebar {
            background-color: #23272a;
        }
        .stSelectbox label, .stNumberInput label, .stTextInput label {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()

# Sidebar navigation
with st.sidebar:
    selected = option_menu("🔬 Disease Predictor",
        ["Home", "Diabetes", "Hypertension", "Heart Disease"],
        icons=["house", "droplet", "thermometer-high", "heart-pulse"],
        default_index=0)

# Prediction result with chart
def predict_result(model, features, label_pos, feature_labels, healthy_vals):
    # Check for shape match
    if hasattr(model, 'n_features_in_'):
        if len(features) != model.n_features_in_:
            st.error(f"🚫 Feature shape mismatch! Model expects {model.n_features_in_} features, but got {len(features)}.")
            return

    prediction = model.predict([features])[0]
    if prediction == 1:
        st.error(f"⚠️ {label_pos} likely. Please consult your doctor.")
    else:
        st.success("✅ No significant risk detected based on current inputs.")

    fig, ax = plt.subplots(figsize=(10, 4))
    index = np.arange(len(feature_labels))
    ax.plot(index, healthy_vals, label='Healthy', color='lime', marker='o', linewidth=2)
    ax.plot(index, features, label='You', color='red', linestyle='--', marker='x', linewidth=2)
    ax.set_xticks(index)
    ax.set_xticklabels(feature_labels, rotation=45, color='white')
    ax.set_ylabel("Values", color='white')
    ax.set_title("Comparison with Healthy Baseline", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor="#2c2f33", edgecolor='white')
    fig.patch.set_facecolor('#2c2f33')
    ax.set_facecolor('#2c2f33')
    st.pyplot(fig)

# Home Page
if selected == "Home":
    st.markdown(
        """
        <style>
        .home-container {
            margin-top: 2rem;
            display: flex;
            justify-content: space-between;
            gap: 32px;
            flex-wrap: nowrap;
        }
        .disease-card {
            flex: 1 1 auto;
            width: 100%;
            background-color: #2c2f33;
            border: 1px solid #444;
            border-radius: 16px;
            padding: 1.8rem;
            transition: 0.3s ease;
        }
        .disease-card:nth-child(1) {
            min-height: 420px; /* Stretch Diabetes card */
        }
        .disease-card:hover {
            background-color: #3a3f47;
            transform: scale(1.02);
            border-color: #7289da;
        }
        .disease-title {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: #7289da;
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 1rem;
            white-space: nowrap;
        }
        .disease-text {
            color: #ffffff;
            font-size: 1rem;
            line-height: 1.7;
        }
        .disease-text p {
            margin: 0.4rem 0 0.8rem 0;
        }
        .disease-text strong {
            font-family: 'Segoe UI Semibold', sans-serif;
            color: #00bcd4;
        }
        @media screen and (max-width: 1200px) {
            .home-container {
                flex-wrap: wrap;
            }
            .disease-card {
                flex: 1 1 100%;
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='color:#7289da;'>🩺 Your Smart Health Predictor</h1>", unsafe_allow_html=True)
    st.write("Stay one step ahead of illness with intelligent, personalized health predictions.")

    st.markdown("""
    <div class="home-container">
        <div class="disease-card">
            <div class="disease-title">🩸 <span>Diabetes</span></div>
            <div class="disease-text">
                <p>A long-term condition that disrupts how your body regulates blood sugar and insulin.</p>
                <p><strong>Causes:</strong> Insulin resistance, unhealthy diet, genetic factors</p>
                <p><strong>Risks:</strong> Vision loss, nerve damage, kidney dysfunction</p>
                <p><strong>Precautions:</strong> Balanced eating, regular exercise, blood sugar tracking</p>
                <p><strong>Consultation:</strong> Endocrinologist</p>
            </div>
        </div>
        <div class="disease-card">
            <div class="disease-title">💢 <span>Hypertension</span></div>
            <div class="disease-text">
                <p>A silent condition where blood pressure stays high, placing strain on your cardiovascular system.</p>
                <p><strong>Causes:</strong> High sodium diet, obesity, chronic stress</p>
                <p><strong>Risks:</strong> Stroke, aneurysms, heart attack</p>
                <p><strong>Precautions:</strong> Low-salt meals, stress relief, routine BP checks</p>
                <p><strong>Consultation:</strong> Cardiologist / General Practitioner</p>
            </div>
        </div>
        <div class="disease-card">
            <div class="disease-title">❤️ <span>Heart&nbsp;Disease</span></div>
            <div class="disease-text">
                <p>A broad range of heart-related issues that interfere with normal cardiac function.</p>
                <p><strong>Causes:</strong> Smoking, elevated cholesterol, sedentary habits</p>
                <p><strong>Risks:</strong> Arrhythmia, cardiac arrest, congestive heart failure</p>
                <p><strong>Precautions:</strong> Heart-healthy diet, regular movement, quit smoking</p>
                <p><strong>Consultation:</strong> Cardiologist</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Diabetes Page
elif selected == "Diabetes":
    st.header("🧪 Diabetes Risk Prediction")

    st.markdown("""
    **It Requires:**  
    🩸 Fasting Blood Sugar (FBS), 🍽️ Oral Glucose Tolerance Test (OGTT), 💉 Hemoglobin A1c (HbA1c),  
    ⚖️ Body Mass Index (BMI), 🧬 Family Medical History  
    These assessments help understand how your body processes sugar and stores energy.
    """)

    st.markdown("Fill out your health information below:")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("👶 Pregnancies", 0, 20, 1)
        bp = st.number_input("💓 Blood Pressure", 0, 200, 70)
        insulin = st.number_input("💉 Insulin", 0, 900, 85)
        diabetes_func = st.number_input("📊 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    with col2:
        glucose = st.number_input("🩸 Glucose", 0, 300, 110)
        skin_thick = st.number_input("🧫 Skin Thickness", 0, 100, 20)
        bmi = st.number_input("⚖️ BMI", 0.0, 70.0, 20.0)
        age = st.number_input("🎂 Age", 10, 100, 25)

    if st.button("Predict Diabetes"):
        features = [pregnancies, glucose, bp, skin_thick, insulin, bmi, diabetes_func, age]
        labels = ["Pregnancies", "Glucose", "BP", "Skin", "Insulin", "BMI", "Function", "Age"]
        healthy_vals = [0, 110, 70, 20, 85, 20.0, 0.5, 25]

        prediction = diabetes_model.predict([features])[0]

        if prediction == 1:
            st.error("⚠️ Diabetes likely. Please consult your doctor.")
        else:
            st.success("✅ No significant risk detected based on current inputs.")

        # Inject CSS
        st.markdown("""
        <style>
        .result-section {
            display: flex;
            gap: 2rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        .chart-box {
            flex: 2;
            min-width: 300px;
        }
        .tips-box {
            flex: 1;
            background-color: #2c2f33;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            color: white;
        }
        .tips-box h4 {
            color: #7289da;
            margin-bottom: 0.8rem;
        }
        .tips-box ul {
            padding-left: 1.2rem;
        }
        .tips-box li {
            margin-bottom: 0.6rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Result Section
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        # Chart Box
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        index = np.arange(len(labels))
        ax.plot(index, healthy_vals, label='Healthy', color='lime', marker='o', linewidth=2)
        ax.plot(index, features, label='You', color='red', linestyle='--', marker='x', linewidth=2)
        ax.set_xticks(index)
        ax.set_xticklabels(labels, rotation=45, color='white')
        ax.set_ylabel("Values", color='white')
        ax.set_title("Comparison with Healthy Baseline", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor="#2c2f33", edgecolor='white')
        fig.patch.set_facecolor('#2c2f33')
        ax.set_facecolor('#2c2f33')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tips Box
        st.markdown("""
        <div class="tips-box">
        <h4>🌿 General Wellness Tips</h4>
        <ul>
            <li>🥗 <strong>Eat whole & colorful meals:</strong> Add veggies, fruits & whole grains daily.</li>
            <li>💧 <strong>Hydrate consistently:</strong> Water supports digestion & energy balance.</li>
            <li>🧘 <strong>Stress less, breathe more:</strong> Use meditation, journaling, or music therapy.</li>
            <li>🏃 <strong>Move your body:</strong> Even light walks can help balance blood sugar.</li>
            <li>😴 <strong>Sleep is self-care:</strong> Aim for 7–8 hours to reset and repair.</li>
        </ul>

        <div style="margin-top: 1.5rem;">
            <h4>👨‍⚕️ Who to Consult</h4>
            <ul>
                <li>💡 <strong>Primary Recommendation:</strong> Endocrinologist — expert in hormonal and metabolic disorders including diabetes.</li>
                <li>🩺 <strong>Alternate Recommendation:</strong> Diabetologist or General Physician — for routine management and lifestyle guidance.</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # closes .result-section

            
# Hypertension Page
elif selected == "Hypertension":
    st.header("💢 Hypertension Risk Prediction")

    st.markdown("""
    **It Requires:**  
    🩺 Systolic & Diastolic Blood Pressure, 💓 Heart Rate, ⚖️ Body Mass Index (BMI),  
    🚬 Lifestyle Factors (Smoking & Activity), 📁 Medical History  
    These help evaluate your cardiovascular strain and overall blood pressure trends.
    """)

    st.markdown("Fill out your health information below:")

    gender_map = {"Female": 0, "Male": 1}

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Age", 10, 100, 25)
        bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 21.5)
        sys_bp = st.number_input("🔺 Systolic BP", 80, 200, 115)
        dia_bp = st.number_input("🔻 Diastolic BP", 50, 150, 75)
        heart_rate = st.number_input("💓 Heart Rate", 50, 200, 72)
    with col2:
        gender = st.selectbox("🧑 Gender", list(gender_map.keys()))
        gender = gender_map[gender]
        med_history = st.selectbox("📁 Medical History (1=Yes)", [0, 1])
        smoking = st.selectbox("🚬 Smoking", [0, 1])
        sporting = st.selectbox("🏃 Physically Active", [0, 1])
        hypertension_tests = st.selectbox("🧪 Hypertension Tests Done? (1=Yes)", [0, 1])

    if st.button("Predict Hypertension"):
        features = [
            age, gender, bmi, sys_bp, dia_bp,
            heart_rate, med_history, smoking, sporting, hypertension_tests
        ]
        labels = [
            "Age", "Gender", "BMI", "SysBP", "DiaBP",
            "HeartRate", "History", "Smoking", "Active", "Tests"
        ]
        healthy_vals = [25, 1, 21.5, 115, 75, 72, 0, 0, 1, 1]

        prediction = hypertension_model.predict([features])[0]

        if prediction == 1:
            st.error("⚠️ Hypertension likely. Please consult your doctor.")
        else:
            st.success("✅ No significant risk detected based on current inputs.")

        # Inject CSS
        st.markdown("""
        <style>
        .result-section {
            display: flex;
            gap: 2rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        .chart-box {
            flex: 2;
            min-width: 300px;
        }
        .tips-box {
            flex: 1;
            background-color: #2c2f33;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            color: white;
        }
        .tips-box h4 {
            color: #7289da;
            margin-bottom: 0.8rem;
        }
        .tips-box ul {
            padding-left: 1.2rem;
        }
        .tips-box li {
            margin-bottom: 0.6rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Result Section
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        # Chart Box
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        index = np.arange(len(labels))
        ax.plot(index, healthy_vals, label='Healthy', color='lime', marker='o', linewidth=2)
        ax.plot(index, features, label='You', color='red', linestyle='--', marker='x', linewidth=2)
        ax.set_xticks(index)
        ax.set_xticklabels(labels, rotation=45, color='white')
        ax.set_ylabel("Values", color='white')
        ax.set_title("Comparison with Healthy Baseline", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor="#2c2f33", edgecolor='white')
        fig.patch.set_facecolor('#2c2f33')
        ax.set_facecolor('#2c2f33')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tips Box
        st.markdown("""
        <div class="tips-box">
        <h4>🍏 General Wellness Tips</h4>
        <ul>
            <li>🍌 <strong>Eat potassium-rich foods:</strong> Bananas, greens, and lentils support healthy pressure.</li>
            <li>🏃 <strong>Stay active:</strong> Aim for 30 minutes of walking or light cardio daily.</li>
            <li>🧘 <strong>Manage stress gently:</strong> Try mindful breathing, music, or nature time.</li>
            <li>💧 <strong>Stay hydrated:</strong> Balanced fluids help regulate pressure and circulation.</li>
            <li>🩺 <strong>Monitor regularly:</strong> Know your numbers and get routine checkups.</li>
        </ul>

        <div style="margin-top: 1.5rem;">
            <h4>👨‍⚕️ Who to Consult</h4>
            <ul>
                <li>💡 <strong>Primary Recommendation:</strong> Cardiologist — for advanced evaluation and blood pressure management.</li>
                <li>🩺 <strong>Alternate Recommendation:</strong> Internal Medicine Specialist or General Physician — for ongoing monitoring and medication adjustments.</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # closes .result-section


# Heart Disease Page  
elif selected == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    st.markdown("""
    **It Requires:**  
    🧪 ECG or EKG, 🩸 Cholesterol & Blood Pressure readings, 🧬 Family History, 🍬 Fasting Blood Sugar,  
    🏃 Stress Test & Physical Activity Assessment  
    These help evaluate your cardiac function, rhythm patterns, and risk profile.
    """)

    st.markdown("Fill out the fields below:")

    sex_map = {"Female": 0, "Male": 1}
    slope_map = {"Up": 0, "Flat": 1, "Down": 2}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
    ecg_map = {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2}
    angina_map = {"No": 0, "Yes": 1}

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Age", 10, 100, 30)
        cp = st.selectbox("💥 Chest Pain Type", list(cp_map.keys()))
        trestbps = st.number_input("🔴 Resting BP", 80, 200, 120)
        ecg = st.selectbox("📉 Resting ECG", list(ecg_map.keys()))
        exang = st.selectbox("🏋️ Exercise Induced Angina", list(angina_map.keys()))
        slope = st.selectbox("📈 ST Slope", list(slope_map.keys()))
    with col2:
        sex = st.selectbox("🧑 Sex", list(sex_map.keys()))
        chol = st.number_input("🥩 Cholesterol", 100, 400, 180)
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120", [0, 1])
        thalach = st.number_input("🏃 Max Heart Rate", 70, 220, 160)
        oldpeak = st.number_input("📉 Oldpeak", 0.0, 6.0, 0.0)

    if st.button("Predict Heart Disease"):
        features = [
            age, sex_map[sex], cp_map[cp], trestbps, chol, fbs,
            ecg_map[ecg], thalach, angina_map[exang], oldpeak,
            slope_map[slope]
        ]
        labels = ["Age", "Sex", "CP", "BP", "Chol", "FBS", "ECG", "MaxHR", "Angina", "Oldpeak", "Slope"]
        healthy_vals = [30, 1, 0, 120, 180, 0, 0, 160, 0, 0.0, 1]

        prediction = heart_model.predict([features])[0]

        if prediction == 1:
            st.error("⚠️ Heart Disease likely. Please consult your doctor.")
        else:
            st.success("✅ No significant risk detected based on current inputs.")

        # Inject CSS
        st.markdown("""
        <style>
        .result-section {
            display: flex;
            gap: 2rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        .chart-box {
            flex: 2;
            min-width: 300px;
        }
        .tips-box {
            flex: 1;
            background-color: #2c2f33;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            color: white;
        }
        .tips-box h4 {
            color: #7289da;
            margin-bottom: 0.8rem;
        }
        .tips-box ul {
            padding-left: 1.2rem;
        }
        .tips-box li {
            margin-bottom: 0.6rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Result Section
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        # Chart Box
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(8, 4))
        index = np.arange(len(labels))
        ax.plot(index, healthy_vals, label='Healthy', color='lime', marker='o', linewidth=2)
        ax.plot(index, features, label='You', color='red', linestyle='--', marker='x', linewidth=2)
        ax.set_xticks(index)
        ax.set_xticklabels(labels, rotation=45, color='white')
        ax.set_ylabel("Values", color='white')
        ax.set_title("Comparison with Healthy Baseline", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor="#2c2f33", edgecolor='white')
        fig.patch.set_facecolor('#2c2f33')
        ax.set_facecolor('#2c2f33')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tips Box
        st.markdown("""
        <div class="tips-box">
        <h4>🍏 General Wellness Tips</h4>
        <ul>
            <li>🥑 <strong>Eat potassium-rich foods:</strong> Bananas, greens, and lentils support healthy pressure.</li>
            <li>🏃‍♀️ <strong>Stay active:</strong> Aim for 30 minutes of walking or light cardio daily.</li>
            <li>🧘‍♂️ <strong>Manage stress gently:</strong> Try mindful breathing, music, or nature time.</li>
            <li>💧 <strong>Stay hydrated:</strong> Balanced fluids help regulate pressure and circulation.</li>
        </ul>

        <div style="margin-top: 1.5rem;">
            <h4>👨‍⚕️ Who to Consult</h4>
            <ul>
                <li>💡 <strong>Primary:</strong> Cardiologist — specializes in heart conditions and can provide targeted treatment.</li>
                <li>🩺 <strong>Alternate:</strong> General Physician — for initial screening and referral to a specialist if needed.</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # closes .result-section
