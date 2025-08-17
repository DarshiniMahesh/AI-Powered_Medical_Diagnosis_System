# AI-Powered_Medical_Diagnosis_System  

The **AI-Powered_Medical_Diagnosis_System** is a machine learning-powered web application built using Streamlit. It enables users to assess the potential risk of **Diabetes**, **Hypertension**, and **Heart Disease** by entering basic clinical data. This tool promotes early detection and health awareness through accessible, user-friendly analytics.

---

## Problem Statement

Chronic diseases such as diabetes, hypertension, and cardiovascular conditions often remain undiagnosed until significant damage occurs. With preventive health screening becoming increasingly important, this project provides an interactive platform that leverages data-driven insights to support early risk identification, self-awareness, and improved health literacy.

---

## Project Structure

```

multiple-disease-predictor/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Required libraries
├── README.md                   # Project documentation

├── data/                       # Health datasets
│   ├── diabetes.csv
│   ├── heart.csv
│   └── hypertension.csv

├── models/                     # Pretrained model files
│   ├── diabetes\_model.pkl
│   ├── heart\_model.pkl
│   └── hypertension\_model.pkl

├── train\_diabetes\_model.ipynb       # Training notebooks
├── train\_heart\_model.ipynb
└── train\_hypertension\_model.ipynb

````

---

## Key Features

### 1. Multi-Disease Prediction

Predicts the risk of three common chronic conditions—diabetes, hypertension, and heart disease—using standard health indicators. Each disease model is trained independently using public medical datasets.

### 2. Personalized User Input Interface

Provides clearly labeled forms for entering clinical data such as glucose levels, blood pressure, BMI, and other parameters, making the tool accessible to non-technical users.

### 3. Visual Comparison with Healthy Baseline

Generates informative visualizations that compare the user’s input values against medically accepted healthy ranges, offering intuitive feedback on each parameter's risk contribution.

### 4. Adaptive Feedback System

Offers general wellness suggestions and highlights the type of specialist to consult when risk is identified. This ensures the output is both informative and actionable.

### 5. Modular and Extensible Architecture

Each prediction logic is encapsulated within separate components, making the application easy to maintain and scalable to include more diseases or risk factors in the future.

### 6. Modern Streamlit Interface

The application features a clean, dark-themed UI with sidebar navigation and responsive layout, ensuring clarity, usability, and smooth interaction across devices.

### 7. Lightweight Local Deployment

Designed to run efficiently on local systems. Once executed, the app can be accessed on:

- Localhost: `http://localhost:8502`
- Network URL: `http://192.168.106.193:8502` *(accessible by devices on the same local network)*

---

## Technologies Used

- Python 3.8+
- Streamlit
- scikit-learn
- matplotlib
- numpy
- streamlit-option-menu

---

## Installation  

To install and run the project locally:

```bash
# Clone the repository
git clone https://github.com/AI-Powered_Medical_Diagnosis_System.git
cd AI-Powered_Medical_Diagnosis_System

# Install required libraries
pip install -r requirements.txt
````

---

## How to Run

After installing the dependencies, start the application using:

```bash
python -m streamlit run app.py
```

Visit the browser link displayed in the terminal (e.g., `http://localhost:8502`) to begin using the app.

---

## How It Works

1. **Input Collection:**
   Users enter relevant health metrics such as glucose level, blood pressure, cholesterol, BMI, and age through an interactive form.

2. **Preprocessing:**
   The app formats the inputs into a structured feature vector aligned with the model’s training data requirements.

3. **Prediction:**
   Pre-trained machine learning models (one per disease) analyze the input data and predict the likelihood of diabetes, hypertension, or heart disease.

4. **Visualization:**
   The app displays a comparison chart between user values and healthy baseline values for better interpretability.

5. **Results & Recommendations:**
   Based on the prediction, users receive an outcome message along with health tips and medical consultation suggestions.

---

## Deployment on Streamlit Cloud

To deploy your version on Streamlit Cloud:

1. Push the project to your GitHub account
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New App"
4. Select your repository and specify `app.py` as the entry point
5. Deploy and share the link

## Live Demo  

Try out the deployed application here:  
AI-Powered Medical Diagnosis System - https://diagnosify-app.streamlit.app/ 

---

## Disclaimer

This application is intended strictly for educational and demonstration purposes. It is not a medical device, nor should it be used as a substitute for professional diagnosis or treatment. For health-related decisions, always consult a licensed medical professional.

---

## License

This project is licensed under the **MIT License**.  
