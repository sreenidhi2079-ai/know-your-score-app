KNOW YOUR SCORE - Smartphone Usage & Addiction Level Prediction System
==============================================================================================================================================================================================
A machine learning-based web application that predicts a user's smartphone addiction level (Low / Medium / High) based on behavioral usage patterns and provides personalized recommendations.
==============================================================================================================================================================================================
Live Demo:
https://know-your-score-app.vercel.app/
==============================================================================================================================================================================================
PROBLEM STATEMENT

Excessive smartphone usage has emerged as a significant behavioral concern in modern society, impacting productivity, sleep quality, academic performance, and overall mental well-being. With the increasing dependence on digital devices for communication, entertainment, and daily tasks, it has become difficult for individuals to self-regulate their usage patterns.

This project focuses on analyzing user smartphone usage behavior based on various lifestyle and interaction-based features such as screen time, frequency of phone checks, usage habits, and daily activity patterns. The primary objective is to identify behavioral trends associated with excessive smartphone usage.

Using machine learning techniques, the project aims to build a predictive model that classifies or estimates a user’s level of smartphone addiction. This can help in early identification of at-risk users and provide insights that may support digital well-being initiatives.

Ultimately, the system is intended to transform raw behavioral data into meaningful predictions, enabling better understanding of smartphone dependency and encouraging healthier digital habits.
==============================================================================================================================================================================================
PROPOSED SOLUTION

The proposed solution is a machine learning-based web application titled “Know Your Score”, designed to assess and predict a user’s level of smartphone addiction based on behavioral usage patterns.

The system collects key input features such as average daily screen time, frequency of phone checks, and duration of time spent without using a phone. These features represent user digital behavior and are used as indicators of smartphone dependency.

A trained Decision Tree Classifier model processes the input data and classifies the user into one of three categories: Low, Medium, or High addiction level. This allows the system to convert raw behavioral data into an interpretable and actionable output.

In addition to prediction, the application provides personalized recommendations aimed at improving digital well-being, helping users reduce excessive smartphone usage and build healthier habits.

The solution is implemented using Python, Flask, and Scikit-learn, and deployed as a lightweight interactive web application, making it accessible and user-friendly for real-time predictions.
============================================================================================================================================================================================
Input Features

The model uses the following features:

- Average Screen Time (hours/day)
- Phone Checks Per Day
- Hours Without Phone
============================================================================================================================================================================================
Output Classes

- Low Addiction 
- Medium Addiction 
- High Addiction

AI-based suggestions are generated based on the predicted addiction level and user behavior patterns. They help users understand their smartphone usage habits and identify areas of improvement. The recommendations aim to promote healthier digital balance and reduce excessive screen dependency.
============================================================================================================================================================================================
TECH STACK

# Backend
- Python
- Flask

# Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Decision Tree Classifier

# Frontend
- HTML
- CSS
============================================================================================================================================================================================
# Machine Learning Workflow

1. Data Collection (survey dataset)
2. Data Preprocessing
3. Feature Engineering
4. Label Creation (Low / Medium / High)
5. Model Training (Decision Tree Classifier)
6. Model Evaluation
7. Deployment 

1. Data Collection:

The data was collected through primary sources using a Google Forms survey, where users themselves provided responses related to their smartphone usage behavior. This self-reported data was then compiled to form the dataset for analysis and model building.

2. Data Preprocessing:

The collected data was cleaned and standardized to make it suitable for model training. Labels were converted into a simplified and more readable format. Range-based values were transformed into numerical representations using the mean method to ensure consistency. Additionally, categorical values such as “Yes” were encoded into shorter formats like “Y” for easier processing by the machine learning model.

3. Feature Engineering

In this step, multiple attributes such as user name, gender, and other irrelevant or less impactful details were removed. Only the most relevant features influencing smartphone addiction were selected for model training. These include daily screen time, number of phone checks per day, and hours spent without using the phone, as they strongly contribute to determining the addiction level.

4. Label Creation (Low / Medium / High)

To generate target labels for the dataset, a rule-based scoring system was implemented using Python. Each user was assigned a score based on their behavioral patterns such as average screen time, number of phone checks per day, and hours spent without using the phone.

A weighted scoring function was defined where higher screen time and frequent phone checks increased the addiction score, while more hours spent without the phone reduced dependency. Based on the final computed score, users were categorized into three classes: Low, Medium, and High addiction levels.

This approach helped in converting raw behavioral data into meaningful labeled data suitable for supervised machine learning model training.

5. Model Training (Decision Tree Classifier)

The labeled dataset was used to train a supervised machine learning model using a Decision Tree Classifier. The selected features (screen time, phone checks, and hours without phone) were split into training and testing sets to evaluate model performance.

The Decision Tree algorithm was chosen due to its simplicity, interpretability, and ability to handle both numerical and categorical patterns effectively. The model learns decision rules from the input features to classify users into Low, Medium, or High smartphone addiction levels.

6. Model Evaluation

The trained Decision Tree Classifier was evaluated using the test dataset to measure its performance and reliability. Standard evaluation metrics such as accuracy were used to assess how well the model predicts smartphone addiction levels.

7. Deployment

The Flask-based ML model was deployed using Render for the backend and Vercel for the frontend. Render runs the prediction API, while Vercel hosts the user interface. This setup enables smooth integration and real-time prediction of smartphone addiction levels with AI-based suggestions.
 ===========================================================================================================================================================================================

PROJECT STRUCTURE

KNOW_YOUR_SCORE/
│
├── __pycache__/
│   └── app.cpython-313.pyc
│
├── venv/
│
├── dataset/
│   ├── Smartphone Usage Survey.csv
│   └── updated_data.csv
│
├── model/
│   ├── label_encoder.pkl
│   └── model.pkl
│
├── static/
│   ├── logo.png
│   └── styles.css
│
├── templates/
│   ├── about.html
│   ├── index.html
│   └── predictor.html
│
├── .env
├── .gitignore
├── app.py
├── install-log.txt
├── notes.txt
├── Procfile
├── README.md
├── requirements.txt
├── runtime.txt
└── train_model.py
============================================================================================================================================================================================
FUTURE IMPROVEMENTS

-Add user accounts with authentication system
-Integrate database for storing user data and history
-Add automatic screen lock feature for digital well-being
============================================================================================================================================================================================
Conclusion

This project demonstrates an end-to-end machine learning workflow from data processing to deployment, showcasing practical application of ML in real-world behavioral analysis.
============================================================================================================================================================================================
Author - M V S SREENIDHI
Interested Domains - Machine Learning / Data Science , Artificial Intelligence
============================================================================================================================================================================================

