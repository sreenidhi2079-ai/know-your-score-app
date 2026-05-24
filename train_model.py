# ==============================
# 📌 STEP 1: IMPORT LIBRARIES
# ==============================
import pandas as pd

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


# ==============================
# 📌 STEP 2: LOAD DATASET
# ==============================
df = pd.read_csv("dataset/Smartphone Usage Survey.csv")

# View basic data info
print(df.head())
print(df.info())

# Check missing values
print(df.isnull().sum())


# ==============================
# 📌 STEP 3: FEATURE ENGINEERING
# (Create Addiction Score)
# ==============================

def calculate_score(row):
    score = 0

    # 🔹 Screen time contribution
    if row['Avg_screen_time_hours'] >= 6:
        score += 2
    elif row['Avg_screen_time_hours'] >= 3:
        score += 1

    # 🔹 Phone checks contribution
    if row['Phone_checks_per_day'] >= 40:
        score += 2
    elif row['Phone_checks_per_day'] >= 20:
        score += 1

    # 🔹 Hours without phone (inverse logic)
    if row['Hours_without_phone'] <= 2:
        score += 2
    elif row['Hours_without_phone'] <= 6:
        score += 1

    return score


# Apply scoring function to dataset
df['score'] = df.apply(calculate_score, axis=1)


# ==============================
# 📌 STEP 4: CREATE LABELS
# ==============================

def label(score):
    if score <= 2:
        return "Low"
    elif score <= 4:
        return "Medium"
    else:
        return "High"


df['Addiction_Level'] = df['score'].apply(label)

print(df[['score', 'Addiction_Level']].head(10))


# Save updated dataset (for reference)
df.to_csv("dataset/updated_data.csv", index=False)


# ==============================
# 📌 STEP 5: PREPARE FEATURES & TARGET
# ==============================

features = [
    "Avg_screen_time_hours",
    "Phone_checks_per_day",
    "Hours_without_phone"
]

X = df[features]  # input features
y = df["Addiction_Level"]  # target label


# ==============================
# 📌 STEP 6: LABEL ENCODING
# ==============================

le = LabelEncoder()
y = le.fit_transform(y)


# ==============================
# 📌 STEP 7: TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ==============================
# 📌 STEP 8: TRAIN MODEL
# ==============================

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# ==============================
# 📌 STEP 9: MODEL EVALUATION
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# ==============================
# 📌 STEP 10: SAVE MODEL
# ==============================

joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("Model saved successfully!")