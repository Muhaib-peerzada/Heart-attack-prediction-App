import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

# Load the CSV data
df = pd.read_csv('heart_attack_data.csv')

# Define Features (X) and Target (y)
X = df.drop('heart_attack', axis=1)
y = df['heart_attack']

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Train Logistic Regression
log_model = LogisticRegression(random_state=42, class_weight='balanced')
log_model.fit(X_train_res, y_train_res)

# Train Random Forest with Hyperparameter Tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)

# Save the trained models
joblib.dump(log_model, 'logistic_regression_model.pkl')
joblib.dump(rf_model.best_estimator_, 'random_forest_model.pkl')

# Ask the user which model they want to use
print("\nSelect the model for prediction:")
print("1. Logistic Regression")
print("2. Random Forest")
choice = input("Enter 1 or 2: ")

if choice == "1":
    selected_model = log_model
    model_name = "Logistic Regression"
elif choice == "2":
    selected_model = rf_model.best_estimator_
    model_name = "Random Forest"
else:
    print("Invalid choice. Defaulting to Logistic Regression.")
    selected_model = log_model
    model_name = "Logistic Regression"

# Predict probabilities and adjust threshold
y_probs = selected_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.30
y_pred = (y_probs >= threshold).astype(int)

# Evaluation
print(f"\n==== {model_name} ====")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.3f}")

