import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Try to import XGBoost, fall back to another ensemble if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier

# 1. Generate Synthetic Medical Dataset (Simulating Patient Data)
print("--- Task 4: Generating Structured Medical Data ---")
np.random.seed(42)
n_patients = 1200

data = {
    'age': np.random.randint(18, 85, n_patients),
    'blood_pressure_systolic': np.random.normal(120, 15, n_patients),
    'cholesterol': np.random.normal(200, 40, n_patients),
    'blood_glucose': np.random.normal(100, 25, n_patients),
    'symptom_severity': np.random.randint(0, 4, n_patients), # 0: None, 3: Severe
    'family_history': np.random.choice([0, 1], size=n_patients, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Define disease probability logic based on symptoms and medical stats
# High glucose, high blood pressure, older age, and family history increase risk
logits = (
    (df['age'] * 0.03) + 
    ((df['blood_glucose'] - 100) * 0.04) + 
    ((df['blood_pressure_systolic'] - 120) * 0.02) + 
    (df['symptom_severity'] * 0.6) + 
    (df['family_history'] * 1.2) - 4.5
)
probabilities = 1 / (1 + np.exp(-logits))
df['disease_present'] = (probabilities > 0.5).astype(int)

# 2. Split Features and Target
X = df.drop(columns=['disease_present'])
y = df['disease_present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Data Preprocessing (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Models
print("\nTraining Classifiers...")

# Model A: Random Forest
rf_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

# Model B: XGBoost (or Gradient Boosting alternative)
if XGBOOST_AVAILABLE:
    print("Using XGBoost Classifier...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
else:
    print("XGBoost library not found. Falling back to Scikit-Learn's GradientBoostingClassifier...")
    xgb_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)

xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

# 5. Evaluate Performance
print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, rf_probs):.4f}")
print(classification_report(y_test, rf_preds))

print("=== XGBoost/Gradient Boosting Results ===")
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, xgb_probs):.4f}")
print(classification_report(y_test, xgb_preds))

# 6. Feature Importance Visualization
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.title("Feature Importances for Disease Prediction (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], color="crimson", align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=30)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()