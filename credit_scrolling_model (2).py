import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, roc_curve
)

# 1. Generate Synthetic Financial Dataset
print("--- Task 1: Generating Financial Data ---")
np.random.seed(42)
n_samples = 1000

data = {
    'income': np.random.normal(55000, 15000, n_samples),
    'debts': np.random.normal(15000, 8000, n_samples),
    'payment_history_score': np.random.randint(300, 850, n_samples),
    'age': np.random.randint(21, 70, n_samples),
    'utilization_ratio': np.random.uniform(0.0, 1.0, n_samples)
}
df = pd.DataFrame(data)

# Feature Engineering: Debt-to-income ratio
df['debt_to_income'] = df['debts'] / (df['income'] + 1)

# Generate a synthetic ground truth target (1 = Creditworthy, 0 = Default Risk)
# A simple rule-based threshold with noise to mimic real data behavior
logits = (df['payment_history_score'] * 0.01) - (df['utilization_ratio'] * 5) - (df['debt_to_income'] * 2)
probabilities = 1 / (1 + np.exp(-logits))
df['creditworthy'] = (probabilities > 0.5).astype(int)

# 2. Data Splitting & Preprocessing
X = df.drop(columns=['creditworthy'])
y = df['creditworthy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training (Random Forest)
print("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Predictions & Accuracy Assessment
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Model Evaluation Summary ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Plotting ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()