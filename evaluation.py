import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

print("Loading Dataset...")

# Dataset Load
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Total Samples:", len(X))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train/Test Split Completed")

# Train Model
model = LogisticRegression(max_iter=10000)

print("Training Model...")
model.fit(X_train, y_train)

print("Model Training Completed")

# Prediction
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

# Classification Report
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc_score = roc_auc_score(
    y_test,
    y_prob
)

print("\nAUC Score =", round(auc_score, 4))

# ROC Plot
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.4f}",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    "r--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY")
