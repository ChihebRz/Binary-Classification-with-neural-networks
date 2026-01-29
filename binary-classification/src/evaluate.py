import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import load_data
from model import SpineClassifier


# -----------------------
# Load data
# -----------------------
X_train, X_test, y_train, y_test = load_data("../data/column_2C_weka.csv")

# -----------------------
# Load model
# -----------------------
model = SpineClassifier()
model.load_state_dict(torch.load("../artifacts/model.pth", map_location="cpu"))
model.eval()

# -----------------------
# Prediction
# -----------------------
with torch.inference_mode():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    y_pred = torch.round(probs)

# Convert tensors → numpy
y_true = y_test.numpy().ravel()
y_pred = y_pred.numpy().ravel()

# -----------------------
# Metrics
# -----------------------
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

print("📊 Evaluation Metrics")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------
# Full report (optional but powerful)
# -----------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))
