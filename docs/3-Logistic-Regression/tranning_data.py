import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

dataset = load_dataset('imodels/diabetes-readmission', split="train")
df = dataset.to_pandas()

X_column = [
    "time_in_hospital", 
    "num_lab_procedures", 
    "num_procedures", 
    "num_medications", 
    "number_outpatient", 
    "number_emergency", 
    "number_inpatient", 
    "number_diagnoses"]

Y_column = 'readmitted'

df = df.dropna()

x = df[X_column]
y = df[Y_column]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = [
    "time_in_hospital", 
    "num_lab_procedures", 
    "num_procedures", 
    "num_medications", 
    "number_outpatient", 
    "number_emergency", 
    "number_inpatient", 
    "number_diagnoses"]

prepocessor = ColumnTransformer(
    transformers =[("num", StandardScaler(), numeric_features)]
)

log_reg = Pipeline(
    steps=[
        ("preprocessor", prepocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)
y_proba = log_reg.predict_proba(x_test)[:, 1]

results = pd.DataFrame({
    "true_readmission": y_test.values,
    "predicted_class": y_pred,
    "predicted_probability": y_proba
})

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

print(classification_report(
    y_test,
    y_pred,
    target_names=["No readmitted", "Readmission"]
))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No readmitted", "Readmission"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()

threshold = 0.5

y_pred_threshold = (y_proba >= threshold).astype(int)

print(f"Metrics with threshold = {threshold}")

print(f"Accuracy:  {accuracy_score(y_test, y_pred_threshold):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_threshold):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_threshold):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred_threshold):.3f}")

# Get the trained logistic regression classifier from the pipeline
classifier = log_reg.named_steps["classifier"]

# Transform X_test using the fitted preprocessor
X_test_transformed = log_reg.named_steps["preprocessor"].transform(x_test)

# Compute the linear score z = Xw + b --> this is what decision_function does, not really used in practice
z = classifier.decision_function(X_test_transformed)

# Compute sigmoid
p_manual = 1 / (1 + np.exp(-z))

# Get probabilities from sklearn
p_sklearn = log_reg.predict_proba(x_test)[:, 1]

# Check that they are the same
print("Manual sigmoid probabilities:")
print(p_manual[:10])

print("\nSklearn probabilities:")
print(p_sklearn[:10])

# Sort values so the curve is smooth
order = np.argsort(z)

z_sorted = z[order]
p_sorted = p_manual[order]

plt.figure(figsize=(8, 5))

plt.plot(
    z_sorted,
    p_sorted,
    linewidth=3,
    label="p = sigmoid(z)"
)

plt.scatter(
    z,
    y_test,
    s=70,
    alpha=0.5,
    label="Observed  labels: 0 or 1"
)

plt.axhline(
    0.5,
    linestyle="--",
    alpha=0.7,
    label="Threshold: p = 0.5"
)

plt.axvline(
    0,
    linestyle="--",
    alpha=0.7,
    label="z = 0"
)

plt.xlabel("Linear score z = Xw + b")
plt.ylabel("Probability of Readmission")

plt.yticks([0, threshold, 1], ["No Readmission (0)", "0.5", "Readmitted (1)"]) 
plt.ylim(0, 1.1)

plt.xlim(-5, 20)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()