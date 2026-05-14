import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# 1. Load Data
df = sns.load_dataset("titanic")

# 2. Select Specific Features
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df[features].copy()
y = df["survived"]

# 3. Handle Missing Values (Only for the columns we kept)
X["age"] = X["age"].fillna(X["age"].median())
X["fare"] = X["fare"].fillna(X["fare"].median())

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
print("-" * 40)

# 5. Define Preprocessing & Pipeline
numeric_features = ["age", "fare"]
categorical_features = ["sex"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

log_reg = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# 6. Train the Model
log_reg.fit(X_train, y_train)

# 7. Make Predictions
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

# Optional: View results dataframe
results = pd.DataFrame({
    "true_survived": y_test.values,
    "predicted_class": y_pred,
    "predicted_probability": y_proba
})

# 8. Evaluate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print("\n" + "="*40 + "\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Did not survive", "Survived"]
))

# 9. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did not survive", "Survived"]
)

disp.plot(cmap="Blues") # Added a nice blue color map
plt.title("Titanic Survival Confusion Matrix")
plt.show()