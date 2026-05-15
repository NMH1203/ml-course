#In this part i fine the data in huggingface to calculate probability of readmision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

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

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=4)

x_pca = pca.fit_transform(x_scaled)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    x_pca[:,0],
    x_pca[:,1],
    c=y,
    s=70
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Iris Dataset After PCA")

plt.grid(True, alpha=0.3)

plt.show()
