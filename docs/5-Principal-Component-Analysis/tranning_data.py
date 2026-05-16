import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datasets import load_dataset

import umap


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



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

numeric_cols = [

    "time_in_hospital", "num_lab_procedures", "num_procedures",

    "num_medications", "number_outpatient", "number_emergency",

    "number_inpatient", "number_diagnoses"

]


categorical_cols = [

    "age",                      

    "discharge_disposition_id",

    "admission_source_id"      

]


Y_column = 'readmitted'


df = df.dropna()

y = df[Y_column]


df[categorical_cols] = df[categorical_cols].astype(str)

x_combined = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)




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

    s=30,

    alpha =0.5

)


plt.xlabel("PC1")

plt.ylabel("PC2")


plt.title("Diabetes Readmission Dataset After PCA")


plt.grid(True, alpha=0.3)


plt.show() 