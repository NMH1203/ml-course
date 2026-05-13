import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load the dataset
dataset = load_dataset("gauss314/bitcoin_daily", split="train")
df = dataset.to_pandas()

#sort the data date 
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

#add data to the row and collum
df['Day_Index'] = np.arange(len(df))

X_column = ["Day_Index"]
Y_column = ["price"] 
#not use the row when have missing data
df = df.dropna()

x = df[X_column].to_numpy()
y = df[Y_column].to_numpy()


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)
# For smooth plotting
x_plot = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)

models = [
    {"degree": 1, "name": "Model A"},
    {"degree": 20, "name": "Model B"},
    {"degree": 50, "name": "Model C"},
]

for item in models:
    degree = item["degree"]

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    #train the data
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y_plot = model.predict(x_plot)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    plt.figure(figsize=(8, 5))


    plt.scatter(x_train, y_train, alpha=0.4, s=10, label="Training data")
    plt.scatter(x_test, y_test, alpha=0.4, s=10, label="Test data")
    
    
    plt.plot(x_plot, y_plot, color='red', linewidth=3, label=item["name"])

    plt.title(
        f"{item['name']} — Polynomial degree {degree}\n"
        f"Train MSE = {train_mse:,.0f} | Test MSE = {test_mse:,.0f}"
    )

    plt.xlabel("Days Since Start")
    plt.ylabel("Bitcoin Price")
    
    
    plt.ylim(y.min() * 0.8, y.max() * 1.2) 
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()