import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


x_column_name = 'Age'
y_column_name = 'Survived'

x = df[x_column_name].to_numpy()
y = df[y_column_name].to_numpy()

weight = -0.04
bias = 0.5

# Linear part
z = weight * x + bias

# Sigmoid
p = 1 / (1 + np.exp(-z))

# Predicted classes
y_pred = (p > 0.5).astype(int)

plt.figure(figsize=(10, 5))

# Real data
plt.scatter(x, y, s=120, label="Real data")

# Smooth sigmoid curve
x_curve = np.linspace(0, 30, 300)
z_curve = weight * x_curve + bias
p_curve = 1 / (1 + np.exp(-z_curve))

plt.plot(x_curve, p_curve, linewidth=3, label="Sigmoid model")

# Decision threshold
plt.axhline(0.5, linestyle="--", alpha=0.7)

plt.xlabel("Age")
plt.ylabel("Probability of survised")
plt.title("Logistic Regression: Pass/ Survised Prediction")

plt.grid(True, alpha=0.3)
plt.legend()

plt.show()