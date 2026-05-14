import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# We create a small synthetic dataset from a sine function.
# The true function is smooth, but the observations contain noise.
X = np.linspace(0, 10, 28).reshape(-1, 1)
y_true = np.sin(X).ravel()
y = y_true + np.random.normal(0, 0.45, size=y_true.shape)

# We split the data into training and test sets.
# The model only learns from the training data.
# The test data is used to check whether the model generalizes.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42
)

# Points used only for drawing smooth curves
x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
# y_true_plot = np.sin(x_plot).ravel()

degree = 10

alpha = 1000

model_reg = make_pipeline(
    PolynomialFeatures(degree, include_bias=False),
    Ridge(alpha=alpha)
)

model_reg.fit(X_train, y_train)

y_plot_reg = model_reg.predict(x_plot)
y_train_pred_reg = model_reg.predict(X_train)
y_test_pred_reg = model_reg.predict(X_test)

train_mse_reg = mean_squared_error(y_train, y_train_pred_reg)
test_mse_reg = mean_squared_error(y_test, y_test_pred_reg)

plt.figure(figsize=(8, 5))

plt.scatter(X_train, y_train, label="Training data", s=70)
plt.scatter(X_test, y_test, label="Test data", s=70)

plt.plot(x_plot, y_plot_reg, linewidth=3, label="Model prediction")

plt.title(
    f"With Ridge Regularization\n"
    f"Degree = {degree} | Alpha = {alpha} | Train MSE = {train_mse_reg:.3f} | Test MSE = {test_mse_reg:.3f}"
)

plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()