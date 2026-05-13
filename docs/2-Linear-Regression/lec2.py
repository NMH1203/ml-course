import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd


dataset = load_dataset('krishal07/student-performance', split="train")
df = dataset.to_pandas()


X_column_name = 'StudyHours'
Y_column_name = 'TestScore_Math'

x = df[X_column_name].to_numpy()
y = df[Y_column_name].to_numpy()

# Manual weights
weight = 0.65
bias = 0

# Predictions and errors
y_pred = (weight * x) + bias
residuals = y - y_pred

l1 = np.sum(np.abs(residuals))
l2 = np.sum(residuals ** 2)
mse = np.mean(residuals ** 2)
rmse = np.sqrt(mse)

# Plotting
plt.figure(figsize=(10, 6))

# Plot actual data points (size reduced slightly so they don't overlap too much)
plt.scatter(x, y, s=50, alpha=0.6, label="Real Japanese Texts")

# Regression line (Changed range from 0-50 to 0-1 to match our data)
x_line = np.linspace(0, 1, 100)
y_line = weight * x_line + bias

plt.plot(
    x_line,
    y_line,
    color='red',
    linewidth=3,
    label=f"Model: y = {weight:.2f}x + {bias:.2f}"
)

for xi, yi, ypi in zip(x, y, y_pred):
    plt.plot([xi, xi], [yi, ypi], color='gray', linewidth=1, alpha=0.3)


plt.title("Linear Regression: Kanji Difficulty vs. Overall Difficulty")
plt.xlabel("Overall_difficulty (x)")
plt.ylabel("Grammar_complexity (y)")

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

loss_text = (
    f"L1 loss:  {l1:.2f}\n"
    f"L2 loss:  {l2:.2f}\n"
    f"MSE:      {mse:.2f}\n"
    f"RMSE:     {rmse:.2f}"
)

plt.text(
    0.05,
    0.95,
    loss_text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.15, facecolor="white"),
    fontsize=12
)

plt.show()

best_weight, best_bias = np.polyfit(x, y, 1)

y_best = best_weight * x + best_bias
best_mse = np.mean((y - y_best) ** 2)
best_rmse = np.sqrt(best_mse)

print(f"Best weight: {best_weight:.3f}")
print(f"Best bias:   {best_bias:.3f}")
print(f"Best MSE:    {best_mse:.3f}")
print(f"Best RMSE:   {best_rmse:.3f}")