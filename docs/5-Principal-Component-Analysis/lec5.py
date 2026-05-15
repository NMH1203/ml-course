import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Apartment example
n = 120

size = np.random.normal(70, 15, n)

rooms = size / 25 + np.random.normal(0, 0.35, n)

age = np.random.normal(30, 10, n)

X = np.column_stack([size, rooms, age])

# Correlated data
plt.figure(figsize=(7,6))

plt.scatter(X[:,0], X[:,1], s=60)

plt.xlabel("Apartment size")
plt.ylabel("Number of rooms")

plt.title("Original Data")

plt.grid(True, alpha=0.3)

plt.show()

# Apply PCA
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

mean = X_scaled.mean(axis=0)

pc1 = pca.components_[0]
pc2 = pca.components_[1]
pc3 = pca.components_[2]

t = np.linspace(-3, 3, 100)

pc1_line = mean + t[:, None] * pc1
pc2_line = mean + t[:, None] * pc2
pc3_line = mean + t[:, None] * pc3

plt.figure(figsize=(7,6))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    s=60,
    alpha=0.7
)

plt.plot(
    pc1_line[:,0],
    pc1_line[:,1],
    linewidth=3,
    label="PC1"
)

plt.plot(
    pc2_line[:,0],
    pc2_line[:,1],
    linewidth=3,
    label="PC2"
)

plt.plot(
    pc3_line[:,0],
    pc3_line[:,1],
    linewidth=3,
    label="PC3"
)

plt.xlabel("Size (standardized)")
plt.ylabel("Rooms (standardized)")

plt.title("PCA Finds the Main Direction")

plt.legend()

plt.grid(True, alpha=0.3)

plt.axis("equal")

plt.show()