##import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

##load data
data = load_wine()

X = data.data
y = data.target

##normalize data****
scaler = StandardScaler()
X = scaler.fit_transform(X)

##train/split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

##build and train NN

model = MLPClassifier(hidden_layer_sizes=(12, 6), #2 layers
                      max_iter=1000, random_state=42)
model.fit(X_train, y_train)

##evaluate

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

##visualization- loss curve

plt.figure()
plt.plot(model.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

##visualization- decision boundary, using first 2 features

X_vis = X[:, :2]


X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.2, random_state=42
)


model_vis = MLPClassifier(hidden_layer_sizes=(12,), max_iter=1000)
model_vis.fit(X_train_vis, y_train_vis)

##create grid

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# Plot
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y)
plt.title("Neural Network Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

##visualization- approx feature importance
import pandas as pd

importance = np.mean(np.abs(model.coefs_[0]), axis=1)
features = data.feature_names

df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

df = df.sort_values(by="Importance", ascending=False)

plt.figure()
plt.barh(df["Feature"], df["Importance"])
plt.title("Feature Importance (Approx)")
plt.xlabel("Importance")
plt.show()

