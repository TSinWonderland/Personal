##import libraries


import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

##load data

data = load_wine()
X = data.data
y = data.target

##split data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

##train model

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

##evaluate


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

##visualize


importances = model.feature_importances_
features = data.feature_names

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

##plot one treee

plt.figure(figsize=(12, 8))

plot_tree(
    model.estimators_[0],  # pick first tree
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True
)

plt.title("One Tree from Random Forest")
plt.show()

##plot multiple trees


for i in range(3):  # show first 3 trees
    plt.figure(figsize=(10, 6))
    plot_tree(
        model.estimators_[i],
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True
    )
    plt.title(f"Tree {i+1}")
    plt.show()

##performance metrics

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))


#confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

##plot cm


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

