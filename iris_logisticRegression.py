#import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

##import built in dataset

from sklearn.datasets import load_iris 
data = load_iris()

X = data.data 
y = data.target

print(data.feature_names) 
print(data.target_names)

#split data

X_train, X_test, y_train, y_test = train_test_split ( X, y, test_size=0.2, random_state=42)

##create and train Logistic regression model

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

##make predictions

y_pred = model.predict(X_test)

##evaluate model

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#####test sample data

sample = [[5.0, 2.3, 4.6, .02]]
prediction = model.predict(sample)

print("predict class:", data.target_names[prediction[0]])

###^basic ai pipeline.
###visualization

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Performance Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Performance Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
