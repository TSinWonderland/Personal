##import libraries

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#load data and split

X,y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#model

model = DecisionTreeClassifier()

##train
model.fit(X_train, y_train)

##predict
y_pred = model.predict(X_test)

#evaluate

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

##visualize

plt.figure(figsize=(12, 8))
plot_tree( model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
