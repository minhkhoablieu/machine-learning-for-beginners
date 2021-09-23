from sklearn import datasets, tree, model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

variety = ['Setosa', 'Versicolor', 'Virginica']

print(variety[y_pred[0]])

print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))