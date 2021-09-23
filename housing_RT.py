from sklearn import datasets, tree, model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


df = pd.read_csv('housing_RT.csv', index_col=0)


X = df.iloc[:,1:5]
y = df.iloc[:,0]


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
model = tree.DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(y_pred[0])

