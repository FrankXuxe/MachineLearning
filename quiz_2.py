import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


gr = pd.read_csv("gameratings.csv")
tn = pd.read_csv("target_names.csv")
te = pd.read_csv("test_esrb.csv")
'''
#print(gr.title.values)
#print(gr.title.values.reshape(-1, 1))

#X_train, X_test, y_train, y_test = gr.title.values.reshape(-1, 1), te.Target.values, random_state=11
'''
X_train = gr.title.values.reshape(-1, 1)
y_train = gr.Target.values
X_test = te.title.values.reshape(-1, 1)
y_test = te.Target.values
'''
lr = LinearRegression()

lr.fit(X=X_train, y=y_train)

coef = lr.coef_
intercept = lr.intercept_

predicted = lr.predict(X_test)
expected = y_test

print(predicted[:20])
print(expected[:20])
'''

knn = KNeighborsClassifier()

knn.fit(X=X_train, y=y_train)

predicted = knn.predict(X=X_test)
expected = y_test

print(predicted[:20])
print(expected[:20])

