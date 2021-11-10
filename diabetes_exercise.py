''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# how many sameples and How many features? (442 sample and 10 features)

diabetes = load_diabetes()
# print(diabetes.data.shape)

# What does feature s6 represent? (glu- blood sugar level)

# print(diabetes.DESCR)
# print(diabetes.data[6])

# print out the coefficient

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)

# There are three steps to medl somehing with sklearn
# 1
mymodel = LinearRegression()

# 2
mymodel.fit(X=X_train, y=y_train)

print(mymodel.coef_)


# print out the intercept

print(mymodel.intercept_)

# 3 Use predict to test the model
# Compare the target and the predicted
predicted = mymodel.predict(X_test)

expected = y_test

# create a scatterplot with regression line
plt.plot(expected, predicted, ".")

x = np.linspace(0, 330, 100)
# print(x)
y = x

plt.plot(x, y)

plt.show()
