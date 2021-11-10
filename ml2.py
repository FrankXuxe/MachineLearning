import matplotlib as plt2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3))

print(nyc.Date.values)

# -1: as many rows as we need for the oolumns we have to only "1" row
print(nyc.Date.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)


lr = LinearRegression()

lr.fit(X=X_train, y=y_train)

coef = lr.coef_
intercept = lr.intercept_

predicted = lr.predict(X_test)
expected = y_test

print(predicted[:20])
print(expected[:20])


def predict(x): return coef * x + intercept


print(predict(2025))


# Why not matplotlib
axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False
)

axes.set_ylim(10, 70)


x = np.array([min(nyc.Date.values), max(nyc.Date.values)])

print(x)

y = predict(x)

print(y)


line = plt.plot(x, y)

plt.show()
