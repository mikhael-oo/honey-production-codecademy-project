# analyze the honey production rate of the country

# import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# import file into a dataframe
df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

# group into yearly average production
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# select the years
X = prod_per_year.year
X = X.values.reshape(-1, 1)

# select the yearly produce
y = prod_per_year.totalprod

# using Linear Regression to predict
regr = linear_model.LinearRegression()
regr.fit(X, y)

# getting the slope of the line
print(regr.coef_[0])
# getting the intercept of the line
print(regr.intercept_)

# get the predicted y values
y_predict = regr.predict(X)

# plot the data
plt.figure(figsize=(8,6))
plt.scatter(X, y, alpha=0.4)
plt.plot(X, y_predict)
plt.xlabel('Year')
plt.ylabel('Average Produce')
plt.title('Average Produce Per Year')
plt.show()
plt.clf()


# to predict rate of produce for coming years

# store the years into an array and rotate them
X_future = np.array(range(2013,2051))
X_future = X_future.reshape(-1, 1)

# future predictions of y_values
future_predict = regr.predict(X_future)

# plot the data
plt.plot(X_future, future_predict)
plt.title('Average Produce Per Year')
plt.xlabel('Year')
plt.ylabel('Average Produce')
plt.show()

