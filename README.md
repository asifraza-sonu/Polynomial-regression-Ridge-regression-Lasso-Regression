# Polynomial-regression-Ridge-regression-Lasso-Regression
Objective The goal of this part of the project is to build a polynomial regression model for a given dataset and demonstrate Lasso and Ridge regressions. The regression models should be implemented using the libraries sklearn and numpy.

Problem Description

1. Implement Polynomial regression (degrees 1 to 10)
● Plot train and test RMSE, for each degree. These plots can be graphs where degree of the polynomial is x-axis and RMSE is y-axis. The graphs should be similar to the following graphs (obviously your graphs will be different but they look like as follows):
● Select the best degree of polynomial based on test error (lower the better.)

2. Regularization using Lasso and Ridge
● For degree 10, find out the train and test RMSE for ridge and lasso regression. You may take the regularization parameter (alpha or lambda) to be the default parameter provided by python. Discuss the improvements brought down by ridge, lasso regression in RMSE as compared to the typical regression model built using the polynomial of degree 10.

Polynomial regression implementation
We do not have a direct API for polynomial regression in scikit-learn library. So, we have to treat polynomial regression ( y = w0 + w1 x + w2 x2 + …. + wD xD) as a multiple regression problem (y = w0 + w1 x1 + w2 x2 + …. + wD xD ) with x1 = x, x2 = x2, x3 = x3 …… xD = xD to build polynomial regression model of degree ‘D’.

Steps Involved :
1. Create an object of sklearn.preprocessing.PolynomialFeatures class.
2. Parameter degree of the class represents the degree of polynomial that we are trying to fit the data.
3. We convert single feature X into [𝟏,𝑿,𝑿𝟐,𝑿𝟑,𝑿𝟒,....,𝑿𝑫], where D represents the parameter degree of the class by applying fit_transform() method on X(features) and store it in an object modified_X.
4. Generate a linear regression model similar to Part-A with passing modified_X instead of X in the fit() method for training.
Lasso/Ridge implementation
Instead of using sklearn.linear_model.LinearRegression for building simple linear regression models, you have to make of sklearn.linear_model.Lasso, and sklearn.linear model.Ridge to build Lasso and Ridge regression models respectively.
