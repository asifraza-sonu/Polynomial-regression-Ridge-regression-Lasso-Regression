import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import math
DataFrame1 = pd.read_csv('project - part D - training data set.csv')
DataFrame2 = pd.read_csv('project - part D - testing data set.csv')
#Training Data
X_train = DataFrame1['Father'].values/10000
X_train = X_train.reshape(-1,1)
y_train = DataFrame1['Son'].values.reshape(-1,1)
#testing Data
X_test = DataFrame2['Father'].values/10000
X_test = X_test.reshape(-1,1)
y_test = DataFrame2['Son'].values.reshape(-1,1)
train_error = []
test_error = []
#function for avoid duplication of list elements
def my_function(x):
    return list(dict.fromkeys(x))
for i in range(1,11):
        poly = PolynomialFeatures(degree=i)
        Modified_X_train = poly.fit_transform(X_train)
        Modified_X_test = poly.fit_transform(X_test)
        #To display number of training and testing points being considered for building the model
        if i == 1:
            print ('The number of Training points are :', Modified_X_train.shape[0])
            print ('The number of Testing points are  :',Modified_X_test.shape[0])
        reg = LinearRegression()
        reg.fit(Modified_X_train,y_train)
        print ('The RMSE of the Trainining data for model of degree',i, 'is :',math.sqrt(metrics.mean_squared_error(y_train,reg.predict(Modified_X_train))))
        print ('The RMSE of the Testing data for model of degree',i, 'is :',math.sqrt(metrics.mean_squared_error(y_test,reg.predict(Modified_X_test))))
        train_error.append(math.sqrt(metrics.mean_squared_error(y_train,reg.predict(Modified_X_train))))
        test_error.append(math.sqrt(metrics.mean_squared_error(y_test,reg.predict(Modified_X_test))))
train_error = my_function (train_error)
test_error = my_function(test_error)
#To print the best degree of the polynomial based on the lower RMSE Test Score
print ('The best degree of the polynomial is  ', test_error.index(min(test_error))+1)
plt.xlabel('polynomial')
plt.ylabel('RMSE')
plt.plot(np.linspace(1,11,10),train_error, 'bo-',label='Train RMSE')
plt.plot(np.linspace(1,11,10),test_error, 'ro-',label='Test RMSE')
plt.legend()
plt.savefig('2018AIML551_poly_part_d.png')
plt.show()