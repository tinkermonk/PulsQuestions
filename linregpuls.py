# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 20:34:10 2018

@author: yassifn
"""
import matplotlib.pyplot as plt
import numpy as np

"""
In statistics, linear regression is a linear approach to modelling the 
relationship between a scalar response (or dependent variable) and one 
or more explanatory variables (or independent variables).

the equation can be expressed as follows:
    
y = b0 + b1 * x

b0 and b1 are the coefficients we must estimate from the training data.

The first step in applying linear regression in machine learning is to calculate
the Mean and Variance

the mean can be expressed as follows:

mean(x) = sum(x) / count(x)

"""
# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

"""
Variance for a list of numbers can be expressed as:

variance = sum( (x - mean(x))^2 )
    
"""

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

"""
The above functions can be put together and tested on a small dataset
"""



"""

The next step is to calculate Covariance

From Wikipedia:
In probability theory and statistics, covariance is a measure of the joint 
variability of two random variables. 
If the greater values of one variable mainly correspond with the greater values 
of the other variable, and the same holds for the lesser values, 
(i.e., the variables tend to show similar behavior), the covariance is positive. 
In the opposite case, when the greater values of one variable mainly 
correspond to the lesser values of the other, (i.e., the variables tend to 
show opposite behavior), the covariance is negative.


"""

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

"""
After calculating the Mean and Covariance, the next step is to calculate the
estimate coefficients in a simple linear regression.

The equation for calculating the first coefficient, made easier as the functions
are now defined as above:
    
B1 = covariance(x, y) / variance(x)
where B1 is the first coefficient.

Where the line intersects at the y axis we can express it as follows:
B0 = mean(y) - B1 * mean(x)

Using the two equations we can now define the function:
"""
# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

"""
We can therefore now make predictions based on the above when applied to
the training data.

The equation is simply:

    	
y = b0 + b1 * xy

We can now express this in a function to take two inputs, the test and training
sets. This will implement the prediction equation on the two sets of data.
"""

def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions