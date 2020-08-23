from tensorlinear.LinearRegression import LinearRegression
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression as LR2
dims = 10
n = 100000




X = np.random.random((n, dims))
params = np.random.random((dims,1))
b = np.random.random(1)
y = np.matmul(X, params)
y = y + b
model = LinearRegression(lr=0.1, fit_intercept=False, iters=1000)
#print(X, params, y)
model.fit(X, y)

print('Real: ', params)
print('Modelled: ', model.params)
print('Difference: ', tf.reduce_mean(params - model.params))


model2 = LR2(fit_intercept=False)
model2.fit(X, y)
print('Model2: ', model2.coef_.flatten())
print('Difference: ', tf.reduce_mean(params - model2.coef_.flatten()))