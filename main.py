import numpy as np

x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

one = np.ones((x.shape[0], 1))
x_bar = np.concatenate((one, x), axis = 1)
A = np.dot(x_bar.T, x_bar)
b = np.dot(x_bar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print(one)
print(x_bar)
print(A)
print(b)
print(w)