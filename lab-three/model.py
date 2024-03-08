import numpy as np

np.set_printoptions(precision=2)

# Define the X
"""
x = np.array([[1, 31.5, 6],
              [1, 36.2, 2],
              [1, 43.1, 0],
              [1, 27.6, 2]])
"""
x = np.reshape([1, 31.5, 6, 1, 36.2, 2, 1, 43.1, 0, 1, 27.6, 2], [4, 3])

# Define the Y
# y = np.array([[21], [25], [18], [30]])
y = np.row_stack([21, 25, 18, 30])

# Initialise Beta vector
beta_vector = np.array([[0], [0], [0]])

# Assign 4 to the learning rate 'eta'
lean_rate = 4


# Define the update rule
def new_beta(b, eta, g):
    return b - eta * g.T


# Define the loss function
def mse(x, y, b):
    y_hat = np.dot(x, b)
    error = y - y_hat
    return error / len(y)


# Define the gradient function
def gr(err):
    b = -1 * np.dot(err.T, x)
    return b


error = mse(x, y, beta_vector)
gradient = gr(error)
beta_vector = new_beta(beta_vector, lean_rate, gradient)
print(f'beta vector on iteration #{1}:\n{beta_vector}')
