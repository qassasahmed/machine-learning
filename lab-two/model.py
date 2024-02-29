import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True)

x_0 = [1, 1, 1, 1]
x_1 = [31.5, 36.5, 43.1, 27.2]
x_2 = [6, 2, 0, 2]
y_ = [21, 25, 18, 30]
X = np.array([[1, 31.5, 6],
              [1, 36.5, 2],
              [1, 43.1, 43.1],
              [1, 27.2, 2]])
y = np.array([[21], [25], [18], [30]])
X_T = np.transpose(X)
X_T_dot_X = np.dot(X_T, X)

inverse_X_T_dot_X = np.linalg.inv(X_T_dot_X)
print(inverse_X_T_dot_X)

X_T_dot_y = np.dot(X_T, y)
print(X_T_dot_y)
coefficients = inverse_X_T_dot_X @ X_T_dot_y
print(coefficients)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define x and y coordinates for the plane
x_plane = np.linspace(-60,60 , 100)
y_plane = np.linspace(-60, 60, 100)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane =  coefficients[0] + coefficients[1]*X_plane + coefficients[2]*Y_plane # z-coordinates are zeros to form the x-y plane


# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot x-y plane
ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3,color='orange')

# Plot scatter points
ax.scatter(x_1, x_2, y, color='green')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Exact Solution')

# Add legend
ax.legend()

# Show plot
plt.show()
