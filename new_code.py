import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate x values from -10 to 10
x = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

# Calculate corresponding y values using the sigmoid function
y1 = sigmoid(x)
y2 = relu(x)
y3 = leaky_relu(x)
y4 = tanh(x)

# Plot the sigmoid function
plt.plot(x, y1)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show

# Plot the relu function
plt.plot(x, y2)
plt.title('Relu Function')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.grid(True)
plt.show

# Plot the leaky_relu function
plt.plot(x, y3)
plt.title('leaky_relu Function')
plt.xlabel('x')
plt.ylabel('leaky_relu(x)')
plt.grid(True)
plt.show

# Plot the tanh function
plt.plot(x, y4)
plt.title('tanh Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True)
plt.show


