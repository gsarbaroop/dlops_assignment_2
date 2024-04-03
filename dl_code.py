import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x)), False

def relu(x):
    return np.maximum(0, x), True

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x), True

def tanh(x):
    return np.tanh(x), True

# Generate x values from -10 to 10
x = np.linspace(-10, 10, 100)

# Calculate corresponding y values using the sigmoid function

y1, boolVal1 = sigmoid(x)
y2, boolVal2 = relu(x)
y3, boolVal3 = leaky_relu(x)
y4, boolVal4 = tanh(x)

if boolVal1:
# Plot the sigmoid function
    plt.plot(x, y1)
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.grid(True)
elif boolVal2 or boolVal3 or boolVal4:    
    # Plot the relu function
    plt.plot(x, y2)
    plt.title('Relu Function')
    plt.xlabel('x')
    plt.ylabel('relu(x)')
    plt.grid(True)

    # Plot the leaky_relu function
    plt.plot(x, y3)
    plt.title('leaky_relu Function')
    plt.xlabel('x')
    plt.ylabel('leaky_relu(x)')
    plt.grid(True)

    # Plot the tanh function
    plt.plot(x, y4)
    plt.title('tanh Function')
    plt.xlabel('x')
    plt.ylabel('tanh(x)')
    plt.grid(True)

    plt.show


