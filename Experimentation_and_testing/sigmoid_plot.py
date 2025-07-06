import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Plotting function
def plot_sigmoid_behavior():
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    y_derivative = sigmoid_derivative(x)

    plt.figure(figsize=(10, 4))

    # Plot sigmoid
    plt.subplot(1, 2, 1)
    plt.plot(x, y, label='Sigmoid')
    plt.title("Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("σ(x)")
    plt.grid(True)
    plt.legend()

    # Plot derivative
    plt.subplot(1, 2, 2)
    plt.plot(x, y_derivative, label='Sigmoid Derivative', color='orange')
    plt.title("Sigmoid Derivative")
    plt.xlabel("x")
    plt.ylabel("σ'(x)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main function
def main():

    plot_sigmoid_behavior()


# Run the main function
if __name__ == "__main__":
    main()