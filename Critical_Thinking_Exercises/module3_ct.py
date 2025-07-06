import numpy as np


# Activation function: Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Dataset creation function for arithmetic sequences
# TODO : Add more complex sequences or different types of arithmetic progressions
def create_arithmetic_dataset():
    X = []
    y = []
    for i in range(1, 11):  # Create 10 samples
        X.append([i, i+1, i+2])
        y.append([i+3])  # The next number in the sequence
    return np.array(X), np.array(y)


# Initialize parameters for the ANN
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(510)  # CSC510! (seed for reproducibility)

    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    return W1, b1, W2, b2


# Feedforward function for the ANN
def feedforward(X, W1, b1, W2, b2):
    # Layer 1: input --> hidden
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    # Layer 2: hidden --> output
    z2 = np.dot(a1, W2) + b2
    y_hat = z2  # No activation function for output layer (regression task)

    return a1, y_hat


# Backpropagation function to compute gradients
def backpropagation(X, y, a1, y_hat, W2):
    # Output layer error
    delta2 = y_hat - y

    # Gradients for W2 and b2
    dW2 = np.dot(a1.T, delta2)  # (4, 1)
    db2 = np.sum(delta2, axis=0, keepdims=True)  # (1, 1)

    # calculate error for hidden layer (derivative of sigmoid)
    delta1 = np.dot(delta2, W2.T) * a1 * (1 - a1)  # (10, 4)

    # Gradients for W1 and b1
    dW1 = np.dot(X.T, delta1)  # (3, 4)
    db1 = np.sum(delta1, axis=0, keepdims=True)  # (1, 4)

    return dW1, db1, dW2, db2


# Update parameters using gradient descent
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


# Mean Squared Error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Main function to run the ANN training and prediction
def main():
    X, y = create_arithmetic_dataset()
    print("\nCSC510 -- Artificial Neural Network (ANN) for Arithmetic Sequences --\n")
    print("This program implements a simple ANN to predict the next number in an arithmetic sequence.")
    print("Training data consists of sequences of 3 numbers, and the target is the next number in the sequence.\n")
    print("Training data is an array structured as follows :\n", X)
    print("")
    hidden_size = input("Enter the number of neurons in the hidden layer (or press Enter for default): ")
    if hidden_size.strip() == "":
        hidden_size = 10
    else:
        hidden_size = int(hidden_size)
    print(f"Using {hidden_size} hidden neurons.\n")
    print("How many epochs do you want to train? (default: 1500)")
    epochs = input("Enter number of epochs (or press Enter for default): ")
    if epochs.strip() == "":
        epochs = 1500
    else:
        epochs = int(epochs)

    # Set static Parameters
    input_size = 3
    output_size = 1
    learning_rate = 0.001

    # Initialize weights and biases
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        a1, y_hat = feedforward(X, W1, b1, W2, b2)

        # Compute loss
        loss = mean_squared_error(y, y_hat)

        # Backpropagation
        dW1, db1, dW2, db2 = backpropagation(X, y, a1, y_hat, W2)

        # Gradient descent update
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # Optional: print every 100 epochs
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    # Output final predictions
    print("\nFinal predictions after training:")
    _, final_output = feedforward(X, W1, b1, W2, b2)
    print("Predicted (y_hat):")
    for i in final_output:
        print(f"{i[0]:.4f}, ", end='')
    print("\nTrue values (y):")
    for i in y:
        print(f"{i[0]:.4f}, ", end='')
    final_loss = mean_squared_error(y, final_output)
    print(f"\nFinal Loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()