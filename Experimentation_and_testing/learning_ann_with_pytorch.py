import torch
import torch.nn as nn
import torch.optim as optim


# Define the model
class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input → Hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden → Output

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Training data: sequences and their next number
X_train = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0],
    [6.0, 7.0, 8.0],
], dtype=torch.float32)

y_train = torch.tensor([
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0],
], dtype=torch.float32)

# Model, loss function, optimizer
model = SimpleANN(input_size=3, hidden_size=5, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")


# CLI for user testing
def predict_next_number(sequence):
    model.eval()
    input_seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_seq)
    return output.item()


if __name__ == "__main__":
    print("\nPyTorch ANN - Predict the next number in a 3-number sequence.")
    print("Example: input `1,2,3` should output something close to `4`.")

    while True:
        user_input = input("\nEnter 3 numbers separated by commas (or 'exit'): ")
        if user_input.lower() == 'exit':
            break
        try:
            numbers = [float(x) for x in user_input.strip().split(',')]
            if len(numbers) != 3:
                print("Please enter exactly 3 numbers.")
                continue
            prediction = predict_next_number(numbers)
            print(f"Predicted next number: {prediction:.2f}")
        except Exception as e:
            print(f"Invalid input: {e}")