import numpy as np
from feedforward_nn import SimpleFeedForwardNN

# Hardcoded XOR-like dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

# Simple binary labels (not XOR, to make it learnable with 1 hidden unit)
y = np.array([0, 1, 1, 0], dtype=np.float32)

# Split data into train and test manually
X_train, y_train = X[:3], y[:3]
X_test, y_test   = X[3:], y[3:]

# Initialize and train the model
model = SimpleFeedForwardNN(input_size=2, hidden_size=1, lr=0.1, epochs=1000)
model.fit(X_train, y_train)

# Predict and print results
preds = model.predict(X_test)
mse = model.score(X_test, y_test)

print("Test Inputs:", X_test)
print("Predicted Outputs:", preds)
print("True Outputs:", y_test)
print("Test MSE:", mse)
