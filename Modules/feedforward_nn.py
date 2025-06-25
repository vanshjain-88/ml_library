import numpy as np

class SimpleFeedForwardNN:
    def __init__(self, input_size, hidden_size=1, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.input_size = input_size

        # Initialize weights and biases
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(1, hidden_size)
        self.b2 = np.random.randn(1)

    def fit(self, X, y):
        m = X.shape[0]
        for epoch in range(self.epochs):
            # Forward pass
            z1 = X.dot(self.W1.T) + self.b1
            a1 = self.sigmoid(z1)
            z2 = a1.dot(self.W2.T) + self.b2
            y_pred = z2.flatten()

            # Compute loss derivative
            error = y_pred - y
            dz2 = 2 * error / m

            # Backpropagation
            dW2 = np.dot(dz2, a1).reshape(self.W2.shape)
            db2 = np.sum(dz2)

            da1 = dz2.reshape(-1, 1) * self.W2
            dz1 = da1 * a1 * (1 - a1)

            dW1 = np.dot(dz1.T, X)
            db1 = np.sum(dz1, axis=0)

            # Update weights and biases
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        z1 = X.dot(self.W1.T) + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.W2.T) + self.b2
        return z2.flatten()

    def score(self, X, y):
        preds = self.predict(X)
        mse = np.mean((preds - y) ** 2)
        return mse
