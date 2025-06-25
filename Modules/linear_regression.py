import numpy as np

class LinearRegression:
    """
    Linear Regression using the closed-form solution (Normal Equation):
    w = (X^T X)^(-1) X^T y
    """

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add bias term (column of ones)
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Closed-form solution
        XTX = X_bias.T @ X_bias
        XTy = X_bias.T @ y
        self.weights = np.linalg.inv(XTX) @ XTy

    def predict(self, X):
        # Add bias term
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ self.weights

    def get_weights(self):
        return self.weights
