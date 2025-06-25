# ML From Scratch

A lightweight Python library that implements core machine learning algorithms from scratch using NumPy.  
Ideal for learning internals of ML models, experimenting, and showcasing on your resume.

---

ml_from_scratch/



## ✅ Features Implemented

-  Maximum Likelihood Estimation (MLE) 
-  PCA (Principal Component Analysis)
-  FDA (Fisher’s Discriminant Analysis)
-  FDA for 2 class
-  LDA & QDA
-  Linear Regression
-  Decision Trees
-  Bagging
-  AdaBoost
-  Gradient Boosting
-  k-Fold Cross Validation
-  Neural Network (Basic MLP)

---

## 🔧 Installation

Clone the repository and import desired modules directly:

```bash
git clone https://github.com/your-username/ml-from-scratch.git
cd ml-from-scratch


-------------------------------------------
### MLEstimator - Maximum Likelihood Estimation
-------------------------------------------

Description:
Computes class-wise mean vectors and covariance matrices using labeled data.
Useful for Gaussian-based classifiers like QDA or generative models.

Usage:

from mle import MLEstimator
import numpy as np

# Example data
X = np.array([
    [1.0, 2.0], [1.2, 1.8], [0.8, 2.2],  # class 0
    [3.0, 3.5], [3.2, 3.6], [2.8, 3.4]   # class 1
])
y = np.array([0, 0, 0, 1, 1, 1])

mle = MLEstimator()
mle.fit(X, y)
means, covariances = mle.get_params()

print("Class 0 Mean:", means[0])
print("Class 0 Covariance:\n", covariances[0])
print("Class 1 Mean:", means[1])
print("Class 1 Covariance:\n", covariances[1])

API:
- fit(X, y): Compute class-wise mean and covariance
- get_params(): Returns (means_dict, covariances_dict)



-------------------------------------------
### PCA - Principal Component Analysis
-------------------------------------------

Description:
Reduces dimensionality of input data by projecting it onto a smaller set of principal components 
that retain a specified amount of variance. Useful for compression and visualization.

Usage:

from pca import PCA
import numpy as np

# Example data
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

pca = PCA(variance_threshold=0.95)
pca.fit(X)
X_reduced = pca.transform(X)

print("Reduced Data Shape:", X_reduced.shape)

API:
- fit(X): Learn principal components from data
- transform(X): Project data onto learned components
- fit_transform(X): Fit and transform in one step




-------------------------------------------
### FDA2Class - Fisher’s Discriminant (2-Class)
-------------------------------------------

Description:
Computes the optimal 1D projection vector for two-class problems using the closed-form:
    w = S_W⁻¹ (μ₁ - μ₂)

Usage:

from fda_2class import FDA2Class
import numpy as np

X = np.array([
    [1, 2], [2, 3], [3, 3],
    [6, 5], [7, 8], [8, 8]
])
y = np.array([0, 0, 0, 1, 1, 1])

fda = FDA2Class()
fda.fit(X, y)
w = fda.get_projection_vector()
X_proj = fda.transform(X)

print("Projection vector:\n", w)
print("Projected data:\n", X_proj)

API:
- fit(X, y): Computes class means, SW, and projection vector
- transform(X): Projects data onto w
- get_projection_vector(): Returns optimal w



-------------------------------------------
### FDA - Fisher’s Discriminant Analysis (Multi-Class)
-------------------------------------------

Description:
Performs dimensionality reduction by projecting input data onto a lower-dimensional space that maximizes class separability.  
Supports multiple classes by solving the generalized eigenvalue problem:
    S_B v = λ S_W v  
The top `k` eigenvectors define the projection directions, where `k ≤ (C - 1)` for `C` classes.

Usage:

from fda import FDA
import numpy as np

X = np.array([
    [1, 2], [2, 3], [3, 3],    # class 0
    [6, 5], [7, 8], [8, 8],    # class 1
    [0, 1], [1, 0], [0, 0]     # optional class 2 (if testing multi-class)
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

fda = FDA()
fda.fit(X, y)
X_proj = fda.transform(X, n_components=1)

print("Projected shape:", X_proj.shape)

API:
- fit(X, y): Computes within-class and between-class scatter matrices and solves for optimal directions.
- transform(X, n_components): Projects data onto the top `n_components` discriminant vectors.



-------------------------------------------
### Discriminant Analysis – LDA & QDA
-------------------------------------------

Description:
------------
This module includes:
- LDAClassifier: Implements Linear Discriminant Analysis, assuming all classes share a common covariance matrix. Decision boundaries are linear.
- QDAClassifier: Implements Quadratic Discriminant Analysis, allowing each class to have its own covariance matrix. Decision boundaries are quadratic.

Both classifiers are based on multivariate Gaussian likelihood and use maximum likelihood estimates for mean and covariance.

Usage Example:
--------------
from discriminant_classifier import LDAClassifier, QDAClassifier
import numpy as np

# Training data (2 classes)
X = np.array([
    [1, 2], [2, 3], [3, 3],    # Class 0
    [6, 5], [7, 8], [8, 8]     # Class 1
])
y = np.array([0, 0, 0, 1, 1, 1])

# Fit LDA
lda = LDAClassifier()
lda.fit(X, y)

# Fit QDA
qda = QDAClassifier()
qda.fit(X, y)

# Predict new sample
new_point = np.array([[7, 3]])

lda_pred = lda.predict(new_point)
qda_pred = qda.predict(new_point)

print(f"\nNew Point {new_point[0]} → LDA Prediction: {lda_pred[0]}")
print(f"New Point {new_point[0]} → QDA Prediction: {qda_pred[0]}")

API:
LDAClassifier
- fit(X, y): Learns class-wise means and shared covariance matrix.
- predict(X): Returns predicted class labels for input X.

QDAClassifier
- fit(X, y): Learns class-wise means and individual covariance matrices.
- predict(X): Returns predicted class labels using quadratic discriminants.




-------------------------------------------
### LinearRegression - Closed Form Linear Regression
-------------------------------------------

Description:
Fits a linear regression model using the analytical solution:
    w = (XᵀX)⁻¹ Xᵀy
Supports single or multi-feature input and returns learned weights including the bias term.

Usage:

from linear_regression import LinearRegression
import numpy as np

# Training data: y = 2x + 1
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

model = LinearRegression()
model.fit(X, y)  # Train the model using closed-form solution

# Predict on training data
predictions = model.predict(X)
print("Predictions on training data:", predictions)

# Print learned weights: [bias, slope]
print("Learned Weights (bias + slope):", model.get_weights())

# Predict on a new test input
test_point = np.array([[6]])
test_prediction = model.predict(test_point)
print(f"Prediction for new input {test_point[0][0]} → {test_prediction[0]}")

API:
- fit(X, y): Trains the model using normal equation
- predict(X): Predicts y for given X
- get_weights(): Returns learned weight vector [bias, w₁, w₂, ...]




-------------------------------------------
### DecisionTreeClassifier - Custom Decision Tree (Gini Index)
-------------------------------------------

Description:
A decision tree classifier built from scratch using Gini impurity to choose the best splits.
Supports binary classification tasks with numerical and encoded categorical features.

The tree is built recursively and predictions are made by traversing from root to leaf.
Also includes a method to print the tree structure.

Usage:

from decision_tree_classifier import DecisionTreeClassifier
import pandas as pd

# Sample data
data = {
    "Age": [25, 30, 35, 40, 45, 50, 55, 60],
    "Income": ["High", "High", "Medium", "Low", "Low", "Low", "Medium", "High"],
    "Student": ["No", "No", "No", "No", "Yes", "No", "Yes", "No"],
    "Credit": ["Fair", "Excellent", "Fair", "Fair", "Excellent", "Excellent", "Excellent", "Fair"],
    "Buy": [0, 0, 1, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Encoding categorical features
category_mappings = {
    "Income": {"Low": 0, "Medium": 1, "High": 2},
    "Student": {"No": 0, "Yes": 1},
    "Credit": {"Fair": 0, "Excellent": 1}
}
for col, mapping in category_mappings.items():
    df[col] = df[col].map(mapping)

X = df.drop(columns=["Buy"])
y = df["Buy"]

# Train the tree
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2)
tree.fit(X, y)

# Predict a new sample
sample = {"Age": 42, "Income": 0, "Student": 0, "Credit": 1}
prediction = tree.predict(sample)
print("Prediction:", "Yes" if prediction == 1 else "No")

# Print the learned tree
tree.print_tree()

API:
- fit(X, y): Builds the tree from training data
- predict(sample_dict): Predicts class (0 or 1) for a new input sample
- print_tree(): Displays the tree structure for inspection



--------------------------------------------------------
### BaggingClassifier - Bootstrap Aggregation (Bagging)
--------------------------------------------------------

Description:
Implements the Bagging ensemble method using Decision Trees as base estimators.
Reduces variance and improves prediction performance by training multiple models on random bootstrap samples of the data and using majority voting.

Standard library-level support for bagging, useful in ensemble learning.

Usage:

from bagging import BaggingClassifier
import pandas as pd

# Sample dataset
data = {
    "Age": [25, 30, 35, 40, 45, 50, 55, 60],
    "Income": ["High", "High", "Medium", "Low", "Low", "Low", "Medium", "High"],
    "Student": ["No", "No", "No", "No", "Yes", "No", "Yes", "No"],
    "Credit": ["Fair", "Excellent", "Fair", "Fair", "Excellent", "Excellent", "Excellent", "Fair"],
    "Buy": [0, 0, 1, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Encode categorical features
category_mappings = {
    "Income": {"Low": 0, "Medium": 1, "High": 2},
    "Student": {"No": 0, "Yes": 1},
    "Credit": {"Fair": 0, "Excellent": 1}
}
for col, mapping in category_mappings.items():
    df[col] = df[col].map(mapping)

X = df.drop(columns=["Buy"])
y = df["Buy"]

# Initialize and train Bagging Classifier
bagger = BaggingClassifier(n_estimators=10, max_depth=3, min_samples_leaf=2)
bagger.fit(X, y)

# Predict on a new data point
sample = {"Age": 42, "Income": 0, "Student": 0, "Credit": 1}
prediction = bagger.predict(sample)

print("Prediction:", "Yes" if prediction == 1 else "No")


API:
- BaggingClassifier(base_estimator=None, n_estimators=10, max_depth=None, min_samples_leaf=1)
    -> base_estimator: Optional base model. Defaults to DecisionTreeClassifier.
    -> n_estimators: Number of bootstrap models to train.
    -> max_depth: Max depth of each decision tree.
    -> min_samples_leaf: Minimum samples per leaf in each tree.

- fit(X, y): Trains n_estimators decision trees on random bootstrap samples.
- predict(sample): Aggregates predictions from all trees via majority voting.



-------------------------------------------
### KFoldCrossValidator - Cross-Validation
-------------------------------------------

Description:
Evaluates a model's performance using k-Fold Cross-Validation. Works with any model
that has fit() and predict() methods, like LinearRegression, DecisionTree, etc.

Usage:

from k_fold import KFoldCrossValidator
from linear_regression import LinearRegression
import numpy as np

# Sample data
X = np.random.rand(100, 1)
y = 3 * X[:, 0] + 1 + np.random.randn(100) * 0.1

model = LinearRegression()
cv = KFoldCrossValidator(model=model, k=5)
avg_mse = cv.evaluate(X, y)

print(f"Average MSE: {avg_mse:.4f}")

API:
- evaluate(X, y): Performs k-Fold validation and returns average metric.




-------------------------------------------
### AdaBoost - Adaptive Boosting Classifier
-------------------------------------------

Description:
AdaBoost (Adaptive Boosting) is a powerful ensemble method that combines multiple 
weak classifiers (like decision stumps) to build a strong classifier. 
Each subsequent classifier is trained to focus more on the samples that previous ones got wrong. 
This version uses decision stumps as the base learners.

Usage:

from adaboost import AdaBoost
import numpy as np

# Example binary classification dataset
X = np.array([
    [1, 2], [2, 3], [3, 1],
    [6, 5], [7, 8], [8, 6]
])
y = np.array([-1, -1, -1, 1, 1, 1])

# Initialize and train model
model = AdaBoost(T=50)
model.fit(X, y)

# Predict for a new point
new_point = np.array([[1.5, 2.5]])
y_pred = model.predict(new_point)

# Accuracy on training data
accuracy = model.score(X, y)

print("Predictions:", y_pred)
print("Accuracy:", accuracy)

Expected Output:
Predictions: [-1]
Accuracy: 1.0

API:
- fit(X, y): Trains the AdaBoost model on data X with binary labels y ∈ {-1, 1}.
- predict(X): Predicts the class label (-1 or 1) for each sample in X.
- score(X, y): Returns the accuracy of the model on the given data and labels.



Notes:
- Ensure labels are binary in {-1, 1}.
- Input to predict() must be a 2D NumPy array, even for a single sample.




-----------------------------------------------
###  Gradient Boosting Regressor - From Scratch
-----------------------------------------------

Description:
This module implements Gradient Boosting Regression using decision stumps as weak learners.
Supports both **squared loss** and **absolute loss** optimization.

Gradient Boosting is an ensemble method where models are trained sequentially.
Each new model tries to correct the errors made by the previous ones using gradient descent on the loss function.

Example Usage:

import numpy as np
from gradient_boost import gb_fit, gb_pred

# Create synthetic data
np.random.seed(42)
n = 100
x = np.random.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * x) + np.random.normal(0, np.sqrt(0.01), n)

# Fit gradient boosting model
splits = np.linspace(0, 1, 20)
F0, stumps = gb_fit(x, y, loss='squared', n_est=500, learning_rate=0.05, splits=splits)

# Predict using the model
y_pred = gb_pred(x, F0, stumps, learning_rate=0.05)

# Print random 5 predictions
indices = np.random.choice(len(x), 5, replace=False)
for i in indices:

API:
- gb_fit(x, y, loss, n_estimators, learning_rate, splits):
    Trains a gradient boosting model on input `x`, target `y`.
    - loss: 'squared' or 'absolute'
    - n_estimators: number of boosting rounds
    - learning_rate: shrinkage rate
    - splits: list or np.array of split points to try for decision stumps
    Returns: initial prediction `F0`, and list of learned stumps

- gb_pred(x, F0, stumps, learning_rate, iters=None):
    Predicts output values for input `x` using the learned boosting model.
    - iters: optionally specify number of boosting rounds to use.




---------------------------------------------------
SimpleFeedForwardNN - Feedforward Neural Network
---------------------------------------------------

Description:
A minimal feedforward neural network implementation in NumPy designed for small-scale regression or binary classification tasks. The network consists of:

- 1 Hidden Layer (with configurable number of units)
- Sigmoid activation in the hidden layer
- Linear activation in the output layer (suitable for regression)
- Trained using Mean Squared Error (MSE) loss and gradient descent

This implementation is intentionally kept lightweight and interpretable, suitable for educational use or simple tasks.

---------------------------------------------------
Usage:

from feedforward_nn import SimpleFeedForwardNN
import numpy as np

# Hardcoded XOR-like dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

# Simple binary labels (not pure XOR, to make it learnable with 1 hidden unit)
y = np.array([0, 1, 1, 0], dtype=np.float32)

# Manually split into train and test
X_train, y_train = X[:3], y[:3]
X_test, y_test   = X[3:], y[3:]

# Initialize and train the model
model = SimpleFeedForwardNN(input_size=2, hidden_size=1, lr=0.1, epochs=1000)
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_test)
mse = model.score(X_test, y_test)

print("Test Inputs:", X_test)
print("Predicted Outputs:", preds)
print("True Outputs:", y_test)
print("Test MSE:", mse)

---------------------------------------------------
API:

- SimpleFeedForwardNN(input_size, hidden_size=1, lr=0.1, epochs=1000)
    Initializes the network with one hidden layer

- fit(X, y)
    Trains the model using mean squared error and gradient descent

- predict(X)
    Predicts outputs for given inputs

- score(X, y)
    Returns mean squared error between predicted and true values



