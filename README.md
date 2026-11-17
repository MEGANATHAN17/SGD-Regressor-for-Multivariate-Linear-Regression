# SGD-Regressor-for-Multivariate-Linear-Regression

# AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Start the program.
2. Create the input and output datasets.
3. Split the data into training and testing sets.
4. Standardize the data.
5. Train the model using `SGDRegressor`.
6. Predict the output values.
7. Calculate Mean Squared Error (MSE).
8. Plot actual and predicted values.
9. Display the results and stop.

# Program:
```
NAME : MEGANATHAN R
REG : 212224230156


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load Dataset
data = fetch_california_housing()

# Select Features and Multi-Output Target
X = data.data[:, :3]
Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# Multi-Output SGD Regression
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model = MultiOutputRegressor(sgd)

model.fit(X_train_scaled, Y_train_scaled)

# Prediction
Y_pred_scaled = model.predict(X_test_scaled)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)

print("\nFirst 5 Predictions:")
print(Y_pred[:5])

# Evaluation
mse = mean_squared_error(Y_test_orig, Y_pred)
print("\nMulti-Output SGD MSE:", mse)

mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)

# Display Results
for i in range(5):
    print("\nExample", i+1)
    print("Inputs:", X_test[i])
    print("Actual Outputs:", Y_test_orig[i])
    print("Predicted Outputs:", Y_pred[i])

# Single-Output Linear Regression and SGD Comparison
X = data.data[:, :3]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# SGD Regression
sgd = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    eta0=0.01,
    learning_rate='constant',
    random_state=42
)

sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

# Compare Results
print("\nModel Comparison:")
print("Linear Regression MSE:", mean_squared_error(y_test, lr_pred))
print("SGD Regression MSE:", mean_squared_error(y_test, sgd_pred))

```

# Output:

<img width="527" height="635" alt="image" src="https://github.com/user-attachments/assets/ebea3bc7-f1ea-43fd-a0ed-a69aea6cccc9" />


# Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
