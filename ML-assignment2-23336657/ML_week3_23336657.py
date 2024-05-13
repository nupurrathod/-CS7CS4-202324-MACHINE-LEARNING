# id:8--8-8 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("week3_data.csv",names=['x1','x2','y'],skiprows=1)
print(data.head())
X = data[['x1','x2']]
y = data['y']
data.describe()
# 1)
# 1.a
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x1'], data['x2'], data['y'], label="data point",c='r')
ax.set_xlabel('x1 input')
ax.set_ylabel('x2 input')
ax.set_zlabel('output')
plt.legend(loc='best')
plt.title("Scatter plot")
plt.show()

# 1.b
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Adding Polynomial Features
poly = PolynomialFeatures(degree=5)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Training Lasso models for different values of c
C_val = [1, 10, 100,1000]

for C in C_val:
    lasso = Lasso(alpha=1/C, max_iter=10000)
    lasso.fit(x_train_poly, y_train)
    print(f"Parameters for C =  {C}: \n {lasso.coef_}")
    print("-----------------------------------------------------------------------------\n")
    y_pred = lasso.predict(x_train_poly)

# 1.c Plot for lasso
import matplotlib.cm as cm

grid = np.linspace(-1, 1, 50)
X_test = np.array([[i, j] for i in grid for j in grid])
X_test_poly = poly.transform(X_test)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x1'], data['x2'], data['y'], label="Data Points", s=10, alpha=0.6, color='red')

colors = cm.viridis(np.linspace(0, 1, len(C_val)))

for index, C in enumerate(C_val):
    lasso = Lasso(alpha=1/C, max_iter=10000)
    lasso.fit(x_train_poly, y_train)
    
    predictions = lasso.predict(X_test_poly).reshape(len(grid), len(grid))
    X_grid, Y_grid = np.meshgrid(grid, grid)
    
    ax.plot_wireframe(X_grid, Y_grid, predictions, label=f"C = {C}", color=colors[index], alpha=0.5)
    
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-2, 1)
ax.set_xlabel('x1 input')
ax.set_ylabel('x2 input')
ax.set_zlabel('output')
plt.title("Lasso prediction for Different C Values")
ax.legend()
plt.show()


grid = np.linspace(-1, 1, 50)
X_test = np.array([[i, j] for i in grid for j in grid])
X_test_poly = poly.transform(X_test)

fig = plt.figure(figsize=(15, 10))

for index, C in enumerate(C_val):
    lasso = Lasso(alpha=1/C, max_iter=10000)
    lasso.fit(x_train_poly, y_train)
    
    ax = fig.add_subplot(2, 3, index + 1, projection='3d')
    ax.scatter(data['x1'], data['x2'], data['y'], label="Data Points", s=10, alpha=0.6, color='g')
    
    predictions = lasso.predict(X_test_poly).reshape(len(grid), len(grid))
    X_grid, Y_grid = np.meshgrid(grid, grid)
    
    ax.plot_wireframe(X_grid, Y_grid, predictions, color='blue', alpha=0.5)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-2, 1)
    
    ax.set_xlabel('x1 input')
    ax.set_ylabel('x2 input')
    ax.set_zlabel('output')
    ax.set_title(f"Lasso prediction for C = {C}")

plt.tight_layout()
plt.show()
# 1.d

# Training ridge for diffrent C values
C_val = [1, 10, 100, 1000]

for C in C_val:
    ridge = Ridge(alpha=1/C, max_iter=10000)
    ridge.fit(x_train_poly, y_train)
    print(f"Parameters for C = {C}: \n {ridge.coef_}")
    print("-----------------------------------------------------------------------------\n")

# Plots for ridge
grid = np.linspace(-1, 1, 50)
X_test = np.array([[i, j] for i in grid for j in grid])
X_test_poly = poly.transform(X_test)
import matplotlib.cm as cm
grid = np.linspace(-1, 1, 50)
X_test = np.array([[i, j] for i in grid for j in grid])
X_test_poly = poly.transform(X_test)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x1'], data['x2'], data['y'], label="Data Points", s=10, alpha=0.6, color='y')

colors = cm.viridis(np.linspace(0, 1, len(C_val)))

for index, C in enumerate(C_val):
    ridge = Ridge(alpha=1/C, max_iter=10000)
    ridge.fit(x_train_poly, y_train)
    
    predictions = ridge.predict(X_test_poly).reshape(len(grid), len(grid))
    X_grid, Y_grid = np.meshgrid(grid, grid)
    
    ax.plot_wireframe(X_grid, Y_grid, predictions, label=f"C = {C}", color=colors[index], alpha=0.5)
    
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-2, 1)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title("Ridge Regression Plots for Different values of C")
ax.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

poly = PolynomialFeatures(degree=5)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

fig = plt.figure(figsize=(15, 10))

for index, C in enumerate(C_val):
    ridge = Ridge(alpha=1/C, max_iter=10000)
    ridge.fit(x_train_poly, y_train)
    
    ax = fig.add_subplot(2, 2, index + 1, projection='3d')
    ax.scatter(data['x1'], data['x2'], data['y'], label="Data Points", s=10, alpha=0.6, color='g')
    
    predictions = ridge.predict(X_test_poly).reshape(len(grid), len(grid))
    X_grid, Y_grid = np.meshgrid(grid, grid)
    
    ax.plot_wireframe(X_grid, Y_grid, predictions, color='blue', alpha=0.5)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-2, 1)
    
    ax.set_xlabel('Input X1')
    ax.set_ylabel('Input X2')
    ax.set_zlabel('Output Y')
    ax.set_title(f"Ridge Predictions for C = {C}")


plt.tight_layout()
plt.show()

#ii.
# Lasso
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

lasso_errors_mean = []
lasso_errors_std = []
C_val = [0.1, 0.5, 1, 10, 15, 20]
colors = ['b', 'g', 'r', 'c', 'm', 'y']  

for C, color in zip(C_val, colors):
    # Lasso Regression
    lasso_model = Lasso(alpha=1 / (2 * C))
    lasso_scores = -cross_val_score(lasso_model, X, y, cv=5, scoring='neg_mean_squared_error')
    lasso_errors_mean.append(np.mean(lasso_scores))
    lasso_errors_std.append(np.std(lasso_scores))

plt.errorbar(C_val, lasso_errors_mean, yerr=lasso_errors_std, label=f'Lasso (C={C})', marker='o')

plt.xlabel('C Values')
plt.ylabel('Mean Squared Error')
plt.title('MSE and Standard Deviation vs Different C Values for Lasso')
plt.legend()
legend = plt.legend()
legend.get_frame().set_alpha(0)

plt.show()

# Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

ridge_errors_mean = []
ridge_errors_std = []
C_val = [0.1, 0.5, 1, 10, 15, 20]
colors = ['b', 'g', 'r', 'c', 'm', 'y']  

for C, color in zip(C_val, colors):
    # Ridge Regression
    ridge_model = Ridge(alpha=1 / (2 * C))
    ridge_scores = -cross_val_score(ridge_model, X, y, cv=5, scoring='neg_mean_squared_error')
    ridge_errors_mean.append(np.mean(ridge_scores))
    ridge_errors_std.append(np.std(ridge_scores))
plt.errorbar(C_val, ridge_errors_mean, yerr=ridge_errors_std, label=f'Ridge (C={C})', marker='o')

plt.xlabel('C Values')
plt.ylabel('Mean Squared Error')
plt.title('MSE and Standard Deviation vs Different C Values for Ridge')
plt.legend()
legend = plt.legend()
legend.get_frame().set_alpha(0)

plt.show()
