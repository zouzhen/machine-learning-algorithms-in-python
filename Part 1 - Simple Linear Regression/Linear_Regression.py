'''
使用sklearn包中的线性回归算法
'''
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the diabetes dataset
# diabetes = datasets.load_diabetes()

# # Use only one feature
# diabetes_X = diabetes.data[:, np.newaxis, 2]

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test,
#                                                       diabetes_y_pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
'''
使用代码实现线性回归算法(一元)
'''

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# path = os.path.dirname(os.getcwd()) + '\data\ex1data1.txt'
# data = pd.read_csv(path, header=None, names=['Population', 'Profit'])


# def computeCost(X, y, theta):
#     '''
#     损失函数
#     X: 自变量
#     y: 因变量
#     theta: 参数向量
#     '''
#     inner = np.power(((X * theta.T) - y), 2)
#     return np.sum(inner) / (2 * len(X))


# def gradientDescent(X, y, theta, alpha, iters):
#     '''
#     梯度下降算法
#     X: 自变量
#     y: 因变量
#     theta: 参数向量
#     alpha: 学习率
#     iters: 计算次数
#     '''
#     # 暂存参数向量
#     temp = np.matrix(np.zeros(theta.shape))

#     # 将参数向量降为一维，返回视图，可以修改原始的参数向量
#     parameters = int(theta.ravel().shape[1])

#     # 损失值消耗记录
#     cost = np.zeros(iters)

#     # 梯度下降的计算
#     for i in range(iters):
#         error = (X * theta.T) - y

#         for j in range(parameters):
#             term = np.multiply(error, X[:, j])
#             temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

#         theta = temp
#         cost[i] = computeCost(X, y, theta)
#     return theta, cost


# # append a ones column to the front of the data set
# data.insert(0, 'Ones', 1)

# # set X (training data) and y (target variable)
# cols = data.shape[1]
# X = data.iloc[:, 0:cols - 1]
# y = data.iloc[:, cols - 1:cols]


# # convert from data frames to numpy matrices
# X = np.matrix(X.values)
# y = np.matrix(y.values)
# theta = np.matrix(np.array([0, 0]))

# # initialize variables for learning rate and iterations
# alpha = 0.01
# iters = 1000

# # perform gradient descent to "fit" the model parameters
# g, cost = gradientDescent(X, y, theta, alpha, iters)

# x = np.linspace(data.Population.min(), data.Population.max(), 100)
# f = g[0, 0] + (g[0, 1] * x)

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()

# 查看损失值的变化
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(iters), cost, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')

'''
使用代码实现线性回归算法(多元)
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.dirname(os.getcwd()) + '\data\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std()


def computeCost(X, y, theta):
    '''
    损失函数
    X: 自变量
    y: 因变量
    theta: 参数向量
    '''
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    '''
    梯度下降算法
    X: 自变量
    y: 因变量
    theta: 参数向量
    alpha: 学习率
    iters: 计算次数
    '''
    # 暂存参数向量
    temp = np.matrix(np.zeros(theta.shape))

    # 将参数向量降为一维，返回视图，可以修改原始的参数向量
    parameters = int(theta.ravel().shape[1])

    # 损失值消耗记录
    cost = np.zeros(iters)

    # 梯度下降的计算
    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

# initialize variables for learning rate and iterations
alpha = 0.01
iters = 1000

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
print(computeCost(X2, y2, g2))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()