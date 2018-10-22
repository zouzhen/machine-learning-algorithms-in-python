'''
使用sklearn包中的逻辑回归算法
'''
# from time import time
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import linear_model
# from sklearn import datasets
# from sklearn.svm import l1_min_c

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# X = X[y != 2]
# y = y[y != 2]

# X /= X.max()  # Normalize X to speed-up convergence

# # #############################################################################
# # Demo path functions

# cs = l1_min_c(X, y, loss='log') * np.logspace(0, 7, 16)


# print("Computing regularization path ...")
# start = time()
# clf = linear_model.LogisticRegression(penalty='l1', solver='saga',
#                                       tol=1e-6, max_iter=int(1e6),
#                                       warm_start=True)
# coefs_ = []
# for c in cs:
#     clf.set_params(C=c)
#     clf.fit(X, y)
#     coefs_.append(clf.coef_.ravel().copy())
# print("This took %0.3fs" % (time() - start))

# coefs_ = np.array(coefs_)
# plt.plot(np.log10(cs), coefs_, marker='o')
# ymin, ymax = plt.ylim()
# plt.xlabel('log(C)')
# plt.ylabel('Coefficients')
# plt.title('Logistic Regression Path')
# plt.axis('tight')
# plt.show()

# '''
# Plot multinomial and One-vs-Rest Logistic Regression
# '''
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.linear_model import LogisticRegression

# # make 3-class dataset for classification
# centers = [[-5, 0], [0, 1.5], [5, -1]]
# X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
# transformation = [[0.4, 0.2], [-0.4, 1.2]]
# X = np.dot(X, transformation)

# for multi_class in ('multinomial', 'ovr'):
#     clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
#                              multi_class=multi_class).fit(X, y)

#     # print the training scores
#     print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))

#     # create a mesh to plot in
#     h = .02  # step size in the mesh
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))

#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#     plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
#     plt.axis('tight')

#     # Plot also the training points
#     colors = "bry"
#     for i, color in zip(clf.classes_, colors):
#         idx = np.where(y == i)
#         plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired,
#                     edgecolor='black', s=20)

#     # Plot the three one-against-all classifiers
#     xmin, xmax = plt.xlim()
#     ymin, ymax = plt.ylim()
#     coef = clf.coef_
#     intercept = clf.intercept_

#     def plot_hyperplane(c, color):
#         def line(x0):
#             return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
#         plt.plot([xmin, xmax], [line(xmin), line(xmax)],
#                  ls="--", color=color)

#     for i, color in zip(clf.classes_, colors):
#         plot_hyperplane(i, color)

# plt.show()

'''
代码实现(简单的逻辑回归)
'''
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.optimize as opt
# import os
# path = os.path.dirname(os.getcwd()) + '\data\ex2data1.txt'
# data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# positive = data[data['Admitted'].isin([1])]
# negative = data[data['Admitted'].isin([0])]

# # fig, ax = plt.subplots(figsize=(8, 6))
# # ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# # ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
# # ax.legend()
# # ax.set_xlabel('Exam 1 Score')
# # ax.set_ylabel('Exam 2 Score')
# # plt.show()

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# # nums = np.arange(-10, 10, step=1)

# # fig, ax = plt.subplots(figsize=(8,6))
# # ax.plot(nums, sigmoid(nums), 'r')
# # plt.show()

# def cost(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
#     second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
#     return np.sum(first - second) / (len(X))

# def gradient(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)

#     parameters = int(theta.ravel().shape[1])
#     grad = np.zeros(parameters)

#     error = sigmoid(X * theta.T) - y

#     for i in range(parameters):
#         term = np.multiply(error, X[:,i])
#         grad[i] = np.sum(term) / len(X)

#     return grad

# def predict(theta, X):
#     probability = sigmoid(X * theta.T)
#     return [1 if x >= 0.5 else 0 for x in probability]

# # add a ones column - this makes the matrix multiplication work out easier
# data.insert(0, 'Ones', 1)

# # set X (training data) and y (target variable)
# cols = data.shape[1]
# X = data.iloc[:, 0:cols-1]
# y = data.iloc[:, cols-1:cols]

# # convert to numpy arrays and initalize the parameter array theta
# X = np.array(X.values)
# y = np.array(y.values)
# theta = np.zeros(3)

# result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
# theta_min = np.matrix(result[0])
# predictions = predict(theta_min, X)
# correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
# accuracy = (sum(map(int, correct)) % len(correct))
# print ('accuracy = {0}%'.format(accuracy))

'''
代码实现(加入正则化)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
path = os.path.dirname(os.getcwd()) + '\data\ex2data1.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)

learningRate = 0.1
result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))

# print(costReg(theta2, X2, y2, learningRate))
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))