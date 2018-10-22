import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable 
import matplotlib.pyplot as plt 

'''
torch 一维线性回归算法
'''
# # 生成训练数据
# np.random.seed(10)
# x = np.linspace(0, 30, 20)
# y = x * 3 + np.random.normal(0, 5, 20)
# x = np.array(x, dtype=np.float32).reshape([20, 1])
# y = np.array(y, dtype=np.float32).reshape([20, 1])

# # 将数据转换为torch中的张量形式
# x_train = torch.from_numpy(x)
# y_train = torch.from_numpy(y)


# class LinearRegression(nn.Module):
#     '''
#     线性回归模型：一维线性回归
#     '''
#     def __init__(self):
#         super(LinearRegression, self).__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, x):
#         out = self.linear(x)
#         return out


# if torch.cuda.is_available():
#     model = LinearRegression().cuda()
# else:
#     model = LinearRegression()

# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# num_epochs = 1000
# for epoch in range(num_epochs):
#     if torch.cuda.is_available():
#         inputs = Variable(x_train).cuda()
#         target = Variable(y_train).cuda()
#     else:
#         inputs = Variable(x_train)
#         target = Variable(y_train)

#     optimizer.zero_grad()
#     out = model(inputs)
#     loss = criterion(out, target)
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 20 == 0:
#         print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.data[0]))

# model.eval()
# predict = model(Variable(x_train))
# predict = predict.data.numpy()

# plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
# plt.plot(x_train.numpy(), predict, label='predict data')
# plt.legend()
# plt.show()

'''
torch 一维线性回归算法(多项式回归)
'''
# 生成训练数据

def make_features(x):
    '''
    建立多项式特征
    '''
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
    '''
    实际函数
    '''
    return x.mm(W_target) + b_target[0]


def get_batch(batch_size=32):
    '''
    生成训练数据
    '''
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    # print(random,x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(random), Variable(x), Variable(y)

class poly_model(nn.Module):
    '''
    多项式线性回归（三维）
    '''
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out

if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epoch = 0

while True:
    _, batch_x, batch_y = get_batch()

    optimizer.zero_grad()
    out = model(batch_x)
    loss = criterion(out, batch_y)
    print_loss = loss.data[0]
    loss.backward()
    optimizer.step()
    print(print_loss)
    epoch += 1 
    if print_loss < 0.001:
        break


# model.eval()
# random, batch_x, batch_y = get_batch()
# predict = model(batch_x)
# predict = predict.data.numpy()
# print(random)
# plt.plot(random.numpy(), batch_y.numpy(), 'ro', label='Original data')
# plt.plot(random.numpy(), predict, label='predict data')
# plt.legend()
# plt.show()