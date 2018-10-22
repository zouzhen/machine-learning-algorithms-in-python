'''
逻辑回归
'''
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable 
import matplotlib.pyplot as plt 


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2,1)
        self.sm = nn.Sigmoid()
    
    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x

logistic_model = LogisticRegression()

if torch.cuda.is_available():
    logistic_model.cuda()

criterion = nn.BCELoss()
optimezer = torch.optim.SGD(logistic_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50000):
    if torch.cuda.is_available():
        x = Variable(x_data).cuda()
        y = Variable(y_data).cuda()
    else:
        x = Variable(x_data)
        y = Variable(y_data)
    
    out = logistic_model(x)
    loss = criterion(out, y)
    print_loss = loss.data[0]
    mask = out.ge(0.5).float()
    correct = (mask == y).sum()
    acc = correct.data[0] / x.size(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print('*'*10)
        print('epoch {}'.format(epoch+1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))

w0, w1 = logistic_model.lr.weight[0]
w0 = w0.data[0]
w1 = w1.data[0]

b = logistic_model.lr.bias.data[0]
plot_x = np.arrange(30,100,0.1)
plot_y = (-w0 * plot_x -b)/ w1
plot.plot(plot_x, plot_y)
plt.show()