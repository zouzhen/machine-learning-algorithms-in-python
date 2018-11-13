import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
argparse是python的一个包，用来解析输入的参数
如：
    python mnist.py --outf model  
    （意思是将训练的模型保存到model文件夹下，当然，你也可以不加参数，那样的话代码最后一行
      torch.save()就需要注释掉了）

    python mnist.py --net model/net_005.pth
    （意思是加载之前训练好的网络模型，前提是训练使用的网络和测试使用的网络是同一个网络模型，保证权重参数矩阵相等）
'''
parser = argparse.ArgumentParser()

parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 模型保存路径
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  # 模型加载路径
opt = parser.parse_args()  # 解析得到你在路径中输入的参数，比如 --outf 后的"model"或者 --net 后的"model/net_005.pth"，是作为字符串形式保存的

# Load training and testing datasets.
ROOT_PATH = "./traffic"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")
pic_dir = './picture/3.jpg'

# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()


def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    tmp = []
    img = skimage.io.imread(picture_dir)
    tmp.append(img)
    img = skimage.io.imread('./picture/4.jpg')
    tmp.append(img)
    img256 = [skimage.transform.resize(img, (256, 256)) for img in tmp]
    # img256 = skimage.transform.resize(img, (256, 256))
    # img256 = img256.unsqueeze(0)
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)

    return transform(img256[0])


def get_picture_rgb(picture_dir):
    '''
    该函数实现了显示图片的RGB三通道颜色
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    skimage.io.imsave('./picture/4.jpg',img256)

    # 取单一通道值显示
    # for i in range(3):
    #     img = img256[:,:,i]
    #     ax = plt.subplot(1, 3, i + 1)
    #     ax.set_title('Feature {}'.format(i))
    #     ax.axis('off')
    #     plt.imshow(img)

    # r = img256.copy()
    # r[:,:,0:2]=0
    # ax = plt.subplot(1, 4, 1)
    # ax.set_title('B Channel')
    # # ax.axis('off')
    # plt.imshow(r)

    # g = img256.copy()
    # g[:,:,0]=0
    # g[:,:,2]=0
    # ax = plt.subplot(1, 4, 2)
    # ax.set_title('G Channel')
    # # ax.axis('off')
    # plt.imshow(g)

    # b = img256.copy()
    # b[:,:,1:3]=0
    # ax = plt.subplot(1, 4, 3)
    # ax.set_title('R Channel')
    # # ax.axis('off')
    # plt.imshow(b)

    # img = img256.copy()
    # ax = plt.subplot(1, 4, 4)
    # ax.set_title('image')
    # # ax.axis('off')
    # plt.imshow(img)

    img = img256.copy()
    ax = plt.subplot()
    ax.set_title('image')
    # ax.axis('off')
    plt.imshow(img)

    plt.show()


class LeNet(nn.Module):
    '''
    该类继承了torch.nn.Modul类
    构建LeNet神经网络模型
    '''
    def __init__(self):
        super(LeNet, self).__init__()

        # 第一层神经网络，包括卷积层、线性激活函数、池化层
        self.conv1 = nn.Sequential( 
            nn.Conv2d(3, 6, 5, 1, 2),   # input_size=(3*256*256)，padding=2
            nn.ReLU(),                  # input_size=(32*256*256)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(32*128*128)
        )

        # 第二层神经网络，包括卷积层、线性激活函数、池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 64, 5, 1, 2),  # input_size=(32*128*128)
            nn.ReLU(),            # input_size=(64*128*128)
            nn.MaxPool2d(2, 2)    # output_size=(64*64*64)
        )

        # 全连接层(将神经网络的神经元的多维输出转化为一维)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),  # 进行线性变换
            nn.ReLU()                    # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(84, 62)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# class LeNet(nn.Module):
#     '''
#     该类继承了torch.nn.Modul类
#     构建LeNet神经网络模型
#     '''
#     def __init__(self):
#         super(LeNet, self).__init__()  # 这一个是python中的调用父类LeNet的方法，因为LeNet继承了nn.Module，如果不加这一句，无法使用导入的torch.nn中的方法，这涉及到python的类继承问题，你暂时不用深究

#         # 第一层神经网络，包括卷积层、线性激活函数、池化层
#         self.conv1 = nn.Sequential(     # input_size=(1*28*28)：输入层图片的输入尺寸，我看了那个文档，发现不需要天，会自动适配维度
#             nn.Conv2d(3, 32, 5, 1, 2),   # padding=2保证输入输出尺寸相同：采用的是两个像素点进行填充，用尺寸为5的卷积核，保证了输入和输出尺寸的相同
#             nn.ReLU(),                  # input_size=(6*28*28)：同上，其中的6是卷积后得到的通道个数，或者叫特征个数，进行ReLu激活
#             nn.MaxPool2d(kernel_size=2, stride=2), # output_size=(6*14*14)：经过池化层后的输出
#         )

#         # 第二层神经网络，包括卷积层、线性激活函数、池化层
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, 5),  # input_size=(6*14*14)：  经过上一层池化层后的输出,作为第二层卷积层的输入，不采用填充方式进行卷积
#             nn.ReLU(),            # input_size=(16*10*10)： 对卷积神经网络的输出进行ReLu激活
#             nn.MaxPool2d(2, 2)    # output_size=(16*5*5)：  池化层后的输出结果
#         )

#         # 全连接层(将神经网络的神经元的多维输出转化为一维)
#         self.fc1 = nn.Sequential(
#             nn.Linear(64 * 5 * 5, 128),  # 进行线性变换
#             nn.ReLU()                    # 进行ReLu激活
#         )

#         # 输出层(将全连接层的一维输出进行处理)
#         self.fc2 = nn.Sequential(
#             nn.Linear(128, 84),
#             nn.ReLU()
#         )

#         # 将输出层的数据进行分类(输出预测值)
#         self.fc3 = nn.Linear(84, 62)

#     # 定义前向传播过程，输入为x
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

# 定义测试数据集
testset = tv.datasets.Traffic(
    root=test_data_dir, # 如果从本地加载数据集，对应的加载路径
    train=True,     # 训练模型
    download=True,  # 是否从网络下载训练数据集
    transform=transform  # 数据的转换形式
)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,                 # 加载测试集
    batch_size=1,   # 最小批处理尺寸
    shuffle=False,           # 标识进行数据迭代时候不将数据打乱
)

# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        print(self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if "fc" in name: 
                print(name)
                x = x.view(x.size(0), -1)
            print(module)
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def get_feature():
    # 输入数据
    img = get_picture(pic_dir, transform)
    # img = transform([img])
    img = img.unsqueeze(0)
    # img,label=iter(testloader).next()
    print(img.shape,len(img))
    # print(img,label)
    img = img.to(device)

    # 特征输出
    net = LeNet().to(device)
    # net.load_state_dict(torch.load('./model/net_050.pth')) 
    exact_list = ["conv1"]
    myexactor = FeatureExtractor(net, exact_list)
    x = myexactor(img)

    # 特征输出可视化
    # for i in range(6):
    #     ax = plt.subplot(2, 3, i + 1)
    #     ax.set_title('Feature {}'.format(i))
    #     ax.axis('off')
    #     plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
    ax = plt.subplot()
    ax.set_title('MaxPool2d Feature Map')
    # ax.axis('off')
    plt.imshow(x[0].data.numpy()[0,1,:,:], cmap='jet')

    plt.show()

# 训练
if __name__ == "__main__":
    get_picture_rgb(pic_dir)
    # get_feature()
    