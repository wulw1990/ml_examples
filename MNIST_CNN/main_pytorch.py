## PyTorch 简介安装 -------------------------------------------------------------
'''
PyTorch优点：
    1，动态图
    2，Tensor Numpy互转
    3，pythonic
    4，debug方便
    5，可读性强
    6，上手快，适合research
    7，轻量级
    PyTorch和keras一样的易于上手，但是避免了keras对具体细节的屏蔽导致的调试和研究的困难。

安装Conda:
    建议采用conda环境来管理python以及组件的版本。
    https://conda.io/miniconda.html

安装PyTorch
    pytorch目前支持linux, mac, windows平台，安装方法见官网。
    http://pytorch.org/

    对于windows平台+conda环境+python3.6+CPU，安装方法为：
    conda install pytorch-cpu -c pytorch
    pip3 install torchvision
'''

import torchvision
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

## 初始化Transform --------------------------------------------------------------
'''
Transform一般用于对数据进入模型前进行一些预处理，比如：归一化（减均值除方差）、resize、RGB2GRAY等。
Transform是pytorch框架中一个很实用的概念，pytorch通过transform概念的的抽象，使得数据预处理
变得非常方便，添加不同的预处理只是在Compose的构造函数中添加对应的transform对象即可。其他的深度
学习框架也会有类似的概念，例如tensorflow的：https://github.com/tensorflow/transform

pytorch 中预定义了一些常用的Transform，需要的时候也可以很方便自定义Transform。
官方文档：
    英文：http://pytorch.org/docs/master/torchvision/transforms.html
    中文：https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-transform/
如文档所述，ToTensor 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloatTensor。H指的是图片的高度，W指的是图片的宽度，
C指的是图片的通道数量。

对于本示例，所采用的是mnist数据。mnist原始数据格式是28x28的gray8图像，我们只需要一个tranform，即ToTensor。
所以本例中通过transform的图像，是一个[W,H,C]=[28,28,1]，且取值范围为[0.0, 1.0]之间。
有兴趣可以参照官方文档，尝试不同的初始化方式对最终结果的影响。
'''
transform = transforms.Compose([transforms.ToTensor()])

## 初始化Dataset ----------------------------------------------------------------
'''
Dataset是pytorch对数据集读取格式的一个抽象概念，使得“如何读取一种数据集”变成了“如何实现一个Dataset类”的问题。
pytorch官方实现了很多常见的Dataset类型，例如专用于读取mnist数据的torchvision.datasets.MNIST。
除了MNIST，官方还实现了很多用于读取其他类型数据集的Dataset:
http://pytorch.org/docs/master/torchvision/datasets.html
需要的时候也可以很方便自定义Dataset。

对于本示例，采用了MNIST Dataset。MNIST类首先检查指定的本地路径下是否存在所需数据，如果存在则直接使用本地数据，否则会自动到数据网站下载所需数据，
例如：http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
下载的数据（data/raw），格式是符合mnist官方定义的格式，pytorch的MNIST类下载数据后，
会按照官方格式进行解读，并按照新的格式存储到指定的目录中（data/processed），
下次使用时直接使用新的存储格式的数据即可。MNIST的官方代码：
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

MNIST Dataset的构造函数可以指定一些参数，比如：
 - root 表示本地出具的存放目录
 - transform 表示数据读取后进行的transform操作，见上文定义
 - train 表示是否是训练数据
 - download 表示如果本地数据不存在，则自动进行下载
其他参数可以参考官方文档。
'''
data_train = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)

print('samples train:', len(data_train))
print('samples test:', len(data_test))
image, label = data_test[0]
print('image size:', image.shape)
print('label:', label)


## 初始化DataLoader -------------------------------------------------------------
'''
DataLoader是pytorch对数据集进行数据读取操作的一个概念。注意：本质上，Dataset类定义了数据集的格式，
而DataLoader定义的数据读取的流程。pytorch或者tensorflow这类现代框架，对这两个概念进行的区分，
使得支持新的数据集类型变得非常方便，因为大部分时候数据读取的流程都是一样的，只是数据集的格式定义不同。
多以大部分时候，不需要重新实现DataLoader，而只需要实现新的Dataset，即可支持新的数据集。
http://pytorch.org/docs/master/data.html

DataLoader的构造函数支持很多参数，例如：
 - dataset 是最关键的，指定了DataLoader将要读取的数据集。
 - batch_size 表示每次将多少样本传入模型，同时处理。
 - shuffle 表示每次进行新一轮数据读取的时候是否对所有样本进行打乱操作。一般是训练时打开。
其他参数可以参考官方文档。
'''
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 64,
                                               shuffle = False)

## 定义模型的网络结构 -------------------------------------------------------------
'''
Module是pytorch的一个核心概念。Module通常用于表示网络的一个层、一个子网络、一个网络、一个模型。
实际上，Module的本质在于：有前向运算、有反向运算、本部可能有参数。所以Module可以通过用数学表示为：
    前向（forward）：y = f(x; w)
    反向（backward）：dy/dx = ? dy/dw = ?
比如，对于普通神经网络，每一层实际是一个全连接操作（即输出的每个数都会依赖前一层所有的数），在pytorch中
这个全连接的操作过程，就是由Linear这个Module来完成。对于一个普通神经网络的多隔层合在一起的网络，
也可以通过定义一个Module来完成。

值得一提的是，虽然每个Module包含了前向和反向的操作，但是反传的过程其实就是前传函数的求导。
因此，pytorch底层实现了自动求导（反传），因此带部分情况下我们只需要实现forward函数即可。

对于本示例，我们一定一个非常简单的多层神经网络（MLP），该网络：输入为28x28=784维度，后面接了3个
全连接层，最后得到10维的输出（对应10个不同可能的数字）。

另外，为了展示卷积神经网络(CNN)的使用，我们也实现了一个简单的LeNet。

注意：对于Linear、Conv2d这类带参数的Module，使用时，应该在self字段进行“注册”，因为pytorch
会根据self存储的字段来知道哪些Module内部包含需要训练或使用的参数。而对于relu、mxal_pool2d这类
则直接使用即可。
'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## 初始化模型、优化器、损失函数 ----------------------------------------------------
'''
optimizer是pytorch中另一个核心概念，表示参数更新的策略，常用的是SGD，即“随机梯度下降”。
http://pytorch.org/docs/master/optim.html

loss function是反向传递过程的起点，实际上参数更新时使用的就是每个参数相对于loss的导数。
optimizer的本质是借助各个参数对loss的导数对参数进行更新，更新的目标是是的下一次前传的时候loss
尽量的小。
'''
# model = MLP()
model = LeNet()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
ceriation = nn.CrossEntropyLoss()

## 训练过程 ---------------------------------------------------------------------
'''
epoch: 一个epoch表示数据集中所有数据都已经输入到模型中，称为完成一轮训练。
iter: 一个iter表示一个batch的数据输入到模型中，称为完成一次迭代。
显然一个epoch过程中会有过个iter，例如：总计6400个样本，batch_size=64，则一个epoch包含100个iter。
整个训练过程一般包含多个epoch，我们按照惯例，每次训练epoch结束，进行一次测试epoch，来观察当前模型的精度。
可以尝试使用不同的模型来观察精度随epoch变化的情况。
当前设置下，10epoch后，MPL的精度大约为98.0%，LeNet的精度约为99.2%
'''
for epoch in range(10):
    model.train()
    for batch_idx, (x, target) in enumerate(data_loader_train):
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        y = model(x)
        loss = ceriation(y, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(data_loader_train):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.
                format(epoch, batch_idx+1, loss))

    # 为了方便知道训练的进度，没一轮(epoch)训练完，进行一次测试
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(data_loader_test):
        x, target = Variable(x), Variable(target)
        y = model(x)
        loss = ceriation(y, target)
        _, pred_label = torch.max(y.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
    print('==>>> epoch: {}, acc: {:.3f}%'.format(
        epoch, float(correct_cnt) / total_cnt * 100))
