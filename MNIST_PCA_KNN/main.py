import numpy as np
import sys
import struct
import cv2
import random

## 按照MNIST官方文档的格式读取数据 -------------------------------------------------
# http://yann.lecun.com/exdb/mnist/
# 需要从官方网站下载四个文件，并解压到当前目录的data目录中。
print('*'*20, 'Read', '*'*20)
def load_mnist(image_file, label_file):
    '''
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
    0004     32 bit integer  60000/10000      number of items 
    0008     unsigned byte   ??               label 
    0009     unsigned byte   ??               label 
    ........ 
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    '''
    with open(label_file, 'rb') as fin:
        magic, num = struct.unpack(">II", fin.read(8))
        label = np.fromfile(fin, dtype=np.uint8)
    
    '''
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000/10000      number of images 
    0008     32 bit integer  28               number of rows 
    0012     32 bit integer  28               number of columns 
    0016     unsigned byte   ??               pixel 
    0017     unsigned byte   ??               pixel 
    ........ 
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 
    0 means background (white), 255 means foreground (black).
    '''
    with open(image_file, 'rb') as fin:
        magic, num, rows, cols = struct.unpack(">IIII", fin.read(16))
        image = np.fromfile(fin, dtype=np.uint8).reshape(num, rows, cols)

    return image, label

train_image, train_label = load_mnist(
    'data/train-images.idx3-ubyte',
    'data/train-labels.idx1-ubyte')

test_image, test_label = load_mnist(
    'data/t10k-images.idx3-ubyte',
    'data/t10k-labels.idx1-ubyte')

print(test_image.shape)
print(test_label.shape)
print(train_image.shape)
print(train_label.shape)


## 随机显示一些图像  --------------------------------------------------------------
print('*'*20, 'Show', '*'*20)
for _ in range(5):
    i = random.randint(0, train_image.shape[0]-1)
    img = train_image[i, :, :]
    img = cv2.resize(img, (7, 7))
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()


## 准备训练和测试：把图像拉直成一行 -------------------------------------------------
print('*'*20, 'Reshape', '*'*20)
train_data = train_image.copy()
train_data.shape = (train_data.shape[0], train_data.shape[1]*train_data.shape[2])

test_data = test_image.copy()
test_data.shape = (test_data.shape[0], test_data.shape[1]*test_data.shape[2])

print(train_data.shape)
print(test_data.shape)


## PCA -------------------------------------------------------------------------
print('*'*20, 'PCA', '*'*20)
class PCA:
    def __init__(self, data, k):
        # 数据进行一次深度拷贝，从而不影响原来数据
        data = data.copy()

        # PCA step 1: 零均值化
        data = data.astype("float64")
        self.mean = np.mean(data, axis=0)
        data -= self.mean

        # PCA step 2: 求协方差矩阵
        cov = np.cov(data, rowvar=0)

        # PCA step 3: 求特征值、特征矩阵
        eigen_value, eigen_vector = np.linalg.eig(np.mat(cov))

        # PCA step 4: 保留主要的成分[即保留值比较大的前k个特征]
        indice = np.argsort(-eigen_value)       #对特征值从大到小排序
        indice = indice[:k]                     #最大的n个特征值的下标
        eigen_vector = eigen_vector[:,indice]   #最大的n个特征值对应的特征向量  
        self.eigen_vector = eigen_vector
    
    def forward(self, data):
        return (data - self.mean) * self.eigen_vector

    def backward(self, data):
        return (data * self.eigen_vector.T) + self.mean

# 验证PCA是否正确运行
print('pca example begin')
data1 = train_data.copy()
pca = PCA(data1, 50)
data2 = pca.forward(data1)
data3 = pca.backward(data2)
print('pca mean error:', np.sum(np.abs(data3[0,:] - data1[0,:]))/784)
print('pca example end')

# 使用train数据计对PCA进行初始化, 然后对train和test数据集都进行降维
print('pca init on train')
pca = PCA(train_data, 50)
print('pca dimension reduction on train and test')
train_data = pca.forward(train_data)
test_data = pca.forward(test_data)
print('pca ok')


## KNN -------------------------------------------------------------------------
print('*'*20, 'KNN', '*'*20)
class KNN:
    def __init__(self, data, labels, k):
        self.data = data
        self.labels = labels
        self.k = k
    def predict(self, sample):
        differences = (self.data - sample)
        distances = np.einsum('ij, ij->i', differences, differences)
        nearest = self.labels[np.argsort(distances)[:self.k]]
        counts = np.bincount(nearest)
        return np.argmax(counts)

# KNN的训练（时间复杂度为0）
knn = KNN(train_data, train_label, 5)
# print(knn.predict(test_data[0]), test_label[0])

# KNN的测试
test_predict = np.zeros((test_data.shape[0]), dtype=np.uint8)
for i in range(test_data.shape[0]):
    test_predict[i] = knn.predict(test_data[i])
    if i%100 == 0:
        print('index:{:}/{:} predict:{:} label:{:}'.format(
            i, test_data.shape[0], test_predict[i], test_label[i]))
accuracy = sum(test_predict==test_label) / test_data.shape[0]
print('accuracy: {:0.2f}%'.format(accuracy*100))

