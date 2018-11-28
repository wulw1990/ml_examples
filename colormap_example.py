import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 假设有一张label图，每个像素的值表示一个label值，如何显示这幅图像呢？
# 这里我们随便初始化一个label图作为演示
label = np.zeros(shape=(100, 500), dtype=np.int32)
label[:,0:100] = 0
label[:,100:200] = 1
label[:,200:300] = 2
label[:,300:400] = 3
label[:,400:500] = 4

# 方法1：显示灰度图:，为了观察方便，需要归一化到0-255之间
gray = (label.astype(np.float) / 4 * 255).astype(np.uint8)
cv2.imshow('gray', gray)

# 方法2：通过colormap映射为彩色图
# 这里我们演示用matplotlib来进行colormap，该库有很多种colormap类型，参考：
# https://matplotlib.org/examples/color/colormaps_reference.html
# 这里我们使用plasma这种类型的映射（感觉比较好看）
# matplotlib.cm.get_cmap是通过一个名字来获取一个映射函数，注意返回值是一个函数
colormap = matplotlib.cm.get_cmap('plasma')
# 然后我们用这个返回的函数来对图像进行转换，把一个0-1之间的单通道图像，转换成0-1之间的3通道图像
color = colormap(label.astype(np.float)/4)
# 然后我们把这个0-1之间的三通道图像先行映射到0-255之间，用于显示
color = (color * 255).astype(np.uint8)
cv2.imshow('color', color)
cv2.waitKey(0)