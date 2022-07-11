from __future__ import absolute_import, division, print_function, unicode_literals

# import matplotlib.pyplot as plt
from matplotlib import image
import tensorflow as tf
import numpy as np
import scipy.misc
import os

# 读取MNIST数据集。如果不存在会事先下载
# mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

print(train_labels)
print("ok")
# 把原始图片保存在MNIST_data/raw/文件夹下 如果没有这个文件夹  会自动创建
save_dir = 'MNIST_data/raw/train/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

# 把原始图片保存在MNIST_data/raw/文件夹下 如果没有这个文件夹  会自动创建
save_ev_dir = 'MNIST_data/raw/eval/'
if os.path.exists(save_ev_dir) is False:
    os.mkdir(save_ev_dir)

# 保存图片
for i in range(len(train_images)):
    # 请注意，mnist.train.images[i, :]就表示第i张图片（序号从0开始）
    image_array = train_images[i, :]
    # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    save_train_dir = 'MNIST_data/raw/train/' + str(train_labels[i]) + '/'
    if os.path.exists(save_train_dir) is False:
        os.mkdir(save_train_dir)

    filename = save_train_dir + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    image.imsave(filename, image_array, cmap='gray')  # cmap常用于改变绘制风格，如黑白gray，翠绿色virdidis

# 保存图片
for i in range(len(test_images)):
    # 请注意，mnist.train.images[i, :]就表示第i张图片（序号从0开始）
    image_array = test_images[i, :]
    # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    save_eval_dir = 'MNIST_data/raw/eval/' + str(test_labels[i]) + '/'
    if os.path.exists(save_eval_dir) is False:
        os.mkdir(save_eval_dir)

    filename = save_eval_dir + 'mnist_eval_%d.jpg' % i
    # 将image_array保存为图片
    image.imsave(filename, image_array, cmap='gray')  # cmap常用于改变绘制风格，如黑白gray，翠绿色virdidis
