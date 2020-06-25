import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 텐서플로우에 기본 내장된 mnist 모듈을 이용하여 데이터를 로드
# 지정한 폴더에 MNIST 데이터가 없는 경우 자동으로 데이터를 다운로드
# one_hot 옵션은 레이블을 one_hot 방식의 데이터로 만들어 준다.

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels

x_val = mnist.test.images
y_val = mnist.test.labels

print(x_train[0].shape)  # (784, )
print(y_train[0].shape)  # (10, )
print(x_val[0].shape)    # (784, )
print(y_val[0].shape)    # (10, )

for i in range(10):
    #number = np.argmax(y_train[i])
    for index in range(10):
        if (y_train[i, index] == 1):
            number = index

    mnist_sample = np.reshape(x_train[i], [28,28])
    plt.subplot(2,5,i+1)
    plt.imshow(mnist_sample)
    plt.gray()
    plt.title(str(number))
    plt.axis('off')

plt.show()