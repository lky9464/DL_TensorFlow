import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
1.
'''
x_data = np.zeros(10)
for i in range(1, 11):
    x_data[i - 1] = i

y_data = np.zeros(10)
for i in range(1, 11):
    y_data[i - 1] = i*10

W = tf.Variable(tf.random_uniform([1], -1., 1.))
b = tf.Variable(tf.random_uniform([1], -1., 1.))

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

H = W*X + b     # hypothesis
cost = tf.reduce_mean(tf.square(H - Y))     # 모든 mean value의 합
Opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # Optimizer *학습이 제대로 안되면 learning rate 조절
train_op = Opt.minimize(cost)               # cost최소화 함수


'''
2.
'''


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y:y_data})

        print(step, cost_val, sess.run(W), sess.run(b))


    print('\n=== TEST ===')
    print('X : 5, Y : ', sess.run(H, feed_dict={X : 5}))
    print('X : 2.5, Y : ', sess.run(H, feed_dict={X : 2.5}))