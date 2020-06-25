import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
1. 신경망 모델
2. 비용(cost), 최적화(optimization)
3. 초기화, 세션열기, 학습
4. 결과확인, 정확도 계산
"""

tf.set_random_seed(2019)    # for reproduction

x_data = np.array([[160, 47], [165, 45], [163, 60], [157, 61], [155, 65],
                   [172, 60], [165, 65], [175, 80], [180, 70], [178, 120]])

y_data = np.array([ [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 1, 0],
                    [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1] ])

# placeholder 정의
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 4])

# weights, bias 정의
w1 = tf.Variable(tf.random_normal(shape=[2,10]), dtype=tf.float32)
w2 = tf.Variable(tf.random_normal(shape=[10,6]), dtype=tf.float32)
w3 = tf.Variable(tf.random_normal(shape=[6,4]), dtype=tf.float32)

b1 = tf.Variable(tf.random_normal(shape=[10]), dtype=tf.float32)
b2 = tf.Variable(tf.random_normal(shape=[6]), dtype=tf.float32)
b3 = tf.Variable(tf.random_normal(shape=[4], dtype=tf.float32))

# hyperparameter
steps = 1000
lr = 0.001

# training
z1 = tf.matmul(X,w1) + b1
a1 = tf.nn.relu(z1)

z2 = tf.matmul(a1,w2) + b2
a2 = tf.nn.relu(z2)

hypothesis = tf.matmul(a2,w3) + b3


# check
model = tf.nn.softmax(logits=hypothesis, axis=1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target), dtype=tf.float32))*100

optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # cost를 step 별로 저장
    steps_cost = []
    steps_acc = []

    for step in range(steps):
        _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})
        steps_cost.append(c)
        steps_acc.append(a)
        print('step:{0}, cost:{1:.5f}, accuracy:{2:.3f}'.format(step, c, a))

    sess.close()


# plot costs
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].set(title='Cost per steps')
axes[1].set(title='Acc per steps')

axes[0].plot(np.arange(steps), steps_cost, c='g')
axes[1].plot(np.arange(steps), steps_acc, c='b')
plt.show()