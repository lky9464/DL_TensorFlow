import tensorflow as tf

hello = tf.constant('Hello TensroFlow')

x = tf.placeholder(tf.float32, [None, 3])       # None -> 크기가 정해지지 않음을 의미
x_data = [[1, 2, 3],[4, 5, 6]]


W = tf.Variable(tf.random.normal([3, 2]))
b = tf.Variable(tf.random.normal([2, 1]))


# 입력값과 변수들을 계산할 수식 작성
expr = tf.matmul(x, W) + b

sess = tf.Session()

# 위에서 설정한 Varible들의 값들을 초기화 하기위해
# 처음에 tf.global_variables_initializer를 한 번 실행
sess.run(tf.global_variables_initializer())

print('x_data')
print(x_data)
print('W')
print(sess.run(W))
print('b')
print(sess.run(b))
print('expr')
print(sess.run(expr, feed_dict={x: x_data}))

sess.close()