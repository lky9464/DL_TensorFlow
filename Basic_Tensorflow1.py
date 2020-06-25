import tensorflow as tf

hello = tf.constant('Hello, TensorFlow')


a = tf.constant(15)
b = tf.constant(30)

c1 = a + b
c2 = tf.add(a, b)


print(c1)


sess = tf.Session()     # 세션 열기

print(sess.run(hello))
print(sess.run([a, b, c1]))

sess.close()