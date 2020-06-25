import tensorflow as tf

a = tf.placeholder("float")     # 데이터 저장장소 설정(a)
b = tf.placeholder("float")     # 데이터 저장장소 설정(b)w

y = tf.multiply(a, b)

sess = tf.Session()

print(sess.run(y, feed_dict={a : 2, b : 3}))