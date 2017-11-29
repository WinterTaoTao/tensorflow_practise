import tensorflow as tf

sess = tf.Session()

c = tf.truncated_normal(shape=[5, 10, 8, 2], mean=0, stddev=1)

sess = tf.Session()
print sess.run(c)
