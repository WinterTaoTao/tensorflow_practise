import tensorflow as tf

sess = tf.Session()

x = tf.constant([[[1, 3], [2, 4], [3, 5]], [[4, 6], [5, 7], [6, 8]]])
tf.reduce_sum(x)  # 6
tf.reduce_sum(x, 0)  # [2, 2, 2]
tf.reduce_sum(x, 1)  # [3, 3]
tf.reduce_sum(x, 1, keep_dims=True)  # [[3], [3]]
tf.reduce_sum(x, [0, 1])  # 6

print(sess.run(tf.reduce_sum(x, [0, 1])))
print(sess.run(x*x))
