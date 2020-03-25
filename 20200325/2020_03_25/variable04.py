# variable04.py

import tensorflow as tf

# 변수 선언
v1 = tf.Variable(tf.zeros([2,3])) # [[ 0. 0. 0.] [ 0. 0. 0.]]
v2 = tf.Variable(tf.ones([2,3], tf.int32)) # [[1 1 1] [1 1 1]]
v3 = tf.Variable(tf.zeros_like(tf.ones([2,3]))) # [[ 0. 0. 0.] [ 0. 0. 0.]]
v4 = tf.Variable(tf.fill([2,3], 2)) # [[2 2 2] [2 2 2]]
v5 = tf.Variable(tf.fill([2,3], 2.0)) # [[ 2. 2. 2.] [ 2. 2. 2.]]

# 변수 출력
print(v1.numpy())
print(v2.numpy())
print(v3.numpy())
print(v4.numpy())
print(v5.numpy())