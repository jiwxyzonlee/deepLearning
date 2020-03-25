# constant_operation03.py

import tensorflow as tf

# 상수 정의
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

# 연산
node3 = tf.add(node1, node2)
node4 = tf.multiply(node1, node2)

# 연산 결과 출력
print(node1.numpy())
print(node2.numpy())
print(node3.numpy())
print(node4.numpy())