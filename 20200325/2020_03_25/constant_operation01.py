# constant_operation01.py

import tensorflow as tf

# 상수 정의
a = tf.constant(2)
b = tf.constant(3)
c = a+b

# 연산 결과 출력
print(a.numpy())
print(b.numpy())
print(c.numpy())
print(a.numpy()+b.numpy())