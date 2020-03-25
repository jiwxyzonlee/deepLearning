# constant_operation02.py

import tensorflow as tf

# 상수 정의
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(4)

# 연산 정의
cal1 = a + b * c
cal2 = (a + b) * c

# 연산 결과 출력
print(cal1.numpy())
print(cal2.numpy())