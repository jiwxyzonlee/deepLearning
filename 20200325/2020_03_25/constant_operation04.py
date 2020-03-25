# constant_operation04.py

import tensorflow as tf

# 상수 선언
x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])

# 연산
cal1 = tf.add(x1, x2) # 더하기
cal2 = tf.subtract(x1, x2) # 빼기
cal3 = tf.multiply(x1, x2) # 곱하기
cal4 = tf.divide(x1, x2) # 나누기

# 연산 결과 출력
print(cal1.numpy())
print(cal2.numpy())
print(cal3.numpy())
print(cal4.numpy())