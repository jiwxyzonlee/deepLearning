# matrix.py

import tensorflow as tf

# 변수 선언
x = tf.Variable([[3., 3.]]) # shape: (1, 2) 1행 2열
y = tf.Variable([[2.],[2.]]) # shape: (2, 1) 2행 1열
mat = tf.matmul(x, y) # matrix 곱셈

# shape 구하기
print(x.get_shape()) # shape: (1, 2) 1행 2열
print(y.get_shape()) # shape: (2, 1) 2행 1열

# 연산 결과 출력
print(x.numpy())
print(y.numpy())
print(mat.numpy())