# variable03.py

import tensorflow as tf

# 변수 선언
x = tf.Variable([[2, 2, 2],[2, 2, 2]]) # rank: 2, shape(2,3) 2행 3열
y = tf.Variable([[3, 3, 3],[3, 3, 3]]) # rank: 2, shape(2,3) 2행 3열

# 연산
z1 = tf.add(x, y) # 덧셈
z2 = tf.subtract(x, y) # 뺄셈
z3 = tf.multiply(x, y) # 곱셈

z4 = tf.matmul(x, tf.transpose(y)) # matrix 곱셈 (tf.transpose() : 전치행렬)
x4 = tf.multiply(x, y)

z5 = tf.pow(x, 3) # 3제곱

# shape 구하기
print(x.get_shape()) # (2, 3)
print(y.get_shape()) # (2, 3)

# 연산 결과 출력
print('z1 =\n', z1.numpy()); print()
print('z2 =\n', z2.numpy()); print()
print('z3 =\n', z3.numpy()); print()
print('z4 =\n', z4.numpy()); print()
print('x4 =\n', x4.numpy()); print()
print('z5 =\n', z5.numpy()); print()