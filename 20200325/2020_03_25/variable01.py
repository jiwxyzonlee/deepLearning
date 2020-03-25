# variable01.py

# Tensorflow의 변수(variable)

import tensorflow as tf

# 변수 선언
# session 만들 필요 없이 바로 출력
v1 = tf.Variable(50) # v1 변수의 초기값 50

# 변수 출력
print('v1=', v1) # v1= <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=50>
print('v1=', v1.numpy()) # v1= 50