# variable02.py

import tensorflow as tf

# 변수 선언
v1 = tf.Variable(50) # v1 변수의 초기값 50
v2 = tf.Variable([1,2,3]) # rank: 1, shape: (3)
v3 = tf.Variable([[1],[2]]) # rank: 2, shape: (2,1)

# 변수 출력
print('v1=', v1.numpy()) # v1= 50
print('v2=', v2.numpy()) # v2= [1 2 3]
print('v3=\n', v3.numpy()) # v3= [[1]//[2]] (2행 1열임)