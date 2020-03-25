# random01.py

import tensorflow as tf

# 난수 생성
a = tf.random.uniform([1], 0, 1) # 0 ~ 1 사이의 난수 1개 발생
b = tf.random.uniform([1], 0, 10) # 0 ~ 10 사이의 난수 1개 발생
print(a.numpy())
print(b.numpy())
""" 난수 생성 """
print()

# 정규분포 난수 : 평균: -1, 표준편차: 4
norm = tf.random.normal([2, 3], mean=-1, stddev=4)
print(norm.numpy())
print()

# 주어진 값들을 shuffle()함수로 무작위로 섞음
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random.shuffle(c)
print(shuff.numpy())
print()

# 균등분포 난수 : 0 ~ 3 값 사이의 2행 3열 난수 발생
unif = tf.random.uniform([2,3], minval=0, maxval=3)
print(unif.numpy())