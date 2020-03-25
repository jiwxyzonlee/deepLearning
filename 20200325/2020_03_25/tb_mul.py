#tb_mul.py
import tensorflow as tf
# 상수 선언
a = tf.constant(20, name="a")
b = tf.constant(30, name="b")
mul = a * b
# 세션 생성하기
sess = tf.Session()
# tensorboard 사용하기
# tensorboard 로그가 저장될 폴더(log_dir)가 생성된다.
tw = tf.summary.FileWriter("log_dir", graph=sess.graph)
# 세션 실행하기
print(sess.run(mul))