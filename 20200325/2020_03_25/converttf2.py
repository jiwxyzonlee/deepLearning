# converttf2.py

# 텐서플로 1.X 버전의 코드를 수정하지 않고 텐서플로 2.0에서 실행할 수 있다.
# 텐서플로 1.X 파일에 아래의 코드 2줄을 추가하면 텐서플로 2.0에서 실행 가능하다.

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 상수 선언
hello = tf.constant('Hello, Tensorflow')
# 세션 시작
sess = tf.Session()
# 세션 실행
print(sess.run(hello))

