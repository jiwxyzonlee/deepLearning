# first.py
import tensorflow as tf

print(tf.__version__)

# 상수 선언
hello = tf.constant('Hello, TensorFlow!')
print(hello)
# tf.Tensor(b'Hello, TensorFlow!', shape=(), dtype=string)
print()

# numpy() 메소드를 사용하여 텐서의 값을 numpy데이터 타입으로 변환하여 출력
print(hello.numpy())
# b'Hello, TensorFlow!'
print()

# decode('utf-8')메소드로 bytes 클래스를 str 클래스로 변환
# 문자열 앞에 보이던 b가 사라짐
print(hello.numpy().decode('utf-8'))
print()

# 상수 선언 : 한글
hi = tf.constant('안녕')
# 텐서값으로 한글을 사용한 경우
print(hi)
# tf.Tensor(b'\xec\x95\x88\xeb\x85\x95', shape=(), dtype=string)
print()

print(hi.numpy())
# b'\xec\x95\x88\xeb\x85\x95'
print()

# decode('utf-8')메소드로 bytes 클래스를 str 클래스로 변환
print(hi.numpy().decode('utf-8'))
# 안녕