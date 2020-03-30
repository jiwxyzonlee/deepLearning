from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. 데이터셋 생성하기
# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터셋 젂처리 : 이미지 정규화 ( 0 ~ 255 --> 0 ~ 1 )
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print(y_train[0])           # MNIST의 첫번째 이미지(5)의 원핫인코딩 : [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=28*28, activation='relu')) # 입력 28*28 , 출력 node 64개
model.add(Dense(10, activation='softmax'))              # 입력 node 64개, 출력 node 10개 (0 ~ 9)

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=5, batch_size=32)

# 5. 학습과정 살펴보기
# 5번 학습후 loss와 accuracy 결과 출력
print('# train loss and accuracy #')
print(hist.history['loss'])
print(hist.history['accuracy'])

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('# evaluation loss and metrics #')
print(loss_and_metrics)