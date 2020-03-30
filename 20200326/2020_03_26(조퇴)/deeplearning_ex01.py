# deeplearning_ex01.py

# 폐암 수술환자의 생존 예측

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 실행할 때마다 같은 결과를 출력하기 위한 난수 시드 설정
# seed를 한 번 쓸 때와 두 번 쓸 때 결과 비교해보기
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 로딩
dataset = np.loadtxt('dataset/ThoraricSurgery.csv', delimiter=',')

x = dataset[:, 0:17]
# 모든 행데이터, 인덱스 0-16번까지 환자 정보 데이터 슬라이싱

y = dataset[:, 17]
# 17번 환자의 생존 유무(0 or 1), 모든 행데이터만

# 모델 생성
model = Sequential()
# 은닉층 : 출력 node=30, 입력 node=17
model.add(Dense(30, input_dim=17, activation='relu'))
# 출력층 : (이중분류는 sigmoid, 다중분류는 softmax)
model.add(Dense(1, activation='sigmoid'))

# Sigmoid 함수는 S와 같은 형태로 미분 가능한 0~1 사이의 값을 반환
# Logistic Classification과 같은 분류 문제의 가설과 비용 함수(Cost Function)에 많이 사용

# node는 neurons, unit으로 불리기도 함

# 학습 과정 설정
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(x, y, epochs=30, batch_size=10)
#470개의 데이터를 10 batch에 나눔

ac = model.evaluate(x, y)
print(type(ac))
print(ac)
"""
<class 'list'>
[0.1401861772892323, 0.85531914]
"""

# 모델평가 (평가지표는 정확도)
print('Accuracy: %.4f' %(model.evaluate(x, y)[1])) # 1은 accuracy 값을 의미
# Accuracy: 0.8553