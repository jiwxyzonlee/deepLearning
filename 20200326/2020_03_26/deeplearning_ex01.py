# 폐암 수술환자의 생존율 예측하기

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import  Dense

# 실행할때 마다 같은 결과를 출력하기 위해서 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 로딩
dataset = np.loadtxt('dataset/ThoraricSurgery.csv', delimiter=',')

x = dataset[: , 0:17]       # 0 ~ 16번 환자정보
y = dataset[: , 17]         # 17번 환자생존 유뮤 ( 0 or 1 )

# 모델 생성
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))  # 은닉층 : 출력node30, 입력node 17
model.add(Dense(1, activation='sigmoid' ))             # 출력층

# 학습 과정 설정
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(x, y, epochs=30, batch_size=10)

ac = model.evaluate(x, y)
print(type(ac))                 # <class 'list'>
print(ac)                       # [0.1401861772892323, 0.85531914]

# 모델 평가 : 정확도
print('Accuracy: %.4f' %(model.evaluate(x, y)[1]))    # Accuracy: 0.8553




