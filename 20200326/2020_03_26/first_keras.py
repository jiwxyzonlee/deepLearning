import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 난수 시드 설정
np.random.seed(1234)

# 학습 데이터
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 학습 모델 생성
model = Sequential()
model.add(Dense(1, input_dim=1))                 # 출력 node 1개, 입력 node 1개

# 모델 학습 방법 설정   # loss: 평균제곱 오차
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))

# 모델 학습
model.fit(x_train, y_train, epochs=20000)

# 모델을 이용해서 예측
y_predict = model.predict(np.array([1,2,3,4]))
print(y_predict)

y_predict = model.predict(np.array([7,8,9,100]))
print(y_predict)
