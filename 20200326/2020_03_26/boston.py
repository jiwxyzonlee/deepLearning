#  보스턴 집값 예측하기

from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

import numpy as np
import pandas as pd
import tensorflow as tf

# 난수 시드 설정
np.random.seed(0)
tf.random.set_seed(0)

# 데이터 불러오기
df = pd.read_csv('dataset/housing.csv', delim_whitespace=True, header=None)
print(df)                       # [506 rows x 14 columns]

# 자료형 확인
print(df.info())

dataset = df.values
x = dataset[:, 0:13]            # 집값에 영향을 주는 컬럼(0 ~ 12)
y = dataset[:, 13]              # 집값 ( 13 )

# 학습 데이터와 테스트 데이터를  7: 3 비율로 분할
x_train, x_test, y_train, y_test = train_test_split(x,             # 독립변수
                                                    y,             # 종속변수
                                                    test_size=0.3, # 테스트 데이터 30%비율
                                                    random_state=0)# 난수 시드 설정

# 모델 생성
model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))  # 입력층+은닉층
model.add(Dense(6, activation='relu'))                 # 은닉층
model.add(Dense(1))     # 예측의 경우에는 활성화 함수를 사용하지 않는다.

# 모델 학습 방식 설정
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.01))

# 모델 학습
model.fit(x_train, y_train, epochs=500, batch_size=10)      # 200번 학습

# 예측값과 실제값을 비교
y_prediction = model.predict(x_test).flatten()

for i in range(10):
    label = y_test[i]                     # 실제 데이터 가격
    prediction = y_prediction[i]          # 예측 데이터 가격

    # print('실제가격: {}, 예상가격: {}'.format(label, prediction))
    print('실제가격:', label, '예상가격:', prediction)




