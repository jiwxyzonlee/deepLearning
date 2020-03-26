# boston.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from tensorflow.keras import optimizers

import numpy
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

# 공백으로 분리된 데이터 파일을 읽어옴
df = pd.read_csv("dataset/housing.csv"
                 # 공백으로 분리된 데이터 속성
                 , delim_whitespace=True
                 , header=None)
print(df.info()) # 데이터프레임의 정보를 구해옴 : 인덱스:506행, 컬럼:14열
"""
RangeIndex: 506 entries, 0 to 505
Data columns (total 14 columns)
"""
print(df.head()) # 5개 데이터 출력

dataset = df.values

# 집값에 영향을 주는 컬럼
X = dataset[:, 0:13]
# 집값(13)
Y = dataset[:,13]

# 전체 데이터를 훈련 데이터와 테스트 데이터를 분리
# test_size=0.3 : 전체 데이터에서 테스트 데이터를 30% 사용
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y
    , test_size=0.3
    , random_state=seed
)

# 모델 생성
model = Sequential()
# 입력층 + 은닉층
model.add(Dense(30, input_dim=13, activation='relu'))

# 은닉층
#model.add(Dense(6, activation='relu'))
# 예측의 경우에는 출력층에 활성화 함수가 필요 없음
model.add(Dense(1))

# 모델학습 방식 설정
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# 정확도가 왜 계속 0에 가깝게 나올가
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=0.01))

# 모델학습
model.fit(X_train, Y_train, epochs=200, batch_size=10) # 200번 학습

# 예측 값과 실제 값의 비교
# flatten() : 데이터의 배열을 1차원으로 바꿔주는 함수
Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    # 506개의 30%(1 ~ 151)
    # 가격 정보가 들어가 있는 변수(Y_test)
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))