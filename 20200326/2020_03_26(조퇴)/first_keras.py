# first_keras.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras import optimizers

# 난수 시드 설정
np.random.seed(1234) # 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분

# 학습 데이터
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 모델 정의
model = Sequential()
model.add(Dense(1, input_dim=1)) # 출력 노드 1개, 입력 node 1개

# 모델학습 방식 설정
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # 평균제곱오차
model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))

# 모델학습
model.fit(x_train, y_train, epochs=20000) # 20000번 학습
# 학습을 10000번 정도 설정하면 정확하게 예측하지 못함

# 모델을 이용해서 예측
y_predict = model.predict(np.array([1, 2, 3, 4]))
#print(y_predict)
#print()

# 모델을 이용해서 예측
y_predict = model.predict(np.array([7, 8, 9, 100]))
print(y_predict)
#print()

"""
1. 데이터 생성
 - 원본데이터를 불러오거나 시뮬레이션을 통해 데이터를 생성
 - 데이터로부터 훈련셋, 시험셋, 검증셋을 생성
 - 이때 딥러닝 모델의 학습 및 평가할 수 있도록 포맷 변환
2. 모델 설정 : Sequential()
 - 시퀀스 모델을 생성한 뒤 필요한 레이어를 추가하여 구성
 - 좀 더 복잡한 모델이 필요할 때는 케라스 함수 API를 사용
3. 모델 학습과정 설정 : compile()
 - 모델 학습을 하기 전에 모델 학습에 대한 설정을 수행
 - 손실함수 및 최적화 방법을 정의
 - 케라스에서는 compile() 함수를 사용
4. 모델 학습 : fit()
 - 구성한 모델을 훈련셋으로 학습
 - 케라스에서는 fit() 함수를 사용
5. 학습과정 살펴보기
 - 모델 학습시 훈련셋, 검증셋의 손실 및 정확도를 측정
 - 반복 횟수에 따른 손실 및 정확도 추이를 보면서 학습 상황을 판단
6. 모델 평가 : evaluate()
 - 준비된 시험셋으로 학습한 모델의 성능을 평가
 - 케라스에서는 evaluate() 함수를 사용
7. 모델 사용 : predict()
 - 임의의 입력 데이터를 모델을 사용하여 예측
 - 케라스에서는 predict() 함수를 사용
"""