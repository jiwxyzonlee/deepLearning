# r_squared.py
# 결정 계수
# y = 3x – 2에 정규분포 난수를 더했을 때, 최소제곱법으로 기울기, 절편 예측
# 오차가 있는 y = 3x – 2 예측결과


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

## 학습 데이터 생성

# 0~1 난수 100개 생성
x = np.random.rand(100, 1)

# -2~2 사이 x 값 변경
x = x * 4 - 2

# y = 3x-2
y = 3 * x - 2

# 표준 정규분포(평균:0, 표준편차:1)의 난수 추가
y += np.random.randn(100, 1)

# 모델 생성
model = linear_model.LinearRegression()

# 모델 학습
model.fit(x, y)

## 회귀계수, 결정계수
print('회귀계수 : ', model.coef_)
print('절편 : ', model.intercept_)

# 결정계수
r2 = model.score(x, y)
print('결정계수 :', r2)

## 산점도 그래프
# 실제 데이터
plt.scatter(x, y, marker='+')
# 예측 데이터
plt.scatter(x, model.predict(x), marker='o')

plt.show()