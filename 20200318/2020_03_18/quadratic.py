# quadratic.py

# 다항회귀분석, 2차 방정식
# y = aX^2 + b
# 결정 계수
# y = 3x^2 – 2에 정규분포 난수를 더했을 때, 최소제곱법으로 기울기, 절편 예측

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

## 학습 데이터

# 0~1까지 난수 100개 생성
x = np.random.rand(100, 1)

# -2~2 범위로 x값 변경
x = x * 4 - 2

# 종속 변수 설정, y = 3x^2 - 2
y = 3 * x**2 - 2

# 표준정규분포(평균 0, 표준 편차 1)의 난수를 추가함
y += np.random.randn(100, 1)

# 학습 모델
model = linear_model.LinearRegression()

## 학습
model.fit(x**2, y)

## 회귀계수, 절편, 결정 계수
print('계수 : ', model.coef_)
print('절편: ', model.intercept_)
print('결정계수 :', model.score(x**2, y))

## 산점도 그래프 출력
# 실제 데이터
plt.scatter(x, y, marker='+')
# 예측 데이터
# predict에도 x를 제곱해 전달
plt.scatter(x, model.predict(x**2), marker='o')

plt.show()