# multiple.py

# 다중 선형 회귀 예제
# y = ax1 + bx2 + c 형태의 데이터

# y = 3x1 – 2x2 + 1 에 정규분포 난수를 더했을 때, 최소제곱법으로 회귀계수, 절편 예측

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

## y = 3x_1 - 2x_2 + 1 학습 데이터 생성

# 0~1까지의 난수 100개 생성
x1 = np.random.rand(100, 1)

# x1값 -2~2 범위로 변경
x1 = x1 * 4 - 2

# x2도 같은 방식으로 생성
x2 = np.random.rand(100, 1)
x2 = x2 * 4 - 2

# 종속 변수 y 생성
y = 3*x1 - 2*x2 + 1

# 표준정규분포(평균 0, 표준편차 1)의 난수를 추가
y += np.random.randn(100, 1)

# x1, x2 값을 가진 행렬 생성
#[[x1_1, x2_1], [x1_2, x2_2], ..., [x1_100, x2_100]] 형태
x1_x2 = np.c_[x1, x2]
print(x1_x2[0:10])

# 학습 모델 생성
model = linear_model.LinearRegression()

# 모델 학습
model.fit(x1_x2, y)

## 회귀계수, 절편, 결정 계수
print('계수 : ', model.coef_)
print('절편: ', model.intercept_)
print('결정계수 :', model.score(x1_x2, y))

## 산점도 그래프 표시
# 회귀식 예측
y_ = model.predict(x1_x2)

# 1행 2열 배치, 첫번째 subplot
plt.subplot(1, 2, 1)
# 실제 데이터
plt.scatter(x1, y, marker='+')
# 예측 데이터
plt.scatter(x1, y_, marker='o')
plt.xlabel('x1')
plt.ylabel('y')

# 1행 2열 배치, 두번째 subplot
plt.subplot(1, 2, 2)
# 실제 데이터
plt.scatter(x2, y, marker='+')
# 예측 데이터
plt.scatter(x2, y_, marker='o')
plt.xlabel('x2')
plt.ylabel('y')

plt.tight_layout()
# 자동으로 레이아웃을 설정해주는 함수
plt.show()
