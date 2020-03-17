# linear_orderly.py
# y = 3x – 2 인 경우에 최소 제곱법으로 기울기와 절편 구하기

# 라이브러리
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# 학습 데이터 생성
# 0~1까지 난수를 100개 만듦
x = np.random.rand(100, 1)
# 값의 범위를 -2~2로 변경
x = 4 * x - 2
# x는 0~1까지의 난수

# y = 3x - 2
y = 3 * x - 2

# 모델 생성
# 최소 제곱법으로 구현
model = linear_model.LinearRegression()

# 모델 학습
model.fit(x, y)

# 계수(기울기), 절편
print('회귀계수(기울기) : ', model.coef_)
print('절편 : ', model.intercept_)

# 산점도 그래프 출력
plt.scatter(x, y, marker='+')
# 회귀함수: y = 3x - 2

# 오차가 없는 y = 3x – 2 예측결과 그래프
plt.show()