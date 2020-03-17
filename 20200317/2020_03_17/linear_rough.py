# linear_rough.py
# y = 3x – 2에 정규분포 난수를 더했을때, 최소제곱법으로 기울기, 절편 예측

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# 학습 데이터 생성
# 0~1까지 난수를 100개 생성
x = np.random.rand(100, 1)
# 값의 범위를 -2~2로 변경
x = x * 4 - 2
# y = 3x – 2
y = 3 * x - 2

# 표준 정규 분포(평균 0, 표준 편차 1)의 난수를 추가함
y += np.random.randn(100, 1)

# 모델 생성
model = linear_model.LinearRegression()

# 모델 학습
model.fit(x, y)

# 예측값 출력
print(model.predict(x)[0:10])

# 회귀계수, 절편
print('계수(기울기) : ', model.coef_)
print('절편 : ', model.intercept_)

"""
계수(기울기) :  [[2.94827263]]
절편 :  [-2.01615838]
"""

# 산점도 그래프 출력
# 실제값 그래프 : + 마커
plt.scatter(x, y, marker ='+')

# 예측값 그래프 : o 마커
plt.scatter(x, model.predict(x), marker='o')
plt.show()