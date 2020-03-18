# polynomial_regression.py

# 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일을 읽어와서 데이터프레임으로 변환
df = pd.read_csv('auto-mpg.csv', header=None)

# 컬럼명 설정
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# # IPython 디스플레이 설정 - 출력할 열의 개수 늘리기
pd.set_option('display.max_columns', 10)
print(df)

# 데이터 자료형 확인
df.info()

# 통계요약 정보 확인 (object인 horespower와 name 없음)
#print(df.describe()); print()

# horsepower 고유값 확인
print(df['horsepower'].unique()) #'193.0' '?' '100.0'

# 출력값 중 '?'를 np.NaN 치환 필요 (결측치, null)
df['horsepower'].replace('?', np.NaN, inplace = True)

# horsepower 컬럼을 문자형에서 실수형으로 변환
df['horsepower'] = df['horsepower'].astype('float')
#print(df.describe())
df.info()
#  3   horsepower    392 non-null    float64
print(df['horsepower'].unique()) # 바뀐 거 비교해 보기 (193.  nan 100.)

# 분석에 활용할 열(속성) 선택
# 연비(mpg), 실린더(cylinders), 출력(horesepower), 중량(weight)
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]
print(ndf)

## 변수 선택
# 독립변수 (중량)
x = ndf[['weight']]

# 종속 변수 (연비)
y = ndf[['mpg']]

# train data와 test data를 분할 (7:3 비율)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    # 독립변수
    x
    # 종속변수
    , y
    # test data 30% 할당
    , test_size=0.3
    # 난수 시드
    , random_state=10)
print('훈련 데이터 : ', x_train.shape)
print('검증 데이터 : ', x_test.shape)
"""
train data 개수 :  (278, 1)
test data 개수 :  (120, 1)
"""

# sklearn 라이브러리에서 필요한 모듈 가져오기
# 선형회귀분석
from sklearn.linear_model import LinearRegression
# 다항식변환
from sklearn.preprocessing import PolynomialFeatures

# 다항식 변환
# 2차항 적용(제곱 형태로 설정)
poly = PolynomialFeatures(degree=2)
# x_train 데이터를 2차항으로 변환
x_train_poly = poly.fit_transform(x_train)
# x_train의 1개 열이 x_train_poly에서는 3개의 열로 늘어넘
print(x_train_poly)

print('원래 데이터 : ', x_train.shape)
print('2차항 변환 데이터: ', x_train_poly.shape)
"""
원래 데이터 :  (278, 1)
2차항 변환 데이터:  (278, 3)
"""

# 모델 생성
model = LinearRegression()

# train data를 이용해서 모델 학습
model.fit(x_train_poly, y_train)

# 학습을 마친 모델에 test data를 적용하여 결정 계수 계산
# x_test 데이터를 2차항(제곱) 형태로 변환
x_test_poly = poly.fit_transform(x_test)
# 결정 계수 구하기
r_square = model.score(x_test_poly, y_test)
print('결정계수 :', r_square)
# 결정계수 : 0.7255470154177006

## train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력
# 예측데이터 x_test_poly와 실제데이터 y_test 도출하고 그래프로 출력하여 비교하기
# test data로 예측하기
y_hat_predict = model.predict(x_test_poly)

# 실제 데이터와 예측데이터를 그래프로 출력
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_train, y_train
        , 'o'
        , label='Train Data')
ax.plot(x_test, y_hat_predict
        # 빨간 + 마크
        , 'r+'
        , label = 'Predicted Data')
ax.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
plt.close()
# 곡선모양의 회귀선

# 실제값인 y값과 예측값인 y_hat의 분포 차이 비교
# x 데이터를 2차항으로 변환 (x는 weight, 독립변수)
x_poly = poly.fit_transform(x)
# 예측값
y_hat = model.predict(x_poly)

# 그래프 출력 distplot() (히스토그램과 kde 그래프)
plt.figure(figsize=(10, 5))
# 실제값
ax1 = sns.distplot(y, hist=False, label='y')
# 예측값
ax2 = sns.distplot(y_hat, hist=False, label="y_hat", ax=ax1)
plt.show()
plt.close()


