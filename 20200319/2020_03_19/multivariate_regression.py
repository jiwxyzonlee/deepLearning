# multivariate_regression.py

# UCI 자동차 연비 데이터셋으로 다중회귀분석

# 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 자동차 정보 파일 읽어오기
# CSV 파일을 읽어와서 데이터프레임으로 변환
df = pd.read_csv('auto-mpg.csv', header=None)
#print(df.head())

# 컬럼명 설정
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower'
    , 'weight', 'acceleration', 'model year', 'origin', 'name']
#print(df.head())
#df.info()
# 3   horsepower    398 non-null    object
# 8   name          398 non-null    object

# 통계요약 정보 확인 (object인 horespower와 name 없음)
#print(df.describe()); print()

# horsepower 고유값 확인
#print(df['horsepower'].unique()) #'193.0' '?' '100.0'

# 출력값 중 '?'를 np.NaN 치환 필요 (결측치, null)
df['horsepower'].replace('?', np.NaN, inplace=True)

# 결측데이터(np.NaN) 제거
df.dropna(subset=['horsepower'], axis=0, inplace=True)

# horsepower 열의 자료형 변경 (문자형 ->실수형)
df['horsepower'] = df['horsepower'].astype('float')

#print(df.describe())

#df.info() #  3   horsepower    392 non-null    float64

#print(df['horsepower'].unique()) # 바뀐 거 비교해 보기 (193.  nan 100.)
#print(df['horsepower'].describe())

# 분석에 사용할 컬럼 선택
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]

# 독립변수와 종속변수 선택
# 독립변수
x = ndf[['cylinders', 'horsepower', 'weight']]
# 종속변수
y = ndf[['mpg']]

# train data와 test data로 분할(7:3 비율)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y
                                                    # 분할 비율
                                                    , test_size=0.3
                                                    # 난수 시드 설정
                                                    , random_state=10)
# 데이터 확인
#print('train data : ', x_train.shape)
#print('test data: ', x_test.shape)
"""
train data :  (274, 3)
test data:  (118, 3)
"""

# 다중회귀 분석
from sklearn.linear_model import LinearRegression

# 모델 생성
model = LinearRegression()

# train data를 이용해서 학습
model.fit(x_train, y_train)

# 결정계수
r_score = model.score(x_test, y_test)

#print('결정계수 : ', r_score) # 결정계수 :  0.6939048496695599

# 회귀계수와 절편 구하기
#print('회귀계수 : ', model.coef_)
#print('절편 : ', model.intercept_)
"""
회귀계수 :  [[-0.60691288 -0.03714088 -0.00522268]]
절편 :  [46.41435127]
"""

# 모델이 예측한 값 구하기
y_hat = model.predict(x_test)
#print(y_hat[0:10])

# 실제 데이터(y_test)와 예측값(y_hat)을 그래프로 출력 (커널 밀도 그래프)
plt.figure(figsize=(10, 5))
# 실제값
ax1 = sns.distplot(y_test, hist=False, label = 'y_test')
# 예측값
ax2 = sns.distplot(y_hat, hist=False, label='y_hat', ax = ax1)
plt.show()