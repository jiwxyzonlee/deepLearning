# simple_linear_regression.py
# 단순 회귀 분석
# UCI(University of California, Irvine) 자동차 연비 데이터셋으로 단순회귀분석

# 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# csv 파일을 읽어와서 데이터프레임으로 변환
df = pd.read_csv('auto-mpg.csv', header=None)

# 데이터 살펴보기
#print(df)
#df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 398 entries, 0 to 397
Data columns (total 9 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       398 non-null    float64
 1   1       398 non-null    int64  
 2   2       398 non-null    float64
 3   3       398 non-null    object 
 4   4       398 non-null    float64
 5   5       398 non-null    float64
 6   6       398 non-null    int64  
 7   7       398 non-null    int64  
 8   8       398 non-null    object 
dtypes: float64(4), int64(3), object(2)
memory usage: 28.1+ KB
"""

# 컬럼명 설정
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# # IPython 디스플레이 설정 - 출력할 열의 개수 늘리기
pd.set_option('display.max_columns', 10)
#print(df)

# 데이터 자료형 확인
#df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 398 entries, 0 to 397
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   mpg           398 non-null    float64
 1   cylinders     398 non-null    int64  
 2   displacement  398 non-null    float64
 3   horsepower    398 non-null    object   <<- 변경 필요
 4   weight        398 non-null    float64
 5   acceleration  398 non-null    float64
 6   model year    398 non-null    int64  
 7   origin        398 non-null    int64  
 8   name          398 non-null    object 
dtypes: float64(4), int64(3), object(2)
memory usage: 28.1+ KB

Process finished with exit code 0

"""

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

## 독립변수(x, weight), 종속변수(y, mpg) 간의 선형관계를 산점도 그래프로 확인
# 1. pandas로 산점도 그리기
ndf.plot(kind='scatter'
         , x='weight'
         , y='mpg'
         , c='coral'
         , figsize=(10, 5))
plt.show()


# 2. seaborn 산점도 그리기
fig = plt.figure(figsize=(10, 5))
# 1행 2열 첫번째
ax1 = fig.add_subplot(1, 2, 1)
# 1행 2열 두번째
ax2 = fig.add_subplot(1, 2, 2)

sns.regplot(x='weight', y='mpg'
            , data=ndf
            , ax=ax1
            # 회귀선 표시 여부
            , fit_reg=True)
sns.regplot(x='weight', y='mpg'
            , data=ndf
            , ax=ax2
            # 회귀선 표시 여부
            , fit_reg=False)
plt.show()
plt.close()

# 3. seaborn의 조인트 그래프 - 산점도와 히스토그램 동시에
sns.jointplot(x='weight', y='mpg'
              , data=ndf)

sns.jointplot(x='weight', y='mpg'
              , data=ndf
              # 회귀선 표시
              , kind='reg')
plt.show()
plt.close()

# 4. seaborn의 pairplot으로 두 변수 간의 모든 경우의 수 그리기
sns.pairplot(ndf)
plt.show()
plt.close()

# 변수 선택
# 독립 변수 (중량)
x = ndf[['weight']]
# 종속 변수 (연비)
y = ndf[['mpg']]

# train data와 test data 분할(7:3 비율)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    # 독립변수
    x
    # 종속변수
    , y
    # test data 30%
    , test_size=0.3
    # 난수 시드
    , random_state=10)
print('train data 개수 : ', len(x_train))
print('test data 개수 : ', len(x_test))
"""
[398 rows x 4 columns]
train data 개수 :  278
test data 개수 :  120
"""

# 모듈 불러오기
from sklearn.linear_model import LinearRegression

# 모델 생성
model = LinearRegression()

# train data를 이용해서 학습
model.fit(x_train, y_train)

# 결정 계수
r_square = model.score(x_test, y_test)
print('결정계수 :', r_square)
#결정계수 : 0.6893638093152089

# 회귀계수, 절편
print('회귀계수 : ', model.coef_)
print('절편 : ', model.intercept_)

## 예측값과 실제값을 그래프로 출력
# 예측값
y_predict = model.predict(x)

plt.figure(figsize=(10, 5))
# 실제값
ax1 = sns.distplot(y, hist=False, label='y')
# 예측값
ax2 = sns.distplot(y_predict, hist=False, label='y_predict')
plt.show()