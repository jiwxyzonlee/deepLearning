# boston.py

# 모듈 import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

# boston 데이터셋 로드
boston = load_boston()

# boston 데이터셋을 이용하여 DataFrame 변환, 생성
bostonDF = pd.DataFrame(boston.data
                        , columns=boston.feature_names)

#print(bostonDF.head())
#[506 rows x 13 columns]

#bostonDF.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 13 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   CRIM     506 non-null    float64
 1   ZN       506 non-null    float64
 2   INDUS    506 non-null    float64
 3   CHAS     506 non-null    float64
 4   NOX      506 non-null    float64
 5   RM       506 non-null    float64
 6   AGE      506 non-null    float64
 7   DIS      506 non-null    float64
 8   RAD      506 non-null    float64
 9   TAX      506 non-null    float64
 10  PTRATIO  506 non-null    float64
 11  B        506 non-null    float64
 12  LSTAT    506 non-null    float64
dtypes: float64(13)
memory usage: 51.5 KB
"""

# boston.target으로 집가격을 불러와서 'PRICE' 컬럼명 부여
bostonDF['PRICE'] = boston.target

# 보스턴 데이터 형태 : (506, 14)
print('boston 데이터 형태 : ', bostonDF.shape)

bostonDF.info()
"""
 13  PRICE    506 non-null    float64  <<-new!
dtypes: float64(14)
memory usage: 55.5 KB
"""

## RM, ZN, INDUS, NOX, AGE, PTRATIO, LSTAT, RAD 의 총 8개의 컬럼
## 값이 증가할수록 PRICE에 어떤 영향을 미치는지 분석하고 시각화

# 2행 4열 그래프
# 2개의 행과 4개의 열의 subplots 로 시각화, axs는 4x2개의 ax를 가짐
fig, axs = plt.subplots(figsize=(16,8)
                        , ncols=4
                        , nrows=2)

lm_features = ['RM', 'ZN', 'INDUS', 'NOX'
    , 'AGE', 'PTRATIO', 'LSTAT', 'RAD']

for i, feature in enumerate(lm_features):
    row = int(i / 4)
    col = i % 4
    # 시본(searborn)의 regplot을 이용해 산점도와 선형 회귀선을 출력
    sns.regplot(x= feature
                , y='PRICE'
                , data=bostonDF
                , ax=axs[row][col])
plt.show()

# RM(방개수)와 LSTAT(하위 계층의 비율)이 PRICE에 영향도가 가장 두드러지게 나타남
# 1. RM(방개수)은 양 방향의 선형성(Positive Linearity)이 가장 큼
# 방의 개수가 많을수록 가격이 증가하는 모습을 확연히 보여줌
# 2. LSTAT(하위 계층의 비율)는 음 방향의 선형성(Negative Linearity)이 가장 큼
# 하위 계층의 비율이 낮을수록 PRICE 가 증가하는 모습을 확연히 보여줌

## LinearRegression 클래스를 이용해서 보스턴 주택 가격의 회귀 모델
# train_test_split()을 이용해 학습과 테스트 데이터셋을 분리해서 학습과 예측을 수행
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 'PRICE' 컬럼값을 y_target에 할당
y_target = bostonDF['PRICE']

# 데이터프레임에서 'PRICE' 컬럼을 삭제한 나머지 데이터를 리턴함
x_data = bostonDF.drop(['PRICE']
                       , axis=1
                       , inplace=False)

# 데이터프레임에서 'PRICE' 컬럼 삭제한 나머지 데이터를 반환
# train data 와 test data로 분할 (7:3 비율로 분할)
x_train, x_test, y_train, y_test = train_test_split(
    # 13개의 컬럼 데이터
    x_data
    # PRICE 컬럼 데이터
    , y_target
    # 테스트 데이터 30%
    , test_size=0.3
    # 난수 시드
    , random_state=156
)
# 난수 시드 값이 없을 경우 결정계수값이 계속 바뀌게 되므로 임의로 설정함

#print('x_train :', x_train); print()
#print('y_train :', y_train)

## 선형회귀 모델생성/ 모델학습 / 모델예측 / 모델평가 수행

# 모델생성
model = LinearRegression()

# 모델학습
model.fit(x_train, y_train)

# 모델예측 (13개의 컬럼 정보)
# 예측값
y_preds = model.predict(x_test)
#print('y_preds: ', y_preds)

# 모델평가 - r2_score(실제 데이터, 예측 데이터)
# 실측값은 y_test 안에 들어가 있음, 예측값과 비교 필요
print('결정계수 :', r2_score(y_test, y_preds)) #결정계수 : 0.7572263323138921
print()
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))
print()

# boston 데이터셋을 boston.csv 파일로 저장
bostonDF.to_csv('boston.csv', encoding='utf-8')

# LinearRegression 으로 생성한 주택가격 모델의 회귀계수(coefficients)와 절편(intercept) 구하기
# 회귀계수는 LinearRegression 객체의 coef_ 속성으로 구함
# 절편은 LinearRegression 객체의 intercept_ 속성으로 구함
#print('회귀계수값:', np.round(model.coef_, 1)) # 소수 첫째자리
print('회귀계수값:', model.coef_, 1)
print()
print('절편값:', model.intercept_)
print()

# 회귀계수를 큰 값 순으로 정렬하기 위해서 Series로 생성함
coff = pd.Series(data=np.round(model.coef_, 1) # 소수 첫째자리
                 , index=x_data.columns)
# 회귀계수를 기준으로 내림차순 정렬
print(coff.sort_values(ascending=False))