# knn_classification.py
# seaborn 라이브러리에서 제공되는 titanic 데이터셋의 탑승객의 생존여부를 KNN 알고리즘으로 분류

import pandas as pd
import seaborn as sns

# seaborn 모듈의 titanic 데이터 로드
titanic = sns.load_dataset('titanic')
#titanic.info()

# 한줄에 15개의 컬럼 출력되도록 설정
pd.set_option('display.max_columns', 15)
#print(titanic)

#titanic.info()
#  3   age          714 non-null    float64 << 개수 차이 유의
#  7   embarked     889 non-null    object  << 중복값 유의
#  11  deck         203 non-null    category << null 값 많은 열 제거(유효 데이터 적어서 제거)
#  12  embark_town  889 non-null    object  << embarked 중복값 제거
# dtypes: bool(2), category(2), float64(2), int64(4), object(5)

# deck(배의 갑판), embark_town(승선 도시) 컬럼 삭제
rdf = titanic.drop(['deck', 'embark_town'], axis=1)
# 삭제 확인
#print(rdf.columns.values)
# ['survived' 'pclass' 'sex' 'age' 'sibsp' 'parch' 'fare' 'embarked' 'class' 'who' 'adult_male' 'alive' 'alone']

# age 열에 누락 데이터 행 삭제 (891개 중 177개의 NaN 값)
rdf = rdf.dropna(subset=['age'], how='any', axis=0)
# 삭제 확인
#print(len(rdf)) #891 - 177 = 714

## embarked 열의 NaN값을 승선도시 중에서 가장 많이 출현한 값으로 치환
# embarked 열의 알파벳은 타이타닉호에 탑승한 승객의 도시
#print(rdf.describe(include='all')); print()

# 최빈값 알아보기
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()
#print(most_freq) # S : Southampton

# embarked 열에 누락 데이터(NaN)를 S로 치환 - fillna() 사용
rdf['embarked'].fillna(most_freq, inplace=True)

# 분석에 사용할 열(속성) 선택
# titanic.info()
ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
# 결과 확인
#ndf.info()
#print(ndf.head())

# KNN모델을 적용하기 위해 sex열과 embarked열의 범주형 데이터를 숫자형으로 변환
# 원핫인코딩 - 범주형 데이터를 모델이 인식할 수 있도록 숫자형으로 변환하는것
# 2   sex       714 non-null    object
# 6   embarked  714 non-null    object
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1)

# embarked열에 생성되는 더미 변수가  town_C, town_Q, town_S로 생성되도록 prefix='town' 옵션 추가
onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town')
ndf = pd.concat([ndf, onehot_embarked], axis=1)

# 열 생성 확인
#ndf.info()

# 기존 sex열과 embarked열 삭제
ndf.drop(['sex', 'embarked'], axis=1, inplace=True)
# 열 삭제 확인
#print(ndf.head())
#ndf.info()

# 변수 정의
# 독립변수 x
x = ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 'town_C', 'town_Q', 'town_S']]
# 종속변수 y
y = ndf['survived']

# 독립 변수 데이터를 정규화(normalization)
# 데이터의 상대적 크기 차이를 없애기 위하여 정규화
from sklearn import preprocessing

x = preprocessing.StandardScaler().fit(x).transform(x)

# train data 와 test data로 분할(7:3 비율)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    # 독립변수, 종속변수
    x, y
    # test data 비율 설정
    , test_size=0.3
    # 난수 seed 설정
    , random_state=10
)

print('train data(x_train) 개수: ', x_train.shape)
print('test data(x_test) 개수: ', x_test.shape)
"""
train data(x_train) 개수:  (499, 9)
test data(x_test) 개수:  (215, 9)
"""
print()

# sklearn 라이브러리에서 KNN 분류 모델 가져오기
from sklearn.neighbors import KNeighborsClassifier

# KNN 모델 객체 생성 (k=5로 설정)
knn = KNeighborsClassifier(n_neighbors=5)

# train data를 가지고 모델 학습
knn.fit(x_train, y_train)

# test data를 가지고 y_hat을 예측 (분류)
# 예측값 구하기
y_hat = knn.predict(x_test)

# 첫 10개의 예측값(y_hat)과 실제값(y_test) 비교 : 10개 모두 일치함 (0:사망자, 1:생존자)
print('예측값(y_hat): ', y_hat[0:10])
print('실제값(y_test): ', y_test.values[0:10])
"""
예측값(y_hat:  [0 0 1 0 0 1 1 1 0 0]
실제값(y_test):  [0 0 1 0 0 1 1 1 0 0]
"""
print()

# KNN 모델 성능 평가
from sklearn import metrics

# 혼돈 행렬
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print('혼돈 행렬: \n', knn_matrix)
"""
[[109  16]
 [ 25  65]]
"""
print()

# 모델 성능평가 지표 출력 (KNN 모델 성능 평가, 평가지표 계산)
knn_report = metrics.classification_report(y_test, y_hat)
print('평가지표 계산: \n',knn_report)
"""
평가지표 계산: 
               precision    recall  f1-score   support

           0       0.81      0.87      0.84       125
           1       0.80      0.72      0.76        90

    accuracy                           0.81       215
   macro avg       0.81      0.80      0.80       215
weighted avg       0.81      0.81      0.81       215
"""
print()