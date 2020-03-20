# svm_classification.py

# 기본 라이브러리 호출
import pandas as pd
import seaborn as sns

# load_dataset 함수로 titanic 데이터를 읽어와서 데이터프레임으로 변환
df = sns.load_dataset('titanic')

# 데이터 살펴보기
#print(df.head())
#df.info()

# IPython 디스플레이 설정 - 출력할 열의 개수를 15개로 늘리기
pd.set_option('display.max_columns', 15)
#print(df)

# 데이터 자료형 확인 : 데이터를 확인하고 NaN이 많은 열 삭제
df.info(); print()

#  11  deck         203 non-null    category
# NaN값이 많은 deck(배의 갑판)열을 삭제
# embarked(승선)와 내용이 겹치는 embark_town(승선 도시) 열을 삭제
rdf = df.drop(['deck', 'embark_town'], axis=1)
print(rdf.columns.values)
# ['survived' 'pclass' 'sex' 'age' 'sibsp' 'parch' 'fare' 'embarked' 'class' 'who' 'adult_male' 'alive' 'alone']
print()

# 승객의 나이를 나타내는 age 열에 누락 데이터가 177개, 누락 데이터가 있는 행을 모두 삭제
# 나이 데이터가 있는 714명의 승객만을 분석
rdf = rdf.dropna(subset=['age'], how='any', axis=0)
print(len(rdf))
# 714
print()

# embarked 열의 데이터는 승객들이 타이타닉호에 탑승한 도시명의 첫 글자
# embarked 열의 NaN값을 승선도시 중에서 가장 많은 값으로 치환하기 (S)
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()
print(most_freq)
# S
print()

# embarked 열에 fillna() 함수를 사용하여 누락 데이터(NaN)를 S로 치환
rdf['embarked'].fillna(most_freq, inplace=True)

# 분석에 사용할 열(속성)을 선택
ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
ndf.info(); print()

# KNN모델을 적용하기 위해 sex열과 embarked열의 범주형 데이터를 숫자형으로 변환
# 원핫인코딩 - 범주형 데이터를 모델이 인식할 수 있도록 숫자형으로 변환
# sex 열은 male과 female을 열 이름으로 갖는 2개의 더미 변수 열이 생성
# concat()함수를 이용하여 더미 변수를 기존 데이터프레임에 연결
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1)

# embarked 3개의 더미 변수(town_X) 열 생성
# prefix='town' 옵션을 사용 (town_C, town_Q, town_S)
onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town')
ndf = pd.concat([ndf, onehot_embarked], axis=1)

# 기존 sex열과 embarked열 삭제
ndf.drop(['sex', 'embarked'], axis=1, inplace=True)
# 수정 내용 확인
ndf.info(); print()

# 독립 변수 x 새로 정의
x = ndf[['pclass', 'age', 'sibsp', 'parch'
    , 'female', 'male'
    , 'town_C', 'town_Q', 'town_S']]
# 종속 변수 y
y = ndf['survived']

# 독립 변수 데이터를 정규화(normalization)
# 데이터의 상대적 크기 차이를 없애기 위하여 정규화

from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)

# train data 와 test data 분할(7:3 비율)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y
                                                    , test_size=0.3
                                                    , random_state=10)

print('train data 개수: ', x_train.shape)
print('test data 개수: ', x_test.shape)
"""
train data 개수:  (499, 9)
test data 개수:  (215, 9)
"""
print()

# sklearn 라이브러리에서 SVM 분류 모델 가져오기
from sklearn import svm

# SVC 모델 객체 생성 (kernel='rbf' 적용)
svm_model = svm.SVC(kernel='rbf')

# train data를 이용하여 모델 학습
svm_model.fit(x_train, y_train)

# test data를 이용해서 y_hat을 예측 (분류)
# 예측값 구하기
y_hat = svm_model.predict(x_test)

# 첫 10개의 예측값(y_hat)과 실제값(y_test) 비교
# 8개 일치함( 0:사망자, 1:생존자)
print('예측값(y_hat[0:10]):\n', y_hat[0:10])
print('실제값(y_test.values[0:10]):\n', y_test.values[0:10])
"""
예측값(y_hat[0:10]):
 [0 0 1 0 0 0 1 0 0 0]
실제값(y_test.values[0:10]):
 [0 0 1 0 0 1 1 1 0 0]
"""
print()

# SVM모델 성능 평가 - Confusion Matrix(혼동 행렬) 계산
from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_hat)
print('혼동 행렬:\n', svm_matrix)
"""
혼동 행렬:
 [[120   5]
 [ 35  55]]
"""
#     P     N
# P [[120   5]
# N [ 35  55]]
# TP(True Positive) : 215명의 승객 중에서 사망자를 정확히 분류한 것이 120명
# FP(False Positive) : 생존자를 사망자로 잘못 분류한 것이 35명
# FN(False Negative) : 사망자를 생존자로 잘못 분류한 것이 5명
# TN(True Negative) : 생존자를 정확하게 분류한 것이 55명

print()

# SVM모델 성능 평가 - 평가지표 계산
svm_report = metrics.classification_report(y_test, y_hat)
print('svm 모델 성능평가(svm_report):\n', svm_report)
"""
svm 모델 성능평가(svm_report):
               precision    recall  f1-score   support

           0       0.77      0.96      0.86       125
           1       0.92      0.61      0.73        90

    accuracy                           0.81       215
   macro avg       0.85      0.79      0.80       215
weighted avg       0.83      0.81      0.81       215
"""
# f1지표(f1-score)는 모델의 예측력을 종합적으로 평가하는 지표
# f1-score 지표를 보면 사망자(0) 예측의 정확도가 0.86이고, 
# 생존자(1) 예측의 정확도는 0.73으로 예측 능력에 차이가 존재
# 전반적으로 KNN모델의 예측 능력과 큰 차이가 없음
