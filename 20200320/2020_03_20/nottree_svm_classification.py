# nottree_svm_classification

# tree_classification.py

import pandas as pd
import numpy as np

# UCI 저장소에서 암세포 진단(Breast Cancer) 데이터셋 가져오기
uci_path = 'https://archive.ics.uci.edu/ml/' \
           'machine-learning-databases/' \
           'breast-cancer-wisconsin/' \
           'breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path, header=None)
print(df)
# [699 rows x 11 columns]
print()

# 11개의 열 이름 지정
df.columns = ['id','clump','cell_size','cell_shape', 'adhesion','epithlial'
    , 'bare_nuclei','chromatin','normal_nucleoli', 'mitoses', 'class']

# IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)
print(df); print()

# 데이터 자료형 확인
df.info()
# 6   bare_nuclei      699 non-null    object
print()

# 데이터 통계 요약정보 확인 : bare_nuclei 열은 출력 안 됨 (10개의 열만 출력)
#print(df.describe())

# bare_nuclei 열의 고유값 확인(unique())
# bare_nuclei 열은 ? 데이터가 포함되어 있음
print(df['bare_nuclei'].unique())
# ['1' '10' '2' '4' '3' '9' '7' '?' '5' '8' '6']
print()

# bare_nuclei 열의 '?' 를 누락데이터(np.NaN)으로 변경
df['bare_nuclei'].replace('?', np.NaN, inplace=True)

# 누락데이터 행 삭제
df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)

# bare_nuclei 열의 자료형 변경 (문자열 -> 숫자)
df['bare_nuclei'] = df['bare_nuclei'].astype('int')

# 데이터 통계 요약정보 확인
#print(df.describe())

# 수정된 자료형 확인
df.info()
# 6   bare_nuclei      683 non-null    int32
print()

# 분석에 사용할 속성(변수) 선택
# 독립변수 x (설명변수)
x = df[['clump','cell_size','cell_shape', 'adhesion','epithlial'
    , 'bare_nuclei','chromatin','normal_nucleoli', 'mitoses']]

# 종속변수 y (예측변수)
y = df['class']
# class (2: benign(양성), 4: malignant(악성))
print('y: \n', y)
print()

# 설명 변수 데이터를 정규화
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)

# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y
                                                    , test_size=0.3
                                                    , random_state=10)

print('train data 개수: ', x_train.shape)
print('test data 개수: ', x_test.shape)
"""
train data 개수:  (478, 9)
test data 개수:  (205, 9)
"""
print()

# sklearn 라이브러리에서 모델 가져오기
from sklearn import tree
from sklearn import svm


# 결정 트리 모델 생성
#tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
# Decision Tree 모델 객체 생성 (criterion='entropy' 적용)
# 각 분기점에서 최적의 속성을 찾기 위해 분류 정도를 평가하는 기준으로 entropy 값을 사용
# 트리 레벨로 5로 지정(5단계 까지 가지를 확장 가능 의미)
# 레벨이 많아질수록 모델 학습에 사용하는 훈련 데이터에 대한 예측은 정확해짐

# SVC 모델 객체 생성 (kernel='rbf' 적용)
svm_model = svm.SVC(kernel='rbf')

# train data를 가지고 모델 학습
svm_model.fit(x_train, y_train)

# test data를 가지고 y_hat을 예측 (분류)
y_hat = svm_model.predict(x_test)
print('y_hat:\n', y_hat)
print('y_test.values:\n', y_test.values)
# 2: benign(양성), 4: malignant(악성)
print()

# 첫 10개의 예측값(y_hat)과 실제값(y_test) 비교 : 10개 모두 일치함
print('y_hat:\n', y_hat[0:10])
print('y_test.values:\n', y_test.values[0:10])
"""
y_hat:
 [4 4 4 4 4 4 2 2 4 4]
y_test.values:
 [4 4 4 4 4 4 2 2 4 4]
"""
print()


if y_hat.all() == y_test.all():
    print('y_hat and y_test are the same')
else:
    print('y_hat and y_test are not the same')
#y_hat and y_test are the same


# 모델 성능 평가 - Confusion Matrix(혼동 행렬) 계산
from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_hat)
print('혼동행렬: \n', svm_matrix)
"""
혼동행렬: 
 [[127   4]
 [  1  73]]
"""
# 양성 종양의 목표값은 2, 악성 종양은 4
# TP(True Positive) : 양성 종양을 정확하게 분류한 것이 127개
# FP(False Positive) : 악성 종양을 양성 종양으로 잘못 분류한 것이 4개
# FN(False Negative) : 양성 종양을 악성 종양으로 잘못 분류한 것이 1개
# TN(True Negative) : 악성 종양을 정확하게 분류한 것이 73개
print()

# Decision Tree 모델 성능 평가 - 평가지표 계산
svm_report = metrics.classification_report(y_test, y_hat)
print('svm_report(평가지표 계산):\n', svm_report)
"""
svm_report(평가지표 계산):
               precision    recall  f1-score   support

           2       0.99      0.97      0.98       131
           4       0.95      0.99      0.97        74

    accuracy                           0.98       205
   macro avg       0.97      0.98      0.97       205
weighted avg       0.98      0.98      0.98       205
"""