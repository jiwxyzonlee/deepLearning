classification_validate.py
# 분류기를 만들어 정답률 평가

import numpy as np
from sklearn import datasets

"""
# 난수 시드값 설정
# 동일한 결과를 출력하기 위하여 사용
np.random.seed(0)
"""

# 데이터셋 로딩
digits = datasets.load_digits()

# 3과 8의 데이터 위치 구하기
flag38 = (digits.target == 3) + (digits.target == 8)

# print(flag38)
# [False False False ...  True False  True]

# 3과 8 이미지와 라벨을 변수에 저장
labels = digits.target[flag38]
images = digits.images[flag38]
# print(labels.shape)
# print(images.shape)
"""
(357,)
(357, 8, 8)
"""

# 3과 8 이미지 데이터를 2차원에서 1차원으로 변환
images = images.reshape(images.shape[0], -1)
#print('reshape: ', images.shape) #(357, 64)

"""
# train data와  test data 분할
# 3과 8의 이미지 개수 357
n_samples = len(flag38[flag38])
print(n_samples) #357

# 학습 데이터 개수 214
train_size = int(n_samples * 3/5)
print(train_size) #214
# 학습데이터: images[:214], labels[:214]
"""

# train data와  test data 분할 (7:3 비율)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    #이미지
    images
    # 라벨
    , labels
    # test data 비율 30%
    , test_size=0.3
    # 난수 시드
    , random_state=10)


# 결정 트리 분류기 모델 생성
from sklearn import tree

classifier = tree.DecisionTreeClassifier()

# 모델 학습 fit(images[:214], labels[:213])
# 학습 데이터는 손으로 쓴 숫자의 전체 이미지 데이터 중 60%를 사용해서 학습
#classifier.fit(images[:train_size], labels[:train_size])
classifier.fit(x_train, y_train)

"""
# 테스트 데이터 구하기 labels[214:]
test_label = labels[train_size:]
# 실제 데이터의 라벨값
print(test_label)
"""
# 실제 데이터의 라벨값 (y_test에 이미 라벨값이 들어가 있음)
#print(y_test)

# 테스트 데이터를 이용하여 라벨 예측
#predict_label = classifier.predict(images[train_size:])
predict_label = classifier.predict(x_test)
# 예측 데이터의 라벨값
#print(predict_label)

# 분류기의 성능 평가
# 성능평가 모듈 import
from sklearn import metrics

# 정답률(Accuracy):  0.8741258741258742
#print('정답률(Accuracy): ', metrics.accuracy_score(test_label, predict_label))
#print('정답률(Accuracy): ', metrics.accuracy_score(y_test, predict_label))
# 정답률(Accuracy):  0.9537037037037037

print('혼돈행렬: ', metrics.confusion_matrix(y_test, predict_label))
print()
print('적합률(3): ', metrics.precision_score(y_test, predict_label, pos_label=3))
print('적합률(8): ', metrics.precision_score(y_test, predict_label, pos_label=8))
print()
print('재현율(3): ', metrics.recall_score(y_test, predict_label, pos_label=3))
print('재현율(8): ', metrics.recall_score(y_test, predict_label, pos_label=8))
print()
print('F값(3): ', metrics.f1_score(y_test, predict_label, pos_label=3))
print('F값(8): ', metrics.f1_score(y_test, predict_label, pos_label=8))