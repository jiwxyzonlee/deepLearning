# AdaBoost.py

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# 손으로 쓴 숫자 데이터 읽기
digits = datasets.load_digits()

# 이미지를 2행 5열로 표시
for label, img in zip(digits.target[:10], digits.images[:10]):
    plt.subplot(2, 5, label + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Digit: {0}'.format(label))
plt.show()

# 3과 8의 데이터 위치를 구하기
flag_3_8 = (digits.target == 3) + (digits.target == 8)

# 3과 8의 데이터를 구하기
images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

# 3과 8의 이미지 데이터를 1차원으로 변환
images = images.reshape(images.shape[0], -1)

# 분류기 생성
n_samples = len(flag_3_8[flag_3_8])
train_size = int(n_samples * 3 / 5)

# base_estimator는 약한 학습기를 지정하는 파라미터로 여기서는 결정트리를 지정,
# n_estimators는 약한 학습기 개수를 지정
classifier = ensemble.AdaBoostClassifier(base_estimator=
                                         tree.DecisionTreeClassifier(max_depth=3)
                                         , n_estimators=20)
classifier.fit(images[:train_size], labels[:train_size])

# 분류기 성능을 확인
expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])
print('Accuracy:', accuracy_score(expected, predicted))
print('Confusion matrix:', confusion_matrix(expected, predicted))
print('Precision:', precision_score(expected, predicted, pos_label=3))
print('Recall:', recall_score(expected, predicted, pos_label=3))
print('F-measure:', f1_score(expected, predicted, pos_label=3))