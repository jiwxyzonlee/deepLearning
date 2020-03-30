# 피마 인디언 당뇨병 분류

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('dataset/pima-indians-diabetes.csv',
 names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

print(df)                       # [768 rows x 9 columns]

# 자료형 확인
print(df.info())

# 통계 요약 정보 확인
print(df.describe())

# 공복 혈당, 클래스 정보 출력
print(df[['plasma','class']])

# 그래프 설정
colormap = plt.cm.gist_heat         # 그래프의 색상 설정
plt.figure(figsize=(12,12))         # 그래프의 크기 설정

# 데이터 간의 상관관계를 heatmap 그래프 출력
# vmax의 값을 0.5로 지정할 0.5에 가까울 수록 밝은 색으로 표시
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True)
plt.show()

# 히스토그램
grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma', bins=10)           # plasma : 공복 혈당
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

# 데이터를 불러오기
# dataset = np.loadtxt("dataset/pima-indians-diabetes.csv", delimiter=",")
# print(dataset)

# x = dataset[:,0:8]          # 8개의 컬럼 정보
# y = dataset[:,8]            # class : 0 or 1

dataset = pd.read_csv("dataset/pima-indians-diabetes.csv", delimiter=",")
x = dataset.iloc[:,0:8]          # 8개의 컬럼 정보
y = dataset.iloc[:,8]            # class : 0 or 1

# 모델 생성
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))    # 입력층, 은닉층
model.add(Dense(8, activation='relu'))                  # 은닉층
model.add(Dense(1, activation='sigmoid'))               # 출력층

# 모델 학습 방법 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(x, y, epochs=1000, batch_size=10)      # 학습 횟수 : 200회

print('accuracy: %.4f' %(model.evaluate(x,y)[1]))





