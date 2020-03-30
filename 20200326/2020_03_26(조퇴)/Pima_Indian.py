# Pima_Indian.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 피마 인디언 당뇨병 데이터셋 로딩
# 불러올 때 각 컬럼에 해당하는 이름을 지정
df = pd.read_csv(
    'dataset/pima-indians-diabetes.csv'
    , names = ["pregnant", "plasma", "pressure", "thickness"
        , "insulin", "BMI", "pedigree", "age", "class"]
)
# 처음 5개 데이터 확인
print(df.head(5))
print(df) # [768 rows x 9 columns]

# 데이터의 자료형 확인
print(df.info())

# 데이터의 통계 요약 정보 확인
print(df.describe())

# 공복혈당, 클래스 정보 출력
print(df[['plasma', 'class']])

# 그래프 설정
colormap = plt.cm.gist_heat # 그래프의 색상 설정
plt.figure(figsize=(12,12)) # 그래프의 크기 설정

# 데이터 간의 상관관계를 heatmap 그래프 출력
# vmax의 값을 0.5로 지정해 0.5에 가까울수록 밝은 색으로 표시
sns.heatmap(df.corr()
            , linewidths=0.1
            , vmax=0.5
            , cmap=colormap
            , linecolor='white'
            , annot=True)
plt.show()

# 히스토그램
grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma', bins=10) # plasma : 공복 혈당
plt.show()

# 딥러닝을 구동하는 데 필요한 케라스 함수 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정
numpy.random.seed(3)
tf.random.set_seed(3)

# 데이터를 불러오기
dataset = numpy.loadtxt(
    "dataset/pima-indians-diabetes.csv"
    , delimiter=",")

X = dataset[:,0:8] # 8개의 컬럼 정보
Y = dataset[:,8] # class : 0 or 1

# 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # 출력 노드:12개, 입력노드:8개
model.add(Dense(8, activation='relu')) # 은닉층
model.add(Dense(1, activation='sigmoid')) # 출력층 이중분류(sigmoid)

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=10) # 200번 학습

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))