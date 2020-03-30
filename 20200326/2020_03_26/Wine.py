# 와인 종류 분류

from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(0)

# 데이터 로딩
df = pd.read_csv('dataset/wine.csv', header=None)
print(df)                   # [6497 rows x 13 columns]

dataset = df.values
x = dataset[:, 0:12]        # 와인의 특징(0 ~ 11)
y = dataset[:, 12]          # 와인종류 (1:레드와인, 0:화이트와인)

# 모델 생성
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))   # 입력층 + 은닉층
model.add(Dense(12, activation='relu'))                 # 은닉층
model.add(Dense(8, activation='relu'))                  # 은닉층
model.add(Dense(1, activation='sigmoid'))               # 출력층

# 모델 학습 방법 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# 모델 학습
model.fit(x, y, epochs=200, batch_size=200)     # 학습 횟수: 200회

print('accurary: %.4f' %(model.evaluate(x, y)[1]))
