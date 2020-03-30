from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 읽어오기
df = pd.read_csv('dataset/iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
print(df)

# 그래프 출력
# sns.pairplot(df, hue='species')
# plt.show()

# 데이터 분류
# dataset = df.values                   # 데이터프레임의 데이터만 불러와서 dataset을 만듬
x = df.iloc[:,0:4].astype(float)        # dataset의 데이터를 float형으로 변환함
y_obj = df.iloc[:,4]                    #  y_obj : species 클래스값 저장함
print(y_obj)                            # y_obj = ['Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'

# 문자열을 숫자로 변환
# array(['Iris-setosa','Iris-versicolor','Iris-virginica'])가 array([0,1,2])으로 바뀜
# e = LabelEncoder()                      # 문자열을 숫자로 변환해주는 함수
# e.fit(y_obj)
# y = e.transform(y_obj)
# print(y)                                # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
                                        #  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...
                                        #  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ]
y = LabelEncoder().fit(y_obj).transform(y_obj)
print(y)

# 원-핫 인코딩(one-hot-encoding)
y_encoded = tf.keras.utils.to_categorical(y)
print(y_encoded)                        # [[1. 0. 0.] [1. 0. 0.] [1. 0. 0.] ...
                                        # [[0. 1. 0.] [0. 1. 0.] [0. 1. 0.] ...
                                        # [[0. 0. 1.] [0. 0. 1.] [0. 0. 1.] ]
# 모델의 설정
model = Sequential()
model.add(Dense(16,  input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(x, y_encoded, epochs=50, batch_size=1)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(x, y_encoded)[1]))