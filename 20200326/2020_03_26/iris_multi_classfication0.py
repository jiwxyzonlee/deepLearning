# 아이리스 품종 분류

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

# 데이터 읽어오기
df = pd.read_csv('dataset/iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"], delimiter=",")
print(df)                               # [150 rows x 5 columns]

# 그래프 출력
# sns.pairplot(df, hue='species')
# plt.show()

# 데이터 분류
# dataset = df.values
x = df.iloc[:, 0:4].astype(float)
y_obj = df.iloc[:,4]
print(y_obj)

# e = LabelEncoder()
# e.fit(y_obj)
# y = e.transform(y_obj)
y = LabelEncoder().fit(y_obj).transform(y_obj)
print(y)



# 원-핫 인코딩(one-hot-encoding)
y_encoded = tf.keras.utils.to_categorical(y)
print(y_encoded)

# 모델의 설정
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))               # 3가지로 분류

# 모델 컴파일
model.compile(loss='categorical_crossentropy',         # 오차함수 : 다중분류 - categorical_crossentropy
              optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(x, y_encoded, epochs=50, batch_size=1)        # 학습횟수(epochs) : 50회

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(x, y_encoded)[1]))


