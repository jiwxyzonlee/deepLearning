from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# MNIST 데이터(학습데이터, 테스트데이터) 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 학습데이터의 크기 : 60000개
print(x_train.shape)        # (60000, 28, 28)
print(y_train.shape)        # (60000,)

# 테스트 데이터 크기 : 10000개
print(x_test.shape)         # (10000, 28, 28)
print(y_test.shape)         # (10000,)

print('학습셋 이미지 수: %d 개' %(x_train.shape[0]))     # 학습셋 이미지 수: 60000 개
print('테스트셋 이미지 수: %d 개' %(x_test.shape[0]))    # 테스트셋 이미지 수: 10000 개

# 첫번째 이미지 출력 : 배열로 출력 ( 0 ~ 255 )
print(x_train[0])

# 그래픽으로 첫번째 이미지 출력
plt.imshow(x_train[0])

# plt.imshow(x_train[0], cmap='Greys')   # 흑백 이미지
plt.show()

# 첫번째 이미지 라벨 출력 : 5
print(y_train[0])

# MNIST 데이터 중 10장만 표시
for i in range(10):
 plt.subplot(2, 5, i+1)             # 2행 5열로 이미지 배치
 plt.title("M_%d" % i)
 plt.axis("off")
 plt.imshow(x_train[i], cmap=None)
 # plt.imshow(x_train[i], cmap='Greys')
plt.show()