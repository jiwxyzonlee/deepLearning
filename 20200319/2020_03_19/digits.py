# digits.py

# digits 데이터셋은 0부터 9까지 손으로 쓴 숫자 이미지 데이터로 구성되어 있다.
# 이미지 데이터는 8 x 8픽셀 흑백 이미지로, 1797장이 들어 있다.

from sklearn import datasets
import matplotlib.pyplot as plt

# digits dataset 로드
digits = datasets.load_digits()

#print(digits)

# 이미지 라벨
#print(digits.target)
# 이미지
#print(digits.images)

# 0부터 9까지 이미지를 2행 5열로 출력
for label, img in zip(digits.target[:10], digits.images[:10]):
    # 2행 5열로 배치 (0번 이미지를 첫번째, 1번 이미지를 두번째, ...)
    plt.subplot(2, 5, label + 1)
    # 축 수치 지우기
    plt.axis('off')
    plt.imshow(img)
    # 흑백 이미지 설정(gray scale, img)
    #plt.imshow(img, cmap=plt.cm.gray)
    # x축 라벨 재설정
    plt.xlabel('')
    # 이미지마다 타이틀 설정
    plt.title('Digit:{0}'.format(label))

plt.show()
