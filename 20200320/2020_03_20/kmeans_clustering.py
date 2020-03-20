# kmeans_clustering.py

import pandas as pd
import matplotlib.pyplot as plt

# UCI 저장소에서 도매업 고객(wholesale customers) 데이터셋 가져오기
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' \
           '00292/Wholesale%20customers%20data.csv'
# 경로 주소를 브라우저에 복붙 시 파일 저장 가능
df = pd.read_csv(uci_path, header=0)
#print(df)
# [440 rows x 8 columns]

# 데이터 살펴보기
print(df.head())
print()

# 데이터 자료형 확인
df.info()
print()

# 데이터 통계 요약정보 확인
print(df.describe())
print()

# 데이터 분석에 사용할 속성(열, 변수)을 선택
# k-means는 비지도 학습모델이기 때문에 예측(종속)변수를 지정할 필요가 없고 모두 설명(독립)변수만 사용
# 데이터프레임의 모든 데이터 (8개의 설명변수)
x = df.iloc[:, :]
print(x); print()
print(x[:5]); print()

# 변수 데이터 정규화
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
print(x[:5]); print()

# k-means 모듈 import
# sklearn 라이브러리에서 cluster 군집 모델 가져오기
from sklearn import cluster

# k-means 모델 객체 생성
# k-means 모델은 8개의 속성(변수)을 이용하여 각 관측값을 5개의 클러스터로 구분
# 클러스터의 개수를 5개로 설정, n_clusters=5
kmeans = cluster.KMeans(n_clusters=5)

# k-means 모델 학습
# k-means 모델로 학습 데이터 x를 학습 시키면, 클러스터 갯수(5) 만큼 데이터를 구분
# 모델의 labels_ 속성(변수)에 구분된 클러스터 값(0~4)이 입력
kmeans.fit(x)

# 예측 (군집) 결과를 출력할 열(속성)의 값 구하기
# 변수 labels_ 에 저장된 값을 출력해보면, 0~4 범위의 5개 클러스터 값이 출력됨
# 각 데이터가 어떤 클러스터에 할당 되었는지를 확인 가능
# (매번 실행 할 때마다 예측값의 결과가 다름)
cluster_label = kmeans.labels_
print(cluster_label); print()

# 예측(군집) 결과를 저장할 열(Cluster)을 데이터프레임에 추가
df['Cluster'] = cluster_label
print(df.head())

# 그래프로 시각화 - 클러스터 값 : 0 ~ 4 모두 출력
# 8개의 변수를 하나의 그래프로 표현할 수 없기 때문에 2개의 변수를 선택하여 관측값의 분포 그리기
# 모델의 예측값은 매번 실행할 때마다 달라지므로, 그래프의 형태도 달라짐
# 산점도 : x='Grocery', y='Frozen' 식료품점, 냉동식품
# 산점도 : x='Milk', y='Delicassen' 우유, 조제식품점
df.plot(kind='scatter'
        , x='Grocery', y='Frozen'
        , c='Cluster'
        , cmap='Set1'
        # colorbar 미적용
        , colorbar=False
        , figsize=(10, 10))
df.plot(kind='scatter'
        , x='Milk', y='Delicassen'
        , c='Cluster'
        , cmap='Set1'
        # colorbar 적용
        , colorbar=True
        , figsize=(10, 10))
plt.show()
plt.close()

# 그래프로 시각화 - 클러스터 값 : 1, 2, 3 확대해서 자세하게 출력
# 다른 값들에 비해 지나치게 큰 값으로 구성된 클러스터(0, 4)를 제외
# 데이터들이 몰려 있는 구간을 확대해서 자세하게 분석
# 클러스터 값이 1, 2, 3에 속하는 데이터만 변수 ndf에 저장함
mask = (df['Cluster'] == 0) | (df['Cluster'] == 4)
ndf = df[~mask]

# 클러스터 값이 1, 2, 3에 속하는 데이터만을 이용해서 분포를 확인
# 산점도 : x='Grocery', y='Frozen' 식료품점, 냉동식품
# 산점도 : x='Milk', y='Delicassen' 우유, 조제식품점
ndf.plot(kind='scatter'
         , x='Grocery', y='Frozen'
         , c='Cluster'
         , cmap='Set1'
         # colorbar 미적용
         , colorbar=False
         , figsize=(10, 10))
ndf.plot(kind='scatter'
         , x='Milk', y='Delicassen'
         , c='Cluster'
         , cmap='Set1'
         # colorbar 적용
         , colorbar=True
         , figsize=(10, 10))
plt.show()
plt.close()