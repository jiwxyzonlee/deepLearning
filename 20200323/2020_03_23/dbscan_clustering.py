# dbscan_clustering.py

# 학교알리미 공개용 데이터 중에서 서울시 중학교 졸업생의 진로현황 데이터셋을 사용하여
# 고등학교 진학률이 비슷한 중학교끼리 군집(cluster)을 생성

# 기본 라이브러리 불러오기
import pandas as pd
import folium

# 학교알리미 공개용 데이터 중에서 서울시 중학교 졸업생의 진로현황 데이터셋
file_path = '2016_middle_shcool_graduates_report.xlsx'
df = pd.read_excel(file_path, header=0)
#print(df)

# IPython Console 디스플레이 옵션 설정하기
# 출력화면 너비
pd.set_option('display.width', None)
# 출력할 행의 개수 한도
pd.set_option('display.max_rows', 100)
# 출력할 열의 개수 한도
pd.set_option('display.max_columns', 30)
# 출력할 열의 너비
pd.set_option('display.max_colwidth', 20)
# 유니코드 사용 너비 조정
pd.set_option('display.unicode.east_asian_width', True)

# 데이터프레임의 열 이름 출력
print(df.columns.values)


# 데이터 형태와 자료형 확인
# print(df.head()); print()
# df.info(); print()

# 데이터 통계 요약정보 확인
# print(df.describe())

# 지도에 위치 표시
mschool_map = folium.Map(
    location=[37.55, 126.98]
    , tiles='Stamen Terrain'
    , zoom_start=12
)

# 중학교 위치정보를 CircleMarker로 표시
for name, lat, lng in zip(df.학교명, df.위도, df.경도):
    folium.CircleMarker(
        [lat, lng]
        # 원의 반지름
        , radius=5
        # 원의 둘레 색상
        , color='brown'
        , fill=True
        # 원 색
        , fill_color='coral'
        # 투명도
        , fill_opacity=0.7
        # 팝업 기능(원형 마커를 클릭하면 학교명이 팝업으로 출력)
        , popup=name
    ).add_to(mschool_map)

# 지도를 html 파일로 저장하기
mschool_map.save('seoul_mschool_location.html')

# 지역, 코드, 유형, 주야 열을 원핫인코딩 처리
from sklearn import preprocessing

# label encoder 생성
label_encoder = preprocessing.LabelEncoder()

# 모델이 인식할 수 없는 문자형 데이터를 원핫인코딩으로 처리하여 더미 변수에 저장
# 지역구 이름
onehot_location = label_encoder.fit_transform(df['지역'])
# 3, 5, 9
onehot_code = label_encoder.fit_transform(df['코드'])
# 국립, 공립, 사립
onehot_type = label_encoder.fit_transform(df['유형'])
# 주간, 야간
onehot_day = label_encoder.fit_transform(df['주야'])

# 원핫인코딩된 결과를 새로운 열(변수)에 할당
# 지역
df['location'] = onehot_location
# 코드
df['code'] = onehot_code
# 유형
df['type'] = onehot_type
# 주야
df['day'] = onehot_day

print(df.head())
# location  code  type  day
print()

# sklearn 라이브러리에서 cluster 군집 모델 가져오기
from sklearn import cluster

# 분석1. 과학고, 외고국제고, 자사고 진학률로 군집
# 분석에 사용핛 속성을 선택 (과학고, 외고국제고, 자사고 진학률)
print('분석1. 과학고, 외고국제고, 자사고 진학률로 군집')
# 각 컬럼의 인덱스 번호
columns_list = [10, 11, 14]
x = df.iloc[:, columns_list]

print(x[:5])
print()

# 설명 변수 데이터를 정규화
x = preprocessing.StandardScaler().fit(x).transform(x)

# DBSCAN 모델 객체 생성
# 밀도 계산의 기준이 되는 반지름 R(eps=0.2)과 최소 포인트 개수 M(min_samples=5) 설정
dbm = cluster.DBSCAN(eps=0.2, min_samples=5)
# DBSCAN 모델 학습
dbm.fit(x)

# 예측 (군집) 결과를 출력할 열(속성)의 값 구하기
# 모델의 labels_ 속성으로 확인하면 5개의 클러스터 값 ( -1, 0, 1, 2, 3 ) 으로 나타남
cluster_label = dbm.labels_
print(cluster_label)
# -1, 0, 1, 2, 3로 구성됨 (-1은 outlier)
print()
# 클러스터 0 : 외고_국제고와 자사고 합격률은 높지만 과학고 합격자가 없다.
# 클러스터 1 : 자사고 합격자만 존재하는 그룹
# 클러스터 2 : 자사고 합격률이 매우 높으면서 과학고와 외고_국제고 합격자도 일부 존재
# 클러스터 3 : 과학고 합격자 없이 외고_국제고와 자사고 합격자를 배출한 점은 클러스터 0과 비슷하지만
# 외고_국제고 합격률이 클러스터 0에 비해현저하게 낮다.

# 예측(군집) 결과를 저장할 열(Cluster)을 데이터프레임에 추가
# Cluster에 열 추가
df['Cluster'] = cluster_label
df.info()
# 25  Cluster     415 non-null    int64
print()

# 클러스터 값으로 그룹화하고, 그룹별로 내용 출력 (첫 5행만 출력)
# 1:지역명, 2:학교명, 4:유형
grouped_cols = [1, 2, 4] + columns_list
grouped = df.groupby('Cluster')
for key, group in grouped:
    print('* key :', key)
    # 클러스터 값: -1, 0, 1, 2, 3
    print('* number :', len(group)); print()
    # 각 클러스터 속한 학교수

    # 5개의 데이터 출력
    print(group.iloc[:, grouped_cols].head())
    print()

"""
* key : -1
* number : 255

     지역                               학교명  유형  과학고  외고_국제고  자사고
0  성북구  서울대학교사범대학부설중학교.....    국립   0.018        0.007   0.227
1  종로구  서울대학교사범대학부설여자중학교...  국립   0.000        0.035   0.043
2  강남구           개원중학교                  공립   0.009        0.012   0.090
3  강남구           개포중학교                  공립   0.013        0.013   0.065
4  서초구           경원중학교                  공립   0.007        0.010   0.282

* key : 0
* number : 102

      지역          학교명  유형  과학고  외고_국제고  자사고
13  서초구  동덕여자중학교  사립     0.0        0.022   0.038
22  강남구      수서중학교  공립     0.0        0.019   0.044
28  서초구      언남중학교  공립     0.0        0.015   0.050
34  강남구      은성중학교  사립     0.0        0.016   0.065
43  송파구      거원중학교  공립     0.0        0.021   0.054

* key : 1
* number : 45

         지역          학교명  유형  과학고  외고_국제고  자사고
46     강동구      동신중학교  사립     0.0          0.0   0.044
103    양천구      신원중학교  공립     0.0          0.0   0.006
118    구로구      개봉중학교  공립     0.0          0.0   0.012
126  영등포구      대림중학교  공립     0.0          0.0   0.050
175    중랑구  혜원여자중학교  사립     0.0          0.0   0.004

* key : 2
* number : 8

       지역      학교명  유형  과학고  외고_국제고  자사고
20   서초구  서초중학교  공립   0.003        0.013   0.085
79   강동구  한영중학교  사립   0.004        0.011   0.077
122  구로구  구일중학교  공립   0.004        0.012   0.079
188  동작구  대방중학교  공립   0.003        0.015   0.076
214  도봉구  도봉중학교  공립   0.004        0.011   0.072

* key : 3
* number : 5

         지역      학교명  유형  과학고  외고_국제고  자사고
35     서초구  이수중학교  공립     0.0        0.004   0.100
177  동대문구  휘경중학교  공립     0.0        0.004   0.094
191    동작구  문창중학교  공립     0.0        0.004   0.084
259    마포구  성사중학교  공립     0.0        0.004   0.078
305    강북구  강북중학교  공립     0.0        0.004   0.088
"""

# 그래프로 표현 - 시각화
colors = {-1:'gray', 0:'coral', 1:'blue', 2:'green', 3:'red', 4:'purple',
          5:'orange', 6:'brown', 7:'brick', 8:'yellow', 9:'magenta', 10:'cyan'}
cluster_map = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain',
                         zoom_start=12)
for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster):
    folium.CircleMarker([lat, lng],
                        radius=5, # 원의 반지름
                        color=colors[clus], # 원의 둘레 색상
                        fill=True,
                        fill_color=colors[clus], # 원을 채우는 색
                        fill_opacity=0.7, # 투명도
                        popup=name
                        ).add_to(cluster_map)
# 지도를 html 파일로 저장하기
cluster_map.save('seoul_mschool_cluster.html')

# 분석2. 과학고, 외고_국제고, 자사고 진학률, 유형(국립,공립,사립)으로 군집
# X2 데이터셋에 대하여 위의 과정을 반복(과학고, 외고_국제고, 자사고 짂학률, 유형)
print('분석2. 과학고, 외고_국제고, 자사고 짂학률, 유형(국립,공립,사립)으로 굮집')
columns_list2 = [10, 11, 14, 23]
x2 = df.iloc[:, columns_list2]
print(x2[:5])
print('\n')
# 설명 변수 데이터를 정규화
x2 = preprocessing.StandardScaler().fit(x2).transform(x2)
# DBSCAN 모델 객체 생성
# 밀도 계산의 기준이 되는 반지름 R(eps=0.2)과 최소 포인트 개수 M(min_samples=5) 설정
dbm2 = cluster.DBSCAN(eps=0.2, min_samples=5)
# DBSCAN 모델 학습
dbm2.fit(x2)

# 예측(군집) 결과를 저장할 열(Cluster2)을 데이터프레임에 추가
df['Cluster2'] = dbm2.labels_ # Cluster2 열 추가됨
# 클러스터 값으로 그룹화하고, 그룹별로 내용 출력 (첫 5행만 출력)
# 1:지역명, 2:학교명, 4:유형
grouped2_cols = [1, 2, 4] + columns_list2
grouped2 = df.groupby('Cluster2')
for key, group in grouped2:
    # 클러스터 값: -1, 0 ~ 10
    print('* key :', key)
    # 각 클러스터 속한 학교수
    print('* number :', len(group))
    # 5개의 데이터 출력
    print(group.iloc[:, grouped2_cols].head())
    print()

cluster2_map = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain',
                          zoom_start=12)
for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster2):
    folium.CircleMarker(
        [lat, lng]
        # 원의 반지름
        , radius=5
        # 원의 둘레 색상
        , color=colors[clus]
        , fill=True
        # 원을 채우는 색
        , fill_color=colors[clus]
        # 투명도
        , fill_opacity=0.7
        , popup=name
    ).add_to(cluster2_map)
# 지도를 html 파일로 저장하기
cluster2_map.save('seoul_mschool_cluster2.html')

# 분석3. 과학고, 외고_국제고 군집
# X3 데이터셋에 대하여 위의 과정을 반복(과학고, 외고_국제고)
print('분석3. 과학고, 외고_국제고 굮집')
columns_list3 = [10, 11]
x3 = df.iloc[:, columns_list3]
print(x3[:5])
print('\n')

# 설명 변수 데이터를 정규화
x3 = preprocessing.StandardScaler().fit(x3).transform(x3)

# DBSCAN 모델 객체 생성
# 밀도 계산의 기준이 되는 반지름 R(eps=0.2)과 최소 포인트 개수 M(min_samples=5) 설정
dbm3 = cluster.DBSCAN(eps=0.2, min_samples=5)

# DBSCAN 모델 학습
dbm3.fit(x3)

# 예측(군집) 결과를 저장할 열(Cluster3)을 데이터프레임에 추가
# Cluster3 열 추가됨
df['Cluster3'] = dbm3.labels_

# 클러스터 값으로 그룹화하고, 그룹별로 내용 출력 (첫 5행만 출력)
# 1:지역명, 2:학교명, 4:유형
grouped3_cols = [1, 2, 4] + columns_list3

grouped3 = df.groupby('Cluster3')

for key, group in grouped3:
    # 클러스터 값: -1, 0 ~ 6
    print('* key :', key)
    # 각 클러스터 속한 학교수
    print('* number :', len(group))
    # 5개의 데이터 출력
    print(group.iloc[:, grouped3_cols].head())
    print()

"""
* key : -1
* number : 61
     지역                             학교명  유형  과학고  외고_국제고
0  성북구  서울대학교사범대학부설중학교.....  국립   0.018        0.007
3  강남구           개포중학교                공립   0.013        0.013
6  강남구         압구정중학교                공립   0.015        0.036
7  강남구  단국대학교사범대학부속중학교.....  사립   0.032        0.005
8  강남구           대명중학교                공립   0.013        0.029

* key : 0
* number : 160
      지역                               학교명  유형  과학고  외고_국제고
1   종로구  서울대학교사범대학부설여자중학교...  국립     0.0        0.035
13  서초구       동덕여자중학교                  사립     0.0        0.022
22  강남구           수서중학교                  공립     0.0        0.019
28  서초구           언남중학교                  공립     0.0        0.015
29  강남구           언북중학교                  공립     0.0        0.007

* key : 1
* number : 111
      지역      학교명  유형  과학고  외고_국제고
2   강남구  개원중학교  공립   0.009        0.012
4   서초구  경원중학교  공립   0.007        0.010
5   강남구  구룡중학교  공립   0.007        0.007
11  강남구  대치중학교  공립   0.007        0.024
14  서초구  반포중학교  공립   0.010        0.013

* key : 2
* number : 50
         지역      학교명  유형  과학고  외고_국제고
46     강동구  동신중학교  사립     0.0          0.0
103    양천구  신원중학교  공립     0.0          0.0
118    구로구  개봉중학교  공립     0.0          0.0
126  영등포구  대림중학교  공립     0.0          0.0
160  동대문구  숭인중학교  공립     0.0          0.0

* key : 3
* number : 11
         지역      학교명  유형  과학고  외고_국제고
100    양천구  신남중학교  공립   0.007          0.0
115    강서구  화곡중학교  사립   0.008          0.0
151  동대문구  대광중학교  사립   0.005          0.0
194    관악구  봉원중학교  공립   0.004          0.0
209    노원구  광운중학교  사립   0.005          0.0

* key : 4
* number : 12
      지역          학교명  유형  과학고  외고_국제고
9   강남구      대왕중학교  공립   0.006        0.028
27  강남구      신사중학교  공립   0.006        0.032
69  송파구      오주중학교  공립   0.003        0.028
72  송파구      잠실중학교  공립   0.007        0.030
96  양천구  봉영여자중학교  사립   0.006        0.028

* key : 5
* number : 5
       지역      학교명  유형  과학고  외고_국제고
16   강남구  봉은중학교  공립   0.010        0.010
85   강서구  덕원중학교  사립   0.010        0.010
179  동작구  강현중학교  공립   0.011        0.011
262  마포구  숭문중학교  사립   0.010        0.010
366  마포구  상암중학교  공립   0.012        0.012

* key : 6
* number : 5
         지역      학교명  유형  과학고  외고_국제고
89     강서구  마포중학교  사립   0.015        0.010
112    강서구  염창중학교  공립   0.015        0.009
265  서대문구  신연중학교  공립   0.016        0.011
287    광진구  광남중학교  공립   0.016        0.010
359    관악구  구암중학교  공립   0.017        0.011


Process finished with exit code 0

"""