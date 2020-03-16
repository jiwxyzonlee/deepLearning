# scatter_graph03.py

# 산점도 - 지역별 인구 밀도 현황

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 한글 글꼴 설정 - (Windows 운영체제의 경우) '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 지역 배열 생성
city = ['서울', '인천', '대전', '대구', '울산', '부산', '광주']

# 위도(latitude)와 경도(longitude)
lat = [37.56, 37.45, 36.35, 35.87, 35.53, 35.18, 35.16]
lon = [126.97, 126.70, 127.38, 128.60, 129.31, 129.07, 126.85]

# 인구 밀도(명/km^2): 2017년 통계청 자료
pop_den = [16154, 2751, 2839, 2790, 1099, 4454, 2995]

# 마커의 크기 지정 (임의지정, 상대적인 크기로 나타냄)
size = np.array(pop_den) * 0.2

# 마커의 색상 지정
colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']

plt.scatter(lon, lat
            , s = size
            , c = colors
            , alpha = 0.5)

plt.xlabel('경도(longitude)')
plt.ylabel('위도(latitude)')

plt.title('지역별 인구 밀도(2017)')

for x, y, name in zip(lon, lat, city):
    # zip(): list slicing (한 줄 만 김밥을 자르기)
    # 위도 경도에 맞게 도시 이름 text 함수로 출력
    # 해당 마크 뒤에 도시 이름 전달
    plt.text(x, y, name)
    # 해당 도시 명을 x, y 좌표 위치에 출력

plt.show()