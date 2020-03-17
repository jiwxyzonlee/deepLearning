# pandas_pie02.py

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# 한글 글꼴 설정 - '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

fruit = ['사과', '바나나', '딸기', '오렌지', '포도']
result = [7, 6, 3, 2, 2]

df_fruit = pd.Series(result
                     # 인덱스 설정
                     , index=fruit
                     # 타이틀 값
                     , name='선택한 학생 수')
print(df_fruit)

df_fruit.plot.pie()

# 그래프 출력
plt.show()

# 정교한 pie 그래프 그리기
# pie 간격 설정
explode_value = (0.1, 0, 0, 0, 0)

# 그래프 설정
fruit_pie = df_fruit.plot.pie(
    # 크기 설정
    figsize=(7, 5)
    # 비율 표시, 소수점 자리 제어
    , autopct='%.1f%%'
    # 90도로 시작위치 각도 설정
    , startangle=90
    # 시계 방향 설정
    , counterclock=False
    # 부채꼴 돌출 설정
    , explode=explode_value
    # 그림자 표시
    , shadow=True
    # 표 출력 설정
    , table=True
)

# 불필요한 y축 라벨('선택한 학생 수') 제거
fruit_pie.set_ylabel("")

# main title 설정
fruit_pie.set_title("과일 선호도 조사 결과")

# 그래프를 이미지 파일로 저장
# dpi는 200으로 설정
plt.savefig('saveFruit.png', dpi = 200)
plt.show()