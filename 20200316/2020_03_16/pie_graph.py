# pie_graph.py

import matplotlib.pyplot as plt
import matplotlib

# 한글 글꼴 처리 '맑은 고딕'으로 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

fruit = ['사과', '바나나', '딸기', '오렌지', '포도']
# 과일 수량
result = [7, 6, 3, 2, 2]

## 1. 파이 그래프가 타원모양으로 출력
plt.pie(result)
plt.show()

## 2. 파이 그래프의 너비와 높이를 1대 1로 출력(원모양) : figsize=(5, 5)
plt.figure(figsize = (5, 5))
plt.pie(result)
plt.show()

## 3. 라벨과 비율 추가
plt.figure(figsize=(5, 5))
plt.pie(result
        , labels= fruit
        # 비율 표시 형식
        , autopct='%.1f%%')
plt.show()

## 4. 각도 90도에서 시작해서 시계방향으로 설정
plt.figure(figsize=(5, 5))
plt.pie(result
        , labels= fruit
        , autopct='%.1f%%'
        # 부채꼴이 그려지는 각도(기본은 0)
        , startangle=90
        # 시계방향 설정
        , counterclock = False)
plt.show()

## 5. 사과 부분 부채꼴로 돌출시킨 뒤 그림자 효과 적용
#explode_value = (0.1, 0, 0, 0, 0)
explode_value = (0.5, 0.1, 0, 0, 0)
#fruit = ['사과', '바나나', '딸기', '오렌지', '포도']
plt.figure(figsize=(5, 5))
plt.pie(result
        , labels= fruit
        , autopct='%.1f%%'
        , startangle=90
        , counterclock = False
        # 부채꼴이 원에서 돌출
        , explode=explode_value
        # 그림자 효과
        , shadow=True)
plt.show()