# seaborn_heatmap.py

# 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')

# 스타일 테마 설정
# (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('darkgrid')

# 피봇테이블 만들기
table = titanic.pivot_table(index=['sex']
                            # 변수
                            , columns=['class']
                            # 데이터 값의 크기를 기준으로 집계
                            , aggfunc='size')

# 히트맵 그리기
sns.heatmap(
    # 데이터프레임
    table
    # 데이터 값 표시 여부
    , annot=True
    # 정수형 포맷
    , fmt='d'
    # 컬러맵 색깔 설정
    , cmap='YlGnBu'
    # 구분 선 두께
    , linewidth=0.5
    # 컬러바 표시 여부
    , cbar=True
)

plt.show()