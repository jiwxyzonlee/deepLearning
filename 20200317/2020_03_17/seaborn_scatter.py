# seaborn_scatter.py
# 범주형 데이터의 산점도

# 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')

# 스타일 테마 설정
# (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('darkgrid')

# 그래프 만들기 (figure에 2개의 서브 플롯을 생성)
# 그래프 크기 설정
fig = plt.figure(figsize=(15, 5))

# figure에 2개의 서브 플롯을 생성
# 1행 2열 - 첫번째 그래프
ax1 = fig.add_subplot(1, 2, 1)
# 1행 2열 - 두번째 그래프
ax2 = fig.add_subplot(1, 2, 2)

# 이산형 변수의 분포 - 데이터 분산 미고려
sns.stripplot(
    # x축 변수
    x='class'
    # y축 변수
    , y='age'
    # 데이터셋 - 데이터프레임
    , data=titanic
    # ax 객체 - 첫번째 그래프
    , ax=ax1
)

# 이산형 변수의 분포 - 데이터 분산 고려 (중복 없음)
sns.swarmplot(
    # x축 변수
    x='class'
    # y축 변수
    , y='age'
    # 데이터셋 - 데이터프레임
    , data=titanic
    # ax 객체 - 두번째 그래프
    , ax=ax2
)

# title 설정
ax1.set_title('stripplot()')
ax2.set_title('swarmplot()')

plt.show()