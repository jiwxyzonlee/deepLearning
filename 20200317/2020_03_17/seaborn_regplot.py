# seaborn_regplot.py
# 회귀선이 있는 산점도

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

# 산점도 그래프 그리기
# 선형회귀선 표시 fit_reg = True
sns.regplot(
    # x축 변수
    x='age'
    # y축 변수
    , y='fare'
    # 데이터
    , data=titanic
    # ax 객체 - 첫번째 그래프
    , ax=ax1
    # 회귀선 표시(기본값)
    , fit_reg=True
)

sns.regplot(
# x축 변수
    x='age'
    # y축 변수
    , y='fare'
    # 데이터
    , data=titanic
    # ax 객체 - 두번째 그래프
    , ax=ax2
    # 회귀선 미표시
    , fit_reg=False
)

plt.show()