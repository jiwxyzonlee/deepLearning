# seaborn_distplot.py
# 히스토그램과 커널밀도함수

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

# figure에 3개의 서브 플롯을 생성
# 1행 2열 - 첫번째 그래프
ax1 = fig.add_subplot(1, 3, 1)
# 1행 2열 - 두번째 그래프
ax2 = fig.add_subplot(1, 3, 2)
# 1행 3열 - 세번째 그래프
ax3 = fig.add_subplot(1, 3, 3)

# 그래프 그리기
# 기본값 (히스토그램 + 커널밀도함수)
sns.distplot(titanic['fare'], ax=ax1)

# hist = False (커널밀도함수)
sns.distplot(titanic['fare'], hist=False, ax=ax2)

# kde = False (히스토그램)
sns.distplot(titanic['fare'], kde=False, ax=ax3)

# 차트 제목 표시
# 히스토그램/커널밀도함수
ax1.set_title('hist & kde')
# 커널밀도함수
ax2.set_title('kde')
# 히스토그램
ax3.set_title('hist')

plt.show()