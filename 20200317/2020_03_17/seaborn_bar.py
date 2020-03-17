# seaborn_bar.py

# 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')

# 스타일 테마 설정
# (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('whitegrid')

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

# 막대 그래프 그리기
sns.barplot(x = 'sex', y = 'survived'
            , data = titanic
            , ax = ax1)

# 막대 그래프 그리기
# hue='class' -> class(first, second, third) 나눠 출력
sns.barplot(x = 'sex', y = 'survived'
            , hue = 'class'
            , data = titanic
            , ax = ax2)

# 막대 그래프 그리기 dodge=False
# dodge=False -> 1개의 막대그래프로 출력
sns.barplot(x = 'sex', y = 'survived'
            , hue = 'class'
            , dodge=False
            , data = titanic
            , ax = ax3)

# title 설정
ax1.set_title('titanic survived - sex')
ax2.set_title('titanic survived - sex/class')
ax3.set_title('titanic survived - sex/class(stacked)')

plt.show()