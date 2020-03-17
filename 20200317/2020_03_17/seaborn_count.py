# seaborn_count.py

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

# 빈도 막대 그래프
sns.countplot(x='class'
              # 팔레트 설정 - 색상 설정
              , palette='Set1'
              , data=titanic
              , ax=ax1)

# 빈도 막대 그래프
# hue='who' who(man, woman, child)값으로 각각 그래프 출력
sns.countplot(x='class'
              # 팔레트 설정 - 색상 설정
              , palette='Set2'
              , data=titanic
              , hue='who'
              , ax=ax2)

# 빈도 막대 그래프
# dodge=False (stacked)
sns.countplot(x='class'
              # 팔레트 설정 - 색상 설정
              , palette='Set3'
              , data=titanic
              , hue='who'
              , dodge=False
              , ax=ax3)

# title 설정
ax1.set_title('titanic class')
ax2.set_title('titanic class - who')
ax3.set_title('titanic class - who(stacked)')

plt.show()