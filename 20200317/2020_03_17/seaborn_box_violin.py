# seaborn_box_violin.py

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
fig = plt.figure(figsize=(15, 10))

# 서브 그래프 만들기
# 2행 2열 - 첫번째 그래프
ax1 = fig.add_subplot(2, 2, 1)
# 2행 2열 - 두번째 그래프
ax2 = fig.add_subplot(2, 2, 2)
# 2행 2열 - 세번째 그래프
ax3 = fig.add_subplot(2, 2, 3)
# 2행 2열 - 네번째 그래프
ax4 = fig.add_subplot(2, 2, 4)

# 1. 박스 그래프 (기본값)
sns.boxplot(x='alive'
            , y='age'
            , data=titanic
            , ax=ax1)

# 2. 박스 그래프 (hue='sex' 추가, 남녀 데이터 구분 출력)
sns.boxplot(x='alive'
            , y='age'
            , hue='sex'
            , data=titanic
            , ax=ax2)

# 3. 바이올린 그래프 (기본값)
sns.violinplot(x='alive'
               , y='age'
               , data=titanic
               , ax=ax3)

# 4. 바이올린 그래프 (hue='sex' 추가, 남녀 데이터 구분 출력)
sns.violinplot(x='alive'
               , y='age'
               , data=titanic
               , hue='sex'
               , ax=ax4)

plt.show()