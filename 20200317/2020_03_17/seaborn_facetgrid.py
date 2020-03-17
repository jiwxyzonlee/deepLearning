# seaborn_facetgrid.py
# 조건을 적용하여 화면을 그리드로 분핛핚 그래프

# 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')

# 스타일 테마 설정
# (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('whitegrid')

# 조건에 따라 그리드 나누기
#  who(man, woman, child) , survived (0 or 1)
g = sns.FacetGrid(data=titanic
                  , col='who'
                  , row='survived')

# 그래프에 적용하기 (히스토그램)
g = g.map(plt.hist, 'age')
plt.show()