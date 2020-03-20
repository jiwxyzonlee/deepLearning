# iris_datasets.py

from sklearn import datasets

# iris 데이터 로드
iris = datasets.load_iris()

# 1. data (붓꽃 측정값)
data = iris['data']
#print(data)

# 2. DESCR (피셔의 붓꽃 데이터 설명 출력)
# iris Data Set Characteristics
#print(iris['DESCR'])

#- class:
                # - Iris-Setosa
                # - Iris-Versicolour
                # - Iris-Virginica

# 3. target (붓꽃의 품종이 ID 번호로 등록되어 있음)
#print(iris['target'])

# 4. target_names (붓꽃의 품종이 등록되어 있음)
print(iris['target_names'])
# ['setosa' 'versicolor' 'virginica']
