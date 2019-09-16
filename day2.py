# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:41:47 2019

@author: user
"""

# 자료 종류, 자료 구조
# 리스트, 튜플, 딕셔너리
# [], (), {키:값, 키:값}
# list(), tuple(), dict(키=값, 키=값)

# 상수 단위 처리 -> 해당 변수의 값을 하나하나 확인해야 함

# 접근 방식
# 1. 숫자(index 방식): 변수명[번호(0부터 시작)]
# ex. x1[0]
# 2. 키 방식: 변수명["키"]
# ex. x2["Age"]

x1=[[1,2,3,[3,4,5]],[4,5,[3,1]]]
len(x1)   # 2

x1[0]   # [1, 2, 3, [3, 4, 5]]
x1[0][3]    # [3, 4, 5]
x1[0][3][1]   # 4

# list에 값 추가
x1.append(5)
x1   # [[1, 2, 3, [3, 4, 5]], [4, 5, [3, 1]], 5]

# tuple
x2=(3,5,2,(5,2,5,(7,4)))
len(x2)   # 4
type(x2)    # tuple
x2[3]   # (5,2,5,(7,4))
x2[3][3] # (7,4)
x2[3][3][0]   # 7
type(x2[3][3])    # tuple

# dictionary
x3={"Age":[3,5], "Name":["kim","lee"]}
type(x3)    # dict
type(x3["Age"])   # list
x3["Age"][0]    # 3
x3.keys()   # dict_keys(['Age', 'Name'])
x3.values()   # dict_values([[3, 5], ['kim', 'lee']])
# 딕셔너리에 값 추가
x3["Height"]=[90,110]
x3    # {'Age': [3, 5], 'Name': ['kim', 'lee'], 'Height': [90, 110]}

# 비교연산자는 상수 단위 즉, 값 하나 단위로 처리하기 때문에 list에 사용할 수 없다.
# list에는 값이 복수개가 들어있기 때문이다.
#  '>' not supported between instances of 'list' and 'int'
x1 > 5
x1[0] > 5
# 하나의 값에 접근해야 가능하다.
x1[0][0] > 5    # false

# 값을 하나하나 가져오려면 번거로우므로 값을 반복해 리턴하는 numpy를 사용한다.
x4=[5,3,2,7,6,9,2]

# 반복문
# indent 중요 
for i in range(len(x4)) :
  print(x4[i])
  
for i in x4 :
  print(i)

# 이제 복수개를 한 번에 비교할 수 있다.
for i in range(len(x4)) :
  print(x4[i] > 5)
print("--end--")  # 위의 반복분이 끝나야 실행된다.
  
for i in x4 :
  print(i > 5)
  
for i in range(len(x4)) :
  print(x4[i] > 5)
  print(i)
  
# numpy
import numpy as np

range(1, 5)   # 1, 2, 3, 4이지만 값이 바로 나오지 않는다.
np.arange(1, 5)   # 1, 2, 3, 4 numpy를 이용해 바로 값을 출력할 수 있다.
len(x4)
np.arange(len(x4))    # array([0, 1, 2, 3, 4, 5, 6])

# [0, 1, 2, 3, 4, 5, 6]가 i에 하나씩 넘겨진다.
for i in np.arange(len(x4)) :
  print(x4[i] > 5)
  print(i)
  
# 중첩 for문 
x5=[[1,2,3], [4,5,6]]
for i in x5:
  for j in i:
      print(j)
  
for i in np.arange(len(x5)):
  for j in np.arange(len(x5[i])):
      print(x5[i][j])

# 탭으로 구분하기
for i in x5:
  for j in i:
      print(j, end="\t")
  print()
  
# if문
for i in x5:
  for j in i:
    if j == 5: continue     # 다음 값으로 skip
    print(j, end="\t")
  print()
  
for i in x5:
  for j in i:
    if j == 5: break     # 5가 되는 순간 분기를 아예 빠져나온다.
    print(j, end="\t")
  print()
  
# 변수 할당
a1, a2=3, 4
a1  # 3
a2  # 4

a3=3, 4
a3  # (3, 4)

# unpacking. 분리해서 값을 받아온다.
a4, *a5=3, 4, 7
a4    # 3
a5    # [4, 7]

# ValueError: too many values to unpack (expected 2)
# 따라서 *를 붙여줘야 한다.
a7, a8=3, 4, 7    

# items - i는 키, j는 값
# x3 =  {'Age': [3, 5], 'Name': ['kim', 'lee'], 'Height': [90, 110]}
for i, j in x3.items():
  print(i)
  print(j)
  
###############################################################################
  
# terminal 실행
!cd
# package 설치
!pip install 패키지명
!conda install 패키지명

# pandas
import pandas as pd

# 해당 csv 파일 경로를 넣어준다.
# sep으로 구분자가 무엇인지 명시해준다.
data1=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", sep=",")

# 제일 첫번째 행은 무조건 헤더로 들어가므로 헤더가 없다고 명시해준다.
data2=pd.read_csv("breast-cancer-wisconsin.data", header=None)
data2

# =============================================================================
# pd.read_csv("파일명", header=None, sep="구분자", )
# C:/how/to/write/directory/path
# C:\\how\\to\\write\\directory\\path
# =============================================================================

type(data2)
data2.head()

# 열별로 집계 값 출력
data2.describe()
# 열 단위로 출력
data2[1]  
type(data2[1])    # pandas.core.series.Series 각 열의 타입은 series다.
# 특정 값에 접근
data2[1][3]     # 6
type(data2[1][3])   # numpy.int64 특정 값이 접근하면 일반 numpy 타입이다.
# 인덱스(=행)로 찾아가기
data2[2,3]      # error
data2.iloc[2,3]   # 1
# 조건에 대한 true, false 출력
data2[1] > 5
# 조건에 맞는 데이터 출력
data2[data2[1] > 5]

# 차트
!pip install matplotlib
import matplotlib.pyplot as plt
data2[1].plot()

data2.columns    # Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='int64') 인덱스 구조
data2.columns.values    # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=int64) 어레이 구조
data2.dtypes    # 들어가있는 데이터의 타입 출


# 이상치: 극단적으로 큰 값 
# 평균: 중앙치를 기준으로 심하게 작거나 크면 이상치가 있다는 뜻이다.
data2.mean()

# 중앙치
data2.median()

# 산술평균: 갖고 있는 값을 모두 더한 뒤 개수로 나눈 값

# 분산: 유사한 평균이면 분산이 작을 수록 좋다. 
data2.var()

# 표준편차: 분산에 루트 씌운 것
data2.std()

# 분위수: 전체 데이터를 몇 등분했는지. ex) 4등분이면 4분위수
# 0.25에 해당하는 값 출력
data2.quantile(0.25)
# 상위/하위 0.5% 출력 
data2.quantile([0.05, 0.25, 0.75, 0.95])

# boxplot 그래프
data2.boxplot()
# data2의 1열에서 3열까지 
data2[1:3].boxplot()
data2[1:2].plot()
# 기본적으로 column명은 string이지만 우리는 자동 생성해둔 것이므로 int로 취급
data2.boxplot(column=[1, 2])
# column명 지정 
data2.columns=["A","B","C","D", "E", "F", "G", "H", "I", "J", "K"]
# 2열부터 4열까지의 코베리언스..??
# 코베리언스: 평균의 차이(편차)를 구해서 두 개의 값이 관련이 있는지 나타냄.
data2[2:4].cov()
data2[["A", "B", "D"]].cov()
data2.iloc[:,1:4].cov()
#  코릴레이션...............?
data2[["A","B","D"]].corr()

# dataframe은 행 단위로 들어가므로 열 단위로 이름을 붙여서 진행하고 싶다면 dict를 이용해 아래와 같이 사용한다. 
d1=pd.DataFrame(dict(A=[1,3,5], B=["A","B","D"]))
# 그냥 A=[1,3,5] 이렇게 쓰면 안 됨.
# d1=pd.DataFrame(A=[1,3,5], B=["A","B","D"])

# 행 단위로 데이터 입력
d4=pd.DataFrame([[3,"M",70],[7,"F",80]])
# 칼럼의 이름 생성 
d4.columns=["Age","Sex","Avg"]
d4
d4["Age"]
d4.Age
type(d4)    # pandas.core.frame.DataFrame

# dataset 추가
# 이렇게 하면 열 방향으로 들어감
d4.append([5,"M","NA"])
# 따라서 정상적으로 넣으려면 대괄호를 한 번 더 써줘야 한다.
# 하지만 인덱스가 설정되어 있지 않아서 0, 1, 2로 들어가있다.
d4.append([[5,"M","NA"]])
# 인덱스 옵션을 설정해준다.
d4.append([[5,"M","NA"]], ignore_index=True)
# 새로운 열을 추가할 때는 기존에 2개의 행이 있었으므로 2개를 함께 써준다.
d4["Height"]=[4,7]
d4
# 이렇게는 생성할 수 없다. 반드시 위처럼 만들어야 한다.
d4.Hee=[8,2]

d4.append(pd.DataFrame([[6,"M",50]]))
# 해당 열이 임시적으로 제거된 결과를 출력한다.
d4.drop(columns="Age")
# 실제 반영되려면 해당 변수에 덮어 써야 한다.
# 이렇듯 dataframe은 바로 반영되는 list와는 다르다.
d4=d4.drop(columns="Age")
d4

# concat은 append와 달리 2개 이상을 한 번에 합칠 수 있다.
# 위아래로 합치러면 서로 동일한 데이터여야 한다. axis를 설정하면 좌우로도 합칠 수 있다. 

data2.columns
list1=["A","C","E"]
list2=["B","F","J"]

s1=data2[list1]
s2=data2[list2]
# iloc은 index만 써야 한다. 섞어서 쓰려면 loc을 사용한다.
s3=data2.loc[0:100, list1]
s4=data2.loc[101:150, list1]
s3
s4
s5=data2.loc[160:190, list1]
s5

# 열 이름 확인: 이름은 다른데 인덱스는 같다.
s1.columns
s2.columns
# concat은 지금까지 dataframe을 쓴 것과 달리 pandas에 직접 적용한다.
pd.concat([s1,s2])
# 좌우 합치기
# 적용하려면 변수에 할당
con1=pd.concat([s1,s2], axis=1)
# 몇행 몇열인지 보는 법
con1.shape
# 기본 방향은 위 아래로 합쳐진다.
con2=pd.concat([s3, s4, s5])
con2=pd.concat([s3, s4, s5], axis=1)
con2

s1.shape
s3.shape
s1.head()
con1.head()
# 모든 열에 대해 출력
con2.iloc[99:104,:]
s3.tail()

list3=["A","F","J"]
s6=data2.loc[1:120, list3]
s6
# merge는 두 개만 붙일 수 있어서 더 많이 붙이려면 계속 연속해서 붙여줘야 한다. 기본은 왼쪽을 기준으로 합쳐진다.
# left join: 왼쪽은 다 두고 중복되는 것만 찾아서 데이터셋을 생성
# right join: 오른쪽은 다 두고 중복되는 것만 찾아서 데이터셋 생성
# 두 개의 변수의 공통된 부분이 무엇인지 on에 설정해준다. 이 키는 복수 개를 쓸 수 있다.
# 동일하지만 이름이 다를 경우는 left와 right에 각각 지정한다. on은 공통으로 쓸 경우 사용한다.
merge1=pd.merge(s1, s6, on="A")
merge1      #  A  C   E  F  J 순으로 출력됨

# left join 옵션
merge2=pd.merge(s1, s6, 'left', on="A")
merge2.shape
merge2  # nan 값이 보임. 중복되지만 값이 없는 것들을 표현.

len(s1.A.unique())
# A를 쓰면 series가 된다.
# 똑같은 값이 몇개 있는지 출력. series에서만 사용 가능.
s1.A.value_counts()
# 중복 값 삭제
s7=s1.A.drop_duplicates()
s7

# C열에서 평균값보다 큰 것을 출력한다. 이 자체가 데이터셋, 데이터 프레임이 된다.
data2[data2.C > data2.C.mean()]
# 따라서 그 데이터셋의 열을 가져오는 것도 가능하다.
data2[data2.C > data2.C.mean()][["B","C"]]

s1.columns
# C와 E를 하나의 열로 통합한다. 원래의 열 이름은 variable이라는 이름으로 바뀐다.
# 원래 그 열에 있던 값은 value로 들어간다.
# variable과 value 이름은 지정해줄 수 있다.
melt1=pd.melt(s1, "A", ["C","E"], var_name="class", value_name="avg")
melt1
