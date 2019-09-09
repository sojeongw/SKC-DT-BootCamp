# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

a1=3
a1+4
a2=3.5
a3='abc'
type(a1)
a4=True
True + False

b1=[4, 2, 5]
b1[0]
b1[1]=7
b1
print(b1)

b2=(4,7,2)
b2[1]
b2[1] = 5

b3 = 4, 9, 2
b4, b5 = 4, 7
type(b1)

b6=[[1,2,3],[1,2],[4]]
b6[0][2]

#slicing
b7=[2, 4, 5, 7, 1, 9, 3]
b7[-1] #뒤에서 n번째 
b7[0:-2]
b7[3:]
b7[:-2]
#b7[] #에러
b7[0:5:2]   #0에서 5까지 중에 간격을 2로 두고 출력
b7[::-1]    #역순으로 나열 
# b7[]
# b7>5리스트에서는 더하기와 곱하기만 가능  
a1>5

c3=b7.copy()
c3
b7.append(5)
b7
b7.pop()    #맨 마지막 제거
b7
b7.remove(1)    #지정한 값 제거
b7
b8="abaacdf"
b8[0]   #string 한 글자 한 글자가 배열 한 칸으로 인식됨
b8.count("a")

# {키:값}는 dict()과 같다. 값에는 리스트, 튜플, 딕셔너리 다 들어갈 수 있다.
d1={"Age":[3,5,9], "Height":[150,160,180]}
type(d1)    # dict
# d1[0] 숫자로 접근 불가
d1["Age"]
type(d1["Age"]) # list
d1.keys()   # key값만 리턴. dict_keys(['Age', 'Height'])
d1.values() # dict_values([[3, 5, 9], [150, 160, 180]])
d1.items()  # dict_items([('Age', [3, 5, 9]), ('Height', [150, 160, 180])])

# 패키지 전체 불러오기
# import func1

# 특정 함수만 가져오 
from func1 import add1, add2

add1(3,4)
add2(3,4)
add1(5,6,2) # 세번째 인자를 정해서 넘겨주면 함수에 미리 정해놨던 값은 적용되지 않는다.

import numpy as np

f1=np.array(b7) # [2, 4, 5, 7, 1, 9, 3]
b7>5  # not supported between instances of 'list' and 'int'
b7[0]>5
f1>5  # array([False, False,  True,  True, False])

sum(b7)
sum(f1)
type(b7)  # list
type(f1)  # numpy.ndarray

f2=np.array([[2,5,7],[5,6,2]])
f2.shape  # (2,3) 즉, 2행 3열이다.
f2[0]   # array([2, 5, 7])
f2[0][1]  # 5
f2[0,1]   # 5

f1[f1>5]  # array([7, 9])
np.sum(f2, 0)   # array([ 7, 11,  9]) axis 옵션을 0으로 하면 열 방향으로
np.sum(f2, 1)   # array([14, 13]) axis 옵션을 1로 하면 행 방향으로 

np.argmax(f2, 0)  # array([1, 1, 0], dtype=int64) 열 방향으로 최대값이 있는 위치값 리턴 

# if 함수
# np.where(조건, 참인 경우 실행할 내용, 거짓인 경우 실행할 내용)
f1  # array([2, 4, 7, 9, 3])
f3=np.where(f1>5, 1, 0)  
f3  # array([0, 0, 1, 1, 0])

f4=range(5)   # 0에서 4까지 
f4  # range(0, 5) 실제 들어가있는 값은 알 수가 없음.

f5=np.arange(0,5)  # array([0, 1, 2, 3, 4]) 실제 들어가있는 값을 알 수 있음.
f5

# 자리를 0으로 채우겠다. 첫번째 인자는 shape, 즉 몇행 몇열인지 지정.
np.zeros(10)  # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) 
np.zeros((5,3)) # 행, 열을 둘 다 쓸 때는 소괄호로 감싸줘야 한다.

f2  # array([[2, 5, 7], [5, 6, 2]])
f2 * f2  # array([[ 4, 25, 49], [25, 36,  4]]) 같은 자리에 있는 숫자끼리 곱한다.
np.matmul(f2, f2.T)   # array([[78, 54], [54, 65]]) 매트릭스별로 multiply
f2.reshape(3,2)   # array([[2, 5],[7, 5],[6, 2]]) 행 방향으로 읽으면서 구조에 맞게 재배치
f2=f2.reshape(3,2)  # 적용하려면 다시 assign 해줘야 한다.
f2[0]   # 할당 전 array([2, 5, 7]) / 할당 후 array([2, 5])

np.ones(10) # array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]) 자리를 1로 채우겠다.
np.linspace(0, 1, 5)  # array([0.  , 0.25, 0.5 , 0.75, 1.  ])
