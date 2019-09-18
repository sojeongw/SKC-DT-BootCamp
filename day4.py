# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:47:09 2019

@author: user
"""

a1=[3,6,9,2,7,20,50,6,7]

# =============================================================================
#  산술평균: mean()
#  지금 갖고 있는 값을 다 더한 후 개수로 나눈 것 = sum(xi)/n
#  n으로 나눴다는 것은 똑같은 비율로 나눴다는 의미이다.
#  1/n이 가지는 weight(가중치)의 확률의 모음집.
#  각각의 값의 비중이 동일하기 때문에 상대적으로 작은 값/큰 값을 받으면 영향을 받는다.
#  이상치(패턴과 다른 값)에 영향을 받는다.
# 
#  중앙값: median()
#  가장 작은값부터 오름차순으로 정렬을 한 뒤 순서적으로 정 중앙인 값.
#  짝수개인 경우 다 더한 후 개수로 나눈다. 즉, 짝수와 홀수의 산출 방식이 다르다.
#  이상치에 영향을 크게 받지 않는다.
# 
#  최빈수: mode()
#  자주 등장하는 수. 빈도 수가 높은 값을 리턴한다.
#  최빈수는 산술평균, 중앙값과 달리 여러 개의 값이 나올 수 있다.
#  정규분포 그래프는 평균에 위치할 때 최빈수가 제일 많이 나온다.
#
#  이상치: 극단적인 값.
#  이상치가 작은 쪽이면 산술평균이 아래로 가고 크면 위로 간다.
#  수치를 보고 위의 내용을 파악할 수 있어야 한다.
# =============================================================================

# a1은 list 구조이기 때문에 아래처럼 쓸 수 없다.
a1.mean()
mean(a1)

# mean을 쓰려면 numpy와 pandas를 이용해야 한다.
import numpy as np

np.mean(a1)
np.median

# mode는 pandas에 있다.
import pandas as pd

# 강제적으로 dataframe이나 series 값으로 변환해야 한다.
# Series()로 강제적으로 한 열로 들어가게 한다.
pd.Series(a1).mean()
a2=pd.Series(a1)

a2.mean()
a2.median()
a2.mode()   # 리턴값이 두 개 -> 6과 7

# 한꺼번에 결과 산출하는 법. 단, 여기엔 mode가 없다.
# series였으면 series로 리턴하고 dataframe이면 열단위로 적용해서 리턴한다.
a2.describe()

# 데이터의 10프로, 30프로, 90프로에 해당하는 데드라인 위치값을 찾아준다.
a2.quantile([0.1, 0.3, 0.9])

# 기술통계: 데이터 탐색 중 요약, 정리하는 부분을 일컫는다. 수치적, 그래프, 표 등으로 표현됨.
# crosstab()으로 교차해서 빈도를 구할 수 있다.
# crosstab(행단위의 index, 열방향인 columns, 변수값 values) index는 행 이름, 기준이 되는 값
# 행과 열을 이용해서 교차표를 정리할 때 이용한다.
pd.crosstab()

# pivot_table(실제 data, 행의 위치 index, 열의 위치 columns) aggfunc에 집계낼 함수를 넣어주면 그 함수에 해당하는 값이 나온다.
# pivot은 values를 많이 이용한다.
pd.pivot_table()

# series 단위로 각 열들에 대한 빈도를 간단히 구할 수 있다.
a2.value_counts()

# =============================================================================
#  데이터가 여러개인 경우는 집계를 한 다음에 위의 함수를 진행한다.

#  먼저 깨끗한 데이터를 만든다 = 전처리
#  해당 dataframe.dropna(): na 데이터를 제거한다.
#  해당 dataframe.fillna(0): na에 0을 넣는다.

#  해당 dataframe.concat(): 강제로 데이터를 합친다.
#  해당 dataframe.merge(): 복수 개의 데이터를 결합한다. 기준을 다르게 설정할 수 있음.
#  해당 dataframe.isna(): na가 존재하는지 체크한다.
#  해당 dataframe.notna(): na가 존재하지 않으면 true를 돌려준다.

#  true라면 이걸 반복적으로 열단위로 처리할 수 있다. 
#  dataframe.apply(lambda 어쩌구): 데이터 처리 방법을 몽땅 섞어서 처리한다. 반복적으로 열 단위에 적용한다.
#  dataframe.map(): 비슷하지만 셀 단위로 처리할 수 있다.
#  dataframe.applymap(): 열과 셀 단위로 처리한다.
# =============================================================================

# =============================================================================
#  dataframe.where(조건, 실행할 것): 특정 조건에 대해 적용한다.
#  dataframe[조건]: 조건에 해당하는 dataset을 바로 추출한다.
#  groupby, crosstab, pivot_table을 dataframe에서 일반적으로 제일 많이 쓴다.
# =============================================================================


# 오늘부터는 matplotlib, scipy, statsmodels에 대해 집중적으로 다룬다.
# matplotlib.pyplot이 numpy로 데이터를 읽어올 수 있게 지원한다.

# plot.타입
# plot(타입): 일일이 변수를 써줘야하므로 plot.타입 방식으로 진행한다.
a2.plot.bar()

# 지금은 값 그 자체를 날 것으로 표현했다.
# value_counts를 사용해보자. 하지만 a1가 numpy 구조면 아래를 실행할 수 없다.
a2.value_counts().plot.bar()
# value_counts 구조만 다시 한 번 보자. series로 되어있음을 알 수 있다.
a2.value_counts()
# 만약 numpy구조로 나오면 plot 기능을 쓸 수 없다.
# 즉, pyplot은 pandas 구조(무조건 열 단위로 처리하는)로만 사용할 수 있다.
# numpy는 행과 열 둘 다 처리가능한 구조다. 그래서 방향을 직접 지정할 수 있다.

import matplotlib.pyplot as plt

# plt 모듈이 있으면 바로 bar로 만들 수 있다. bar(해당 데이터, 높이값)
# arange로 0부터라는 레이블을, 높이는 a2의 실제값으로 설정해준다.
plt.bar(x=np.arange(a2.shape[0]), height=a2)

# pandas는 위처럼 하지 않아도 자동으로 설정해준다.
a3=a2.value_counts()
# a3는 series로 되어있다.
type(a3)
# 행의 위치에 있는 값 즉, index를 확인한다.
a3.index
# index 중 실제 값을 가져오려면 values를 사용한다. 그럼 array 구조가 된다.
a3.index.values
# bar(실제 들어갈 x축 값, 높이) x=으로 명시해주면 x바에 뭘 표시할지 설정할 수 있다. 
# 어느 쪽에 극단적인 값이 몰려있는지, 흩어져있는 정도를 확인할 수 있다.
# 30-40이 비어있으므로 두 갈래로 분리되어 몰려있는 패턴임을 알 수 있다.
plt.bar(x=a3.index.values, height=a3)
# 이 그래프를 보고 값이 어느 범위 안에 들어갈 때 그룹화가 가능하므로 데이터의 특징을 탐색할 수 있다.

# color 및 orientation 지정
# horizontal은 barh를 사용하며, x와 y가 바뀐다.
plt.barh(a3.index.values, a3, color="indigo", orientation='horizontal')

# =============================================================================
# numpy 짚고 넘어가기
b1=np.array([[3,5,8],[5,7,2]])
b1.shape
type(b1)
# numply는 mean을 쓸 수 없다. array 구조이기 때문이다.
mean(b1)
# array 구조에 맞게 쓰려면 np를 사용한다. 이는 전체 값을 하나의 값을 보고 평균을 낸다.
np.mean(b1)
# numpy라서 열 방향으로 선택할 수 있다.
np.mean(b1, axis=0)
# numpy는 요소를 인덱스로 직접 지정할 수 있다.
b1[0,1]

b2=pd.DataFrame(b1)
# dataframe이 되면 요소를 직접 지정할 수 없다.
b2[0,1]   # 실패
# 이렇게 하나에 접근한 뒤에 그 안으로 들어가야 한다.
b2[0][1]
# 다른 방법으로는 iloc을 써서 행과 열의 위치를 지정할 수 있다. iloc[행,열]
b2.iloc[0,1]
# pandas는 기본이 열 처리이기 때문에 mean()을 계산하면 열 대로 나온다. 
b2.mean()
# pandas를 numpy 구조로 보고 처리할 수도 있다. 자동으로 열의 값으로 들어가지만 numpy처럼 axis를 설정할 수 있다.
np.mean(b2, axis=1)

# pandas는 무조건 열 방향이기 때문에 plot()을 바로 사용할 수 있다. numpy는 행을 어떻게 할지, 열을 어떻게 할지 하나하나 설정해줘야 사용 가능하다.
b1.plot()   # 불가능
b2.plot()   # 가능 -> pandas가 기본 열 단위로 처리하기 때문

# 만약 b1의 요소를 하나의 열로 만들고 싶다면 dict로 바꾼 뒤 DataFrame으로 변환한다.
b1  # 각각의 값이 셀이 들어가있다.
b2=dict(a=[[3,5,8],
           [5,7,2]])
b3=pd.DataFrame(b2)   # 한 열의 값으로 들어간다.
# b3는 value를 보면 column names가 a로 지정되어 있으므로 a로 불러와야 한다.
b3['a']
type(b3)
type(b3['a'])
# 위의 값(pandas로 접근한 내부값 즉 b3['a']안에 있는 값)은 list 구조이므로 list 처럼 접근할 수 있다.
# dict 구조였다면 key 값으로 접근한다.
b3.loc[0, 'a']
b3.iloc[0,0][1]

# array는 무조건 한 데이터 종류만 가져올 수 있지만 dataframe은 여러 종류의 데이터를 넣을 수 있다.
# numpy는 pandas를 수용하지 못하지만 pandas는 numpy를 수용한다.
# =============================================================================

# subplots(): matplotlib은 기본적으로는 오버랩을 하게 되어있는데, 이 함수를 이용해 여러 개의 차트를 설정하고 그릴 수 있도록 기능을 제공한다.
# 오버랩은 하나하나 블럭 잡고 실행하기만 하면 된다.

# =============================================================================
# 오버랩 방식
import matplotlib.pyplot as plt
import numpy as np

# x를 arange로 자동 생성한다. 0부터 시작하면 축에 겹치기 때문에 1부터 시작해서 10까지 0.1의 간격으로 만든다. 간격을 줄일 수록 촘촘하고 부드러운 그래프가 나온다.
x=np.arange(1, 10, 0.1)
y=x*0.2
y2=np.sin(x)

# 첫번째 Axes 설정 (x축, y축, 컬러, 범례)
plt.plot(x, y, 'b', label='first')
# 두번째 Axes 설정
plt.plot(x, y2, 'r', label='second')
# plot() 블락을 쭉 잡고 실행해야 한다. 안그러면 개별 실행되어 그래프가 동시에 출력되지 않는다.

# x축 이름
plt.xlabel('x axis')
# y 축 이름
plt.ylabel('y axis')
# 그래프 이름
plt.title('matplotlib sample')
# 그래프 범례의 위치 설정
plt.legend(loc='upper right')
plt.show()

# 여러 개의 차트를 그릴 때 개별적으로 직접 선의 종류의 색을 지정할 수 있다.
plt.plot(x, y, "gD")
plt.plot(x, y2, "r:")

# 아래의 방법 둘 다 가능한데 주로 후자처럼 color가 맨 앞에 오는 걸 많이 쓴다.
# fmt = "[marker][line][color]"
# fmt = "[color][line][marker]"

# =============================================================================

# =============================================================================
# scatter plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

salary=pd.read_csv("https://raw.githubusercontent.com/duchesnay/pylearn-doc/master/data/salary_table.csv")

# 불러온 csv의 experience, salary column을 각각 x, y축으로 그래프로 출력한다.
# o 옵션을 주면 scatter plot이 된다.
# 아래의 그래프처럼 x값이 증가할 때 y값도 증가하면 양의 상관이라고 한다.
# 상황에 따라 양의 상관이 좋을 수도, 음의 상관이 좋을 수도 있다.
# 조율을 위해서 양과 음 둘 다 표현하는 것도 좋다.
plt.plot(salary["experience"],
         salary["salary"], "o")

# 위의 그래프는 양의 상관이 존재한다는 것만 알 수 있다. 그룹에 따라 양의 상관인지 알아보자.
# 먼저 변수명을 확인한다.
salary.columns
# education에서 어떤 요소가 있는지 확인한다.
edu=salary["education"].unique()
# edu에 담아 각 education 요소를 불러올 수 있게 되었다.
edu[0]
# edu 요소에 따라 색상을 설정한다.
col1=np.where(salary.education==edu[0], 'r', 
    np.where(salary.education==edu[1], 'b', 'k'))

# 설정한 색을 적용해준다. 색이 single로 들어갈땐 plot으로 적용해도 되나, sequence로 쓰려면 특정 plot(scatter 등)으로 설정해줘야 한다.
plt.scatter(salary["experience"],
         salary["salary"], color=col1)

# 또다른 방법: 각각의 x값이 넘어올 때마다 col2를 적용한 c1을 만든다.
col2={edu[0]:'r', edu[1]:'b', edu[2]:'k'}
# series 단위의 apply는 개별 값이 되고, dataframe의 apply는 열이 된다.
c2=salary["education"].apply(lambda x: col2[x])
plt.scatter(salary["experience"],
         salary["salary"], color=c2)

# =============================================================================

# =============================================================================
# box plot: 이상치를 쉽게 찾아낼 수 있음.
# 미리 만들어진 팔레트를 사용할 수 있다.
# =============================================================================

# =============================================================================
# heap map

# 이항 분포: 결과가 둘 중 하나인 것이 여러 번 일어날 때의 분포. 동전 앞뒤, yes or no 등.
# 5*5의 난수 생성
np.random.random([5, 5])
np.random.randn(5,5)    # 표준 정규 분포로 생성됨
# 표준 정규 분포: 95프로(2 시그마) 이내에 들어가 있는 것이 정규 분포.
# 분포 함수: 정규 분포의 제일 가운데 높이를 구하는 것
# =============================================================================

import scipy.stats as st

# 평균 60을 기준으로 95%에 해당하는 값을 구해보자. 간격에 대한 표준 편차는 4.
# 평균에 대한 높이와
# norm(): 정규분포 호출
n1=st.norm

# cdf(해당 좌표, 평균, 표준편차): 누적 확률. 맨 앞부터 특정 좌표까지의 누적된 확률
# pmf:
# pdf:
# ppf: 데드라인의 포인트

n1.ppf(0.95,70,5)

# 70에서 75 사이의 확률을 구하고 싶다면
n1.pdf(70, 75, 60, 4)

# 연속형은 pdf가 아니라 cdf를 사용한다.
n1.cdf(75, 60, 4) - n1.cdf(70, 60, 4)



n1.st.norm.stats(loc=60, scale=4, moments="mvsk")

n1.cdf(4)
# 상위 5프로에 대한 값
n1.ppf(0.95, 60, 4)
n1.ppf(0.025)*4+60
# ppf(확률): 해당 확률의 표준 정규 분포 상 좌표값 리턴

n2=st.binom
# 주사위 예제: 내가 관심갖는 숫자 3개, n의 개수 5, 확률 1/6
# 발생할 확률이 0.032인것
n2.pmf(3, 5, 1/6)





#=======시험========

import numpy as np
test=[5, 6, 4, 7, 7, 12, 8]
np.median(test)
np.mean(test)
np.mode(test)
# mode는 pandas에 있다.
import pandas as pd

# 강제적으로 dataframe이나 series 값으로 변환해야 한다.
# Series()로 강제적으로 한 열로 들어가게 한다.
pd.Series(test).mean()
a2=pd.Series(test)

a2.mean()
a2.median()
a2.mode() 

n1.cdf(3.5, 2.8, 0.5) - n1.cdf(3.3, 2.8, 0.5)
n2.pmf(2, 5, 0.4)

import scipy.stats as st

sleep=pd.read_csv("dataset/sleep.csv")
group_list=[1,2]
data1 = sleep['group'].isin(group_list)
id_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data2=sleep[data1].groupby(sleep['ID'])


result = pd.pivot_table(sleep[sleep['group'].isin(group_list)],
                          index='ID',
                          columns='group',
                          values='extra')

st.ttest_rel(result[1], result[2])
