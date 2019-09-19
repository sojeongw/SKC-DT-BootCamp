# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 00:39:39 2019

@author: Mac
"""
##  분산분석 실습 예제 모음

import scipy.stats as stats
import pandas as pd
import urllib
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import numpy as np

## 분산분석을 위한 분산의 특징 확인을 위한 데이터 시각화
centers = [5,5.3,4.5]
std = 0.1
colors = 'brg'

# 이렇게 집단 내의 분산 정도가 작고 집단 간 분산이 커야 좋다.
data_1 = []
for i in range(3):
    data_1.append(stats.norm(centers[i], std).rvs(100))
    plt.plot(np.arange(len(data_1[i]))+i*len(data_1[0]),data_1[i], '.', color = colors[i])

# 하지만 이런 상황이 생길 수 있다. 
std_2 = 2
data_2 = []
for i in range(3):
    data_2.append(stats.norm(centers[i], std_2).rvs(100))
    plt.plot(np.arange(len(data_1[i]))+i*len(data_2[0]), data_2[i], '.', color = colors[i])
    
# 분산분석은 각각에 대한 흩어진 정도의 제곱 합을 자유도로 나눠서 평균을 구하면 분산이 되고, 이 분산의 비를 구하면 F가 되는데 F가 같은지 아닌지를 판단한다. 멀리 떨어져 있다 아니다 즉, 차이가 난다 아니다 정도의 판단밖에 하지 못한다. 그래서 사후 분석을 이용하게 된다.

# 단일 검정과 달리 비포 애프터, 성능 차이, 동일한 걸 비교하고 있는지 등을 보려면 두 집단간의 차이 검정을 이용한다. 이질적인 성향이 있다면 이게 좋은건지 나쁜건지 확인할 수 있어야 한다. 품질 검증에서는 무조건 동일하게 나와야 이상적인 것이다. 만약 성능이 뭐가 더 좋은지 알고 싶으면 서로 이질적이어야 좋다. 내가 분석하고자 하는 목적에 따라 달라진다.

# =============================================================================
# 예시 데이터(Altman 910)
#
# 22명의 심장 우회 수술을 받은 환자를 다음의 3가지 그룹으로 나눔
# 
# Group I: 50% 아산화 질소(nitrous oxide)와 50%의 산소(oxygen) 혼합물을 24시간 동안 흡입한 환자
# Group II: 50% 아산화 질소와 50% 산소 혼합물을 수술 받는 동안만 흡입한 환자
# Group III: 아산화 질소 없이 오직 35-50%의 산소만 24시간동안 처리한 환자
# => 적혈구의 엽산 수치를 24시간 이후에 측정
# =============================================================================

# url로 데이터 얻어오기
url = 'https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/altman_910.txt'
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')

# Sort them into groups, according to column 1
group1 = data[data[:,1]==1,0]
group2 = data[data[:,1]==2,0]
group3 = data[data[:,1]==3,0]

# matplotlib plotting
plot_data = [group1, group2, group3]
ax = plt.boxplot(plot_data)
plt.show()

# -> 평균이 어떤게 높고 중앙값이 어떤지 확인이 쉽게 가능하다.
# 중앙값만 봤을 때는 1이 제일 높아보인다. 하지만 그룹 내의 분산이 너무 크다.
# 집단 내의 분산이 너무 크면 어떤게 좋은지 판단이 힘드므로 서로 같은지 아닌지 부터 알아봐야 한다.


# =============================================================================
## Boxplot 특징
# 평균값의 차이가 실제로 의미가 있는 차이인지, 
# 분산이 커서 그런것인지 애매한 상황
# =============================================================================


# 원래 분산분석은 서로 분산이 같다고 생각하고 계산한다. 하지만 이럴 경우 무시한 부분이 묻혀버리게 된다. 그냥 자체적으로 처리해버린다.
# 일원분산분석에는 두 가지 방법이 있다.


# =============================================================================
# scipy.stats으로 일원분산분석
# =============================================================================

# 그룹이 최소 세 개 이상 필요하다. scipy는 각 그룹별로 따로따로 분리하는 반면 statsmodel은 한 열로 처리한다.
# f_value[0] 은 F, [1]은 p-value
f_value = stats.f_oneway(group1, group2, group3)  
# 각각의 값을 개별 분리해서 저장하는 법
F_statistic, pVal = stats.f_oneway(group1, group2, group3)

# 연산 수치가 작은 게 좋은지 큰 게 좋은지 알아야 한다. 작은 게 좋은 거라고 가정하자. 가장 작은 쪽을 찾아서 그게 좋다고 말해야 한다. 만약 이질적인 그룹이 큰쪽이라면 이 그룹은 위험한 것이니 그 그룹 제품은 사용하면 안되는 것이다. 문제는 해석이다!

print('Altman 910 데이터의 일원분산분석 결과 : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))
if pVal < 0.05:
    print('P-value 값이 충분히 작음으로 인해 그룹의 평균값이 통계적으로 유의미한 차이가 있음')

# =============================================================================
# statsmodel을 사용한 일원분산분석: 전처리 없이 사용 가능
# =============================================================================

# 경고 메세지 무시하기
import warnings
warnings.filterwarnings('ignore')

# 판다스로 읽으면 바로 dataframe으로 들어오는데 이걸 변환해서 사용할 것이다.
df = pd.DataFrame(data, columns=['value', 'treatment'])    

# the "C" indicates categorical data
# ols를 사용하면 각각의 열이 그룹변수인지 데이터인지 위치를 가지고 분별할 수 있다.
# 데이터 ~ C(그룹변수)
model = ols('value ~ C(treatment)', df).fit()   # fit()까지 써줘야 작동된다.

# 결과 출력
print(anova_lm(model))
# df(자유도) / sum_sq = mean_sq
# F: F값
# PR: P-Value
# residual: 흩어진 정도에 따른 정보


# =============================================================================
# 이원분산분석(two-way ANOVA)
# => 독립변인의 수가 두 개 이상일 때 집단 간 차이가 유의한지를 검증
# 
# 상호작용효과(Interaction effect)
# => 한 변수의 변화가 결과에 미치는 영향이 다른 변수의 수준에 따라 달라지는지를 확인하기 위해 사용
# 
# 예제 데이터(altman_12_6) 설명
# => 태아의 머리 둘레 측정 데이터
# => 4명의 관측자가 3명의 태아를 대상으로 측정
# => 이를 통해서 초음파로 태아의 머리 둘레측정 데이터가 
#    재현성이 있는지를 조사
# =============================================================================

inFile = 'altman_12_6.txt'
url_base = 'https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/'
url = url_base + inFile
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')

# dataframe-format로 데이터셋 가져오기
df = pd.DataFrame(data, columns=['head_size', 'fetus', 'observer'])
# df.tail()

# 태아별 머리 둘레 plot
df.boxplot(column = 'head_size', by='fetus' , grid = False)

# =============================================================================
# #그림 결과 설명
# #태아(fetus) 3명의 머리 둘레는 차이가 있어보임
# 이것이 관측자와 상호작용이 있는것인지 분석을 통해 확인 필요
# =============================================================================

# =============================================================================
# 분산분석으로 상호(상관, 교호)관계 파악
# =============================================================================
formula = 'head_size ~ C(fetus) + C(observer) + C(fetus):C(observer)'
lm = ols(formula, df).fit()
print(anova_lm(lm))

# =============================================================================
# 결과 설명
# P-value 가 0.05 이상. 따라서 귀무가설을 기각할 수 없음
# 측정자와 태아의 머리둘레값에는 연관성이 없다고 할 수 있음
# 측정하는 사람이 달라도 머리 둘레값은 일정하는 의미
# 
# 결론적으로 초음파로 측정하는 태아의 머리둘레값은 믿을 수 
# 있다는 의미
# =============================================================================

# =============================================================================
# 사후분석(Post Hoc Analysis)
# =============================================================================

import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np

# url로 데이터 얻어오기
url = 'https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/altman_910.txt'
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')

# 그룹 단위로 불러오기
group1 = data[data[:,1]==1,0]
group2 = data[data[:,1]==2,0]
group3 = data[data[:,1]==3,0]

# pandas로 데이터 불러오기
df = pd.DataFrame(data,columns=['value', 'treatment']).set_index('treatment')

# 예시 데이터 시각화 하기
plot_data = [group1, group2, group3]
ax = plt.boxplot(plot_data)
plt.show()


df.head()

df2 = df.reset_index()
df2.head()

# 다중 비교: 그룹 간 얼마나 차이가 나고, 어떤 집단이 좋은지 판단하는 방법 
# pairwise: 데이터를 변환하지 않아도 쌍으로 맞춰서 비교를 해준다.
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df

# 다중 비교의 여러 방식 중 튜키 방식을 진행한다.
# pairwise_tukeyhsd(y값, 열의 변수들, 유의수준)
posthoc = pairwise_tukeyhsd(df['value'], df['treatment'], alpha=0.05)
print(posthoc)
# 나올 수 있는 짝의 모든 경우의 수대로 데이터 출력
# mean diff: 1에서 2를 뺀 값
# p-adj: 0.05보다 크면 두 집단의 차이가 나지 않는다고 판단
# reject: 귀무가설 기각 여부
# 하지만 이 결과에선 평균값이 나오지 않는다. 그래서 평균값을 다시 구해야 한다. 평균이 클 수록 한 쪽이 쏠려 있는 것이므로 이게 좋은 건지 나쁜 건지 판단해야 한다.

fig = posthoc.plot_simultaneous()



