# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:30:57 2019

@author: HP27
"""


# =============================================================================
# 3개 이상 집단의 차이 검정
# 분산분석(anova)
# H0: mu_a = mu_b = mu_c = mu_d = 0(기준값이 0라는 뜻)
# H1: 적어도 한 그룹의 mu는 0이 아니다.
# -> 모두 같은지 아니면 하나라도 다른 그룹이 존재하는지 그 유무만 파악한다.
# 다중비교, 사후분석: 두 개씩 묶어서 어떤 게 같고 어떤 게 다른지 알아내는 것. mu_a = mu_b, mu_a != mu_b / mu_b = mu_c, mu_b != mu_c, ...)

# 각 그룹 내에서의 분산이 같다는, 즉 등분산이라는 조건에서 해를 찾게 된다.
# 분산 개념으로 접근해서 평균들 끼리 차이가 있는지 파악하도록 설계한다.

# 일일이 x와 y가 뭔지 지정하는 방법
import statsmodels.api as sm
# 식으로 넘겨주는 방법
from statsmodels.formula.api import ols

moore = sm.datasets.get_rdataset("Moore", "carData", cache=True) # load
data = moore.data

# 특정 칼럼의 이름 변경
data = data.rename(columns={"partner.status" :"partner_status"}) # make name pythonic

# 그룹 변수는 총 3개이고, 데이터는 15개씩 들어가있다.
data.fcategory.value_counts()
  
# ols()로 회귀분석 트릭을 써서 분산분석을 한다. 각각 제곱합을 구해야 서로 차이가 있는지 없는지 판별할 수 있는데 그걸 구하도록 만든 트릭이다. 그래서 anava_lm()을 하기 전에 써야 한다.
# 실제 관측된 값을 ~ 모양 왼쪽에, 
# C: 카테고리컬 = 범주형 데이터니까 그룹단위로 인식하세요.
# *: 각각의 변수를 더한 다음에 join한 것을 식으로 표현해주세요.
  
# fit(): 이 데이터를 가지고 값을 찾아주세요. 최종적으로 fit()을 해줘야 결과가 나온다.
# ols('실제 비교하f려는 데이터 즉,b에 있는 그룹 별로 나온 값 conformity ~ 실제 어떤 그룹으로 이루어져 있는지 한 열로 묶은 변수 fcategory, 더 꾸며주고 싶은 글자 sum)
  
# one way
moore_lm = ols('conformity ~ fcategory',
                data=data).fit()
# two way
moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                data=data).fit()

# anova에 숨어있는 attributes와 methods 살펴보기
moore_lm._wrap_attrs
moore_lm._wrap_methods

table1 = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 Anova DataFrame
table2 = sm.stats.anova_lm(moore_lm, typ=2)
print(table1)
print(table2)

# 자유도: n-1, 정해진 평균값을 맞추기 위한 데이터 1개를 제외한 모든 데이터.
# mean square(MS)
# SSR/df = MSR. 그룹간 자유도로 나눈 것
# SSE(=residual)/df = MSE. 그룹 내의 자유도로 나눈 것
# F = MSR / MSE
# =============================================================================


# 완전 확률화: A,B / B,C / C,A 묶어서 비교해야할 때 array 구조로 만들면 서로 데이터 크기가 같아야만 한다. 