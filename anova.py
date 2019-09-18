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

import statsmodels.api as sm
from statsmodels.formula.api import ols

moore = sm.datasets.get_rdataset("Moore", "carData", cache=True) # load
data = moore.data
data = data.rename(columns={"partner.status" : "partner_status"}) # make name pythonic

moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)', data=data).fit()
table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 Anova DataFrame
print(table)
# =============================================================================
