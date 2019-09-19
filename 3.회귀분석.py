# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:06:45 2019
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import statsmodels.regression.linear_model as lm



# 1번 문제 
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')

# column 이름 지정
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# head(int) 숫자만큼의 row를 화면에 뿌려주세요
df.head()

sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

# =============================================================================
# ~ 안 쓰고 dataframe 통째로 열 단위로 계산하는 법
# MEDV를 제외한 모든 데이터를 담는다.
xx=df.drop(columns="MEDV")
y=df["MEDV"]
lm_model0=lm.OLS(y, xx).fit()
# attribute 확인 후
lm_model0._wrap_attrs
# 호출한다.
lm_model0.pvalues
lm_model0.summary()

# =======================================================================================
# Dep. Variable:                   MEDV   R-squared (uncentered):                   0.959
# Model:                            OLS   Adj. R-squared (uncentered):              0.958
# Method:                 Least Squares   F-statistic:                              891.3
# Date:                Thu, 19 Sep 2019   Prob (F-statistic):                        0.00
# Time:                        15:52:20   Log-Likelihood:                         -1523.8
# No. Observations:                 506   AIC:                                      3074.
# Df Residuals:                     493   BIC:                                      3128.
# Df Model:                          13                                                  
# Covariance Type:            nonrobust                                                  
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# CRIM          -0.0929      0.034     -2.699      0.007      -0.161      -0.025
# ZN             0.0487      0.014      3.382      0.001       0.020       0.077
# INDUS         -0.0041      0.064     -0.063      0.950      -0.131       0.123
# CHAS           2.8540      0.904      3.157      0.002       1.078       4.630
# NOX           -2.8684      3.359     -0.854      0.394      -9.468       3.731
# RM             5.9281      0.309     19.178      0.000       5.321       6.535
# AGE           -0.0073      0.014     -0.526      0.599      -0.034       0.020
# DIS           -0.9685      0.196     -4.951      0.000      -1.353      -0.584
# RAD            0.1712      0.067      2.564      0.011       0.040       0.302
# TAX           -0.0094      0.004     -2.395      0.017      -0.017      -0.002
# PTRATIO       -0.3922      0.110     -3.570      0.000      -0.608      -0.176
# B              0.0149      0.003      5.528      0.000       0.010       0.020
# LSTAT         -0.4163      0.051     -8.197      0.000      -0.516      -0.317
# ==============================================================================
# Omnibus:                      204.082   Durbin-Watson:                   0.999
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1374.225
# Skew:                           1.609   Prob(JB):                    3.90e-299
# Kurtosis:                      10.404   Cond. No.                     8.50e+03
# =============================================================================
# p-value가 0.05보다 작아야 의미가 있다. 0.05보다 큰 것들은 집값에 영향이 없는 변수이므로 제외한다. ex. INDUS, AGE

# 원래의 AIC
lm_model0.summary()   # 3074, 3128

# 영향을 미치지 않는 변수를 지웠을 때 AIC, BIC의 변화 확인
xx1=xx.drop(columns="INDUS")
lm_model1=lm.OLS(y, xx1).fit()
# AIC 확인
lm_model1.summary()   # 3071, 3122
# 작아졌으므로 INDUS를 빼는 게 낫다는 결론

xx2=xx.drop(columns="AGE")
lm_model2=lm.OLS(y, xx2).fit()
lm_model2.summary()   # 3072, 3123
# 역시 줄어들었으므로 AGE를 빼는게 좋다.

# 이렇게 하나하나 구하는 대신 scikit을 사용하면 한 번에 가능하다.
# 하지만 skikit은 회귀계수만 보여주기 때문에 상세한 정보를 하나하나 보고싶다면 위처럼 하면 된다.
# =============================================================================





sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
# plt.savefig('./figures/scatter.png', dpi=300)
plt.show()

# 관계를 수치로 바꾼 것
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)

# 각 칸에 상관계수를 출력함
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
plt.show()

# 오리지널로 리셋
sns.reset_orig()

# 데이터 셋을 df에 넣는다
lm_model=ols('MEDV~RM', df).fit()
lm_model.params
lm_model.summary()
# AIC, BIC가 작을 수록 좋다. 이 값으로 인해 잔차가 줄어들 수 있다는 의미다. 변수가 어떤 변동이 있는지 판별하는 단위다.
# intercept: 알파값
# RM: 베타값
# std err: coef 값에 해당되는 표준 오차
# t: 알파에 대한 표준편차
# p: p-value. 0.05보다 크면 귀무가설 성립. 여기서 귀무가설은 알파나 베타가 0이라는 것. 즉, 귀무가설이 성립하면 = 알파와 베타가 0이면 그들의 의미가 없다는 것.

# 이 안에 있는 값을 호출해야 한다.
lm_model._attrs
lm_model._wrap_attrs
# 그 중에 내가 원하는 항목만 이렇게 뽑아서 볼 수 있다.
lm_model.pvalues
lm_model.fittedvalues

# anova_lm으로 model 호출
anova_lm(lm_model)
sm.stats.anova_lm(lm_model)


# RM: 가구당 평균 방의 개수
# MEDV
X = df[['RM']].values
y = df['MEDV'].values

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)



# =============================================================================
# 머신러닝 기법으로 회귀분석하기

# 1. 데이터 분리
# 1) training용 데이터
# 2) test용 데이터

# 2. 데이터 단위 통일

# 3. 식 도출(학습)

# 4. 새로운 test값을 넣어 검정 -> 학습 데이터에 넣은 추정값이 도출됨

# 5. 검정 결과가 나오면 이걸 기반으로 변수 선택 기법 후 최적 변수를 찾거나 변수 축약 기법을 사용한다.

# 6. 다시 검정을 해서 이전보다 얼마나 좋아졌는지, 성능이 유사한지 확인한다. 변수는 줄었는데 성능이 유지되어야 좋으므로.
# =============================================================================

# 1. 데이터 분리
# 데이터 확인
df.shape

# sciki learn 가져오기
from sklearn.model_selection import train_test_split

# (x, y, test size 지정(전체 데이터 중 몇프로를 가져올 것인지), random state로 모양을 맞춰줌(1234라는 규칙으로))
# 30프로의 데이터를 추출하게 된다. 보통 트레이닝과 테스트용을 7:3으로 나눈다.
# xx와 y의 트레이닝용, test용 각각이 만들어지므로 총 4개가 나오니까 4개의 변수에 담는다.
tr_x, te_x, tr_y, te_y=train_test_split(xx, y, test_size=0.3, random_state=1234)

# xx 데이터 506개 중에서
xx.shape
# 354만큼의 tr_x에 왔음
tr_x.shape
# 트레이닝용과 테스트용을 합치면 다시 원본 506개가 나온다는 걸 알 수 있다.
tr_x.shape[0]+te_x.shape[0]

# 2. 데이터 단위 통일
# 가끔 dataframe이 아니라 array로 오는 경우도 있어서 type을 확인한다.
type(tr_x)

# 산술평균을 확인해서 각각의 단위를 확인한다. -> 0~1 사이로 통일해야 함.
tr_x.describe()

from sklearn.preprocessing import minmax_scale, MinMaxScaler

# 단위 통일을 위한 클래스 호출
minmax=MinMaxScaler()

# 데이터를 알맞게 변환
tr_xs=minmax.fit_transform(tr_x)
type(tr_xs)   # numpy.ndarray

# 최소가 0, 최대가 1이 맞는지 확인
np.min(tr_xs, axis=0)

# 검증용에도 마찬가지로 적용한다. fit은 단위를 찾아오는 기능인데 앞에서 이미 했으므로 transform으로 변환만 하면 된다.
te_xs=minmax.transform(te_x)
np.min(te_xs, axis=0)


from sklearn.linear_model import LinearRegression

# 식을 찾을 장비를 준다.
lm_mod=LinearRegression()

# 3. 식 도출
# stats와 x, y 순서를 반대로 해야 함.
# 어떤 변수가 더 중요한지를 함께 확인하기 위해 변수를 표준화 해온 것이다.
lm_mod.fit(tr_xs, tr_y)

# 상수값
lm_mod.intercept_
# 회귀계수
lm_mod.coef_

# 4. 테스트 데이터로 검정
# 표준화된 테스트 데이터를 넣었을 때 아까 만든 lm_mod에 넣은 회귀식에 따라 예측값이 나온다. 즉 y hat 값이다.
pred1=lm_mod.predict(te_xs)

# 각각의 편차 도출 
te_y-pred1

# 편차를 제곱한 값의 평균
np.mean((te_y-pred1)**2)

# 5. 변수 제거
# te_xs와 te_y에 대한 R sqaure 값. 흩어진 정도. MSE는 작을 수록 좋지만 얘는 1에 가까울 수록 즉, 크면 클 수록 좋다. 이걸 반복하면서 어떤 게 좋은지 확인한다.
lm_mod.score(te_xs, te_y)

# lm_mod.coef_ 중 가장 큰 값의 위치를 불러온다.
np.argmax(lm_mod.coef_)   # 5

# 그 위치의 변수명을 알 수 있다.
te_x.columns.values[np.argmax(lm_mod.coef_)]
# 따라서 가장 중요한, 영향력 있는 변수는 'RM'(가구당 평균 방의 개수)


