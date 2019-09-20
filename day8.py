# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 08:38:41 2019

@author: user
"""

## 머신러닝 기법의 회귀

# (1) 데이터 분리
# train/test
# (2) 데이터 단위
# (3) 식을 도출(학습)
# (4) 검정(test)-> 추정값 나옴
# (5) 변수선택기법, 최적변수, 변수 축약
# (6) 검정

# =============================================================================
# 1. 데이터 로딩 및 탐색
# =============================================================================

import numpy as np
import pandas as pd

boston = pd.read_csv('dataset/boston.csv')

# 칼럼 내용 확인
boston.columns

# 각 데이터의 타입 확인
boston.dtypes

# boston 복제
boston1 = boston.copy()

# A라는 새로운 열에 A를 넣는다.
boston1["A"]="A"

# dataframe에 담긴 데이터 확인
boston1.head()

# A열의 값이 object 형태임을 알 수 있다.
boston1.dtypes

# float 타입인 데이터를 찾고 싶을 때
boston1.dtypes=='float64'

# object가 아닌 데이터를 찾고 싶을 때
boston1.dtypes=='object'
# -> 이걸 이용해 원하는 열만 불러올 수 있다.

# 수치형인 데이터만 찾아서 일괄적으로 데이터를 만든다.
boston1.loc[:,boston1.dtypes!="object"]

# 해당 데이터를 apply()을 이용해 다른 값으로 바꿀 수 있다.
boston1.loc[:,boston1.dtypes!="object"].apply()
# na가 들어있는걸 그 열의 평균값으로 바꾸는 법 알아두기

# count, mean, std, mmin, max 등을 한 번에 조회하는 법
boston1.describe()


# =============================================================================
# 2. 회귀분석 중 다중회귀로 분석하기 -> 입력 변수와 종속 변수 확인 필요
# =============================================================================

# boston의 집값을 구하려고 한다. medv가 집값을 나타낸다. = 종속 변수
# 처리를 쉽게 하려면 입력 변수와 종속 변수를 분리해 다른 변수에 담아 놓는다.

# 맨 처음엔 na 데이터가 없는지 확인 후 있다면 일괄적으로 처리하고 시작한다.
np.sum(boston.isna())   # na 데이터가 전혀 없음을 알 수 있다.
# na가 있다면 na가 없는 것과 있는 것을 분리해서 각각의 회귀 분석을 진행한다.

## 데이터 분리
# 종속변수를 제거한 데이터를 만든다.
xx = boston.drop(columns="medv")

# 종속 변수만 있는 데이터를 만든다
y = boston["medv"]


# =============================================================================
# 3. 이상치 존재 여부 확인
# =============================================================================

## 1) 회귀식을 구한 다음 그 모듈 안에서 이상치를 체크한다. -> statsmodels

## 2) 회귀식 없이 순수 데이터만 이용해 이상치를 체크한다. -> sklearn.neighbors 최근접이웃 분류기법(KNN)

from sklearn.neighbors import LocalOutlierFactor

# LocalOutlierFactor(n_neighbors=이웃의 숫자, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination="legacy", novelty=False, n_jobs=None)
# 이웃의 숫자가 너무 많으면 엉뚱한 값이 나오므로 적당히 낮은 걸로 한다.
lof1 = LocalOutlierFactor(n_neighbors=5)

# 실제 그 값을 찾으려면 fit()
# fit(수치형 데이터만 가능하다. 범주형은 불가능)
lof1.fit(xx)

# fit()을 한 뒤에는 변수명을 새로 만들지 않는다. 그 안에서 값을 만들고 가지고 있으라고 실행하는 것이기 때문이다.
# 이상치를 구한다.
lof1.negative_outlier_factor_ # -2보다 크면 정상치, 작으면 이상치이다.

# 이상치의 총 개수는 506개다.
len(lof1.negative_outlier_factor_)

# 정상치는 값을 행에 넣고 모든 값을 열로 넣는다.
xx1 = xx.loc[lof1.negative_outlier_factor_ > -2, :]

# 확인해보면 485로 줄어든 것을 볼 수 있다. 이상치를 날린 것이다.
xx1.shape

# y에도 동일하게 적용한다.
# 행이 하나이면 그냥 이렇게 해도 된다.
y1 = y[lof1.negative_outlier_factor_ > -2]
len(y1)

# =============================================================================
# na가 없는 수치형 데이터만 이용해서 이상치를 체크하고 na가 존재하면 na을 보정한후 이상치 유무를 체크한다.
# na를 어떻게 보정할지 생각해야 한다. 열 별 평균, 그룹별 평균...etc -> 그 후에 이상치를 체크한다.
# =============================================================================


# =============================================================================
# 4. 데이터 검정
# =============================================================================

# 데이터를 분리해주는 패키지
from sklearn.model_selection import train_test_split

# 트레이닝용/테스트용으로 나눠서 저장한다. test_size 하나만 지정해주면 나머지는 자동으로 설정된다.
# random_stat으로 결과가 동일하게 나오도록 설정한다.
tr_x, te_x, tr_y, te_y = train_test_split(xx1, y1, test_size=0.3, random_state=100)

type(tr_x)
tr_x.shape


# =============================================================================
# 5. 단위(scale) 표준화
# =============================================================================

# '수치형' 단위 표준화. 범주형은 불가능하다.
from sklearn.preprocessing import MinMaxScaler

# 일단 불러온다.
# MinMaxScaler(feature_range=(0, 1), copy=True)
minmax1 = MinMaxScaler()

# 트레이닝용 x의 값만 가져와서 chas를 제외한다.
# chas는 0 아니면 1이다. 0은 강이 없다, 1은 강이 있다는 뜻이다.
x_list = tr_x.columns.drop("chas")

# fit()을 실행한다. 이 또한 변수에 할당할 필요가 없다. 이미 보관하고 있는 것이니까.
# xx 데이터에서 x_list에 해당하는 값만 학습을 시킨다.
minmax1.fit(xx1[x_list])

# 스케일을 맞춘 tr_x라는 변수에 트레이닝용 x를 복사한다.
tr_xs = tr_x.copy()

# tr_xs의 모든 행 중 x_list 열(=chas 뺀 열) 자체를 변환해서 덮어쓴다.
# 그럼 똑같은 열에 값만 변경되어 넣게 된다.
tr_xs.loc[:, x_list] = minmax1.transform(tr_xs.loc[:, x_list])

# axis로 행방향으로 할 것인지 열방향으로 할 것인지 지정한다.
# np.min(tr_xs, axis=0)

# 테스트용에도 적용한다.
te_xs = te_x.copy()
te_xs.loc[:, x_list] = minmax1.transform(te_xs.loc[:, x_list])


# =============================================================================
# 6. 회귀분석적용
# =============================================================================

from sklearn.linear_model import LinearRegression

# 회귀분석을 불러와서
lm_model = LinearRegression()

# 트레이닝용 데이터를 적용한다.
lm_model.fit(tr_xs, tr_y)


# =============================================================================
# 7. 검정 및 성능 평가
# =============================================================================

# 아래의 두 값을 predict()가 알아서 계산한다.
lm_model.intercept_
lm_model.coef_

# 검정용 데이터를 넣어준다.
pred = lm_model.predict(te_xs)

# 테스트 데이터에 대한 r sqaure 값 확인
r_square = lm_model.score(te_xs, te_y)  # 0.765

# 성능 평가용 패키지
from sklearn.metrics import mean_squared_error

# 테스트 데이터 y의 실제값과 예측값을 넣는다.
mean_squared_error(te_y, pred)  # 16.971
# mean square가 작아져야 한다. 음수는 나올 수 없다.


# =============================================================================
# 8. 변수 선택 기법
# =============================================================================

# 제거 기법으로 변수를 추정하는 패키지. LinearRegression()을 쓴 변수가 있어야 쓸 수 있다.
from sklearn.feature_selection import RFE

# RFE(estimator, n_features_to_select=선택할 임의의 데이터 개수, step=1, verbose=0)
rfe1 = RFE(estimator=lm_model, n_features_to_select=4)

# 표준화된 트레이닝 데이터를 넣어준다.
rfe1.fit(tr_xs, tr_y)

# 선택한 4개 변수에만 true가 설정된다.
rfe1.get_support()

# 타입을 확인해보니 dataframe이다.
type(tr_xs)

# dataframe의 칼럼에 true 해당하는 이름만 뽑아온다.
tr_xs.columns.values[rfe1.get_support()]  # ['rm', 'dis', 'ptratio', 'lstat']
# 이 4개를 가지고 회귀 분석을 돌려본다.

# 트레이닝용 x 변수를 넣어준다. 
tr_xs4 = rfe1.transform(tr_xs)

# np이기 때문에 아래의 형식으로 4개의 데이터를 확인한다.
tr_xs4[0:3, :]
tr_xs4.shape

# 4개 변수만 담긴 데이터로 다시 회귀분석을 한다.
lm_model2 = LinearRegression()
lm_model2.fit(tr_xs4, tr_y)

# 다시 predict를 하기 전에 te_xs도 변수 수가 4개가 되어야 하므로 변환해준다.
te_xs4 = rfe1.transform(te_xs)

# predict를 실행한다.
pred2 = lm_model2.predict(te_xs4)

# 변수 4개만 가지고도 13개일 때의 성능과 유사하게 나온다.
lm_model2.score(te_xs4, te_y) # 0.757 (75.7%)

# MSE도 올라간다.
mean_squared_error(te_y, pred2) # 17.5

# 참고: 변수값이 많아질 수록 r square와 MSE가 올라간다. = 성능이 올라간다.
# RFE()에 변수 개수를 입력하지 않고 진행하면 최적의 변수 개수를 찾아준다.


# =============================================================================
# 9. 변수 축약 기법
# =============================================================================

from sklearn.decomposition import PCA

# 5개를 가지고 변수 축약 기법을 실행한다.
pca1 = PCA(n_components=5)
# pca1 = PCA(n_components=8)

# tr_xs에 적용한다.
pca1.fit(tr_xs)

# 확인하면 람다와 일치하는 벡터값이 5개의 입력 변수 수만큼 들어가있다.
pca1.components_
# [[ 0.14058577, -0.20666103,  0.33643462,  0.01211327,  0.30343206,
#         -0.07883427,  0.33377618, -0.21349436,  0.49618136,  0.4693754 ,
#          0.17313515, -0.16840888,  0.20993595], 
# -> 이 한 칸에 여러 벡터가 있다. 이때 사용된 데이터는 공분산이다. 즉, 각 데이터의 관계성 위주로 구했다는 거다. 숫자가 큰 값끼리 유사하다는 의미를 내포한다.(같은 성향)
# ex) 0.49618136,  0.4693754 이 둘이 유사한 성향을 가진다.

#        [-0.09551555, -0.33514122,  0.16243567,  0.44187529,  0.23760956,
#         -0.0118904 ,  0.45261549, -0.25709321, -0.41093552, -0.33420413,
#         -0.19285242,  0.08558777,  0.06544393],
#        [ 0.02438982,  0.28990848, -0.05739819,  0.82193128, -0.00673945,
#          0.14435316, -0.18625513,  0.07335743,  0.27223745,  0.18806982,
#         -0.19960751, -0.01292249, -0.16137986],
#        [ 0.04055875,  0.37616096, -0.02065537, -0.29341662,  0.2961007 ,
#          0.13964543,  0.14698234, -0.06483821, -0.03553329,  0.04748219,
#         -0.72242581, -0.33371477,  0.02391495],
#        [ 0.05922173,  0.07981422,  0.06373453,  0.19173286, -0.09790852,
#         -0.27057212, -0.12016446,  0.17854261, -0.20542092, -0.14759969,
#          0.19318949, -0.75240742,  0.39085043]]

# 해당 데이터를 dataframe으로 변경 후 엑셀로 export 하기
pd.DataFrame(pca1.components_).to_excel("dataset/decom1.xlsx")

# 각각의 분산값
pca1.explained_variance_
# [0.43637962, 0.10747217, 0.07678906, 0.04844739, 0.03404881]
# 이 다섯개의 변수를 가지고는 70%도 안된다. 그래서 전체 100프로 중 작은 데이터만 설명하고 있다. PCA(n_components=5) 수를 올려주면 score가 올라간다.
# 누적했을 때 90% 되는 시점을 components 개수로 정해주면 설명력을 가진다.

# 다시 트레이닝용 x를 축약 기법에 적용하기 위해 변환한다.
tr_xs5 = pca1.transform(tr_xs)

tr_xs.shape   # 13개의 변수가 있을 때
tr_xs5.shape  # 5개의 변수가 있을 때

# 선형 회귀
lm_model3 = LinearRegression()

# 실행
lm_model3.fit(tr_xs5, tr_y)

# tr 만든 것과 같은 방식으로 변환
te_xs5 = pca1.transform(te_xs)

# 축약시킨 변수를 넣어준다.
pred3 = lm_model3.predict(te_xs5)

lm_model3.score(te_xs5, te_y)   # 0.570 (57%) 변수 선택 기법보다 작다.
# 입력값 간의 관계성이 있다면 축약 기법이 좋다.

mean_squared_error(te_y, pred3)   # 17.646


# =============================================================================
# chas: 범주형 데이터. 회귀는 무조건 수치형이라는 고정관념을 깨야 한다.
# =============================================================================

# chas: 0, 1 or 1, 2
# y = a + 0.6*x1 + 0,3*x2 + 0.1*chas
# 일 때 chas가 0이면 0.1*chas 앞의 식이 y를 형성하는 값이다.
# chas가 1이라면 앞의 식에 0.1만 더해서 넘겨준다. 이때 0.1은 y에 영향을 주는 가중치가 된다.
# chas가 2가 되면 0.1을 한 번 더 더해준 효과가 난다. chas가 바뀔 때마다 0.1의 비율만큼 넘겨주는 형태가 된다. 그룹의 특징이 반영되는 것이 아니라 0.1의 배수만큼 일정한 비율로 넘기는 것이다.
# 하지만 chas가 0, 1, 2, 3...이라고 해서 실제 0배, 1배, 2배, 3배의 의미가 아니라 분별하기 위한 용도일 뿐이다. 
# 만약 그룹의 특징을 넣어서 해당 변수가 들어갔을 때의 변동량을 구하는 식을 구하고 싶다면? 얘가 들어감으로 인해서 변동되는 가중치를 구해야 한다.
# -> One-Hot 기법

## One-Hot
# label 수(0, 1, 2, 3...총 4개) 대로 변수를 만든다. d1=0, d2=1, d3=2, d4=3
# 특정 변수에 1을 준다면 나머지는 모두 0으로 처리한다. d2에 1을 주면 d1, d3, d4는 0이 된다.
# 각각의 가중치가 0.5, 0.2, 0.1, 0.1 이라면 0.5*0 + 0.2*1 + 0.1*0 + 0.1*0 이다.
# 즉, 1이 들어간 변수를 제외하고는 모두 반영이 되지 않는다.


# =============================================================================
# dummy
# =============================================================================

# One-Hot에서 첫번째 열이 사라진 형태. 0.2*1 + 0.1*0 + 0.1*0

survey = pd.read_csv("dataset/survey.csv")
survey.dtypes
# 데이터 타입 변경
survey.Sex = survey.Sex.astype("category")
# category 타입으로 변경됨
survey.dtypes

# one-hot 기법
survey2 = pd.get_dummies(survey)
# dummy 기법
survey3 = pd.get_dummies(survey, drop_first=True)


## 회귀식 만들기
survey.columns.values
x_list2 = ['Wr.Hnd', 'NW.Hnd', 'Exer', 'Pulse']

survey4 = pd.get_dummies(survey[x_list2], drop_first=True)

# na값이 있으므로 dropna()로 보정
survey5 = survey4.dropna()

lm_model_dum = LinearRegression()
# 모든 행에 대해 0번부터 3번 전까지 Pulse 데이터 적용
lm_model_dum.fit(survey5.drop(columns="Pulse"), survey5["Pulse"])

# coef 값 출력
lm_model_dum.coef_  # [ 1.44220748, -1.32426954,  4.33425855,  4.44479091]
# intercept 값 출력
lm_model_dum.intercept_   # 69.65 -> group 0이 포함된 값
# 3번 그룹이 되는 순간 4.33을 더해주는 것이다.
# pulse = 69.65 + 1.44*Wr.Hnd + -1.32*NW.Hnd + 4.33*Exer1 + 4.44*Exer2 일때
# group 0: 69.65 + 1.44*Wr.Hnd + -1.32*NW.Hnd
# group 1: 69.65 + 1.44*Wr.Hnd + -1.32*NW.Hnd + 4.33
# group 2: 69.65 + 1.44*Wr.Hnd + -1.32*NW.Hnd + 4.44


# =============================================================================
# 분류 기법
# =============================================================================

Sonar = pd.read_csv("dataset/Sonar.csv")
Sonar.head() # Class 열이 실제 y값이다. 암석인지 강석인지 60개의 센서가 체크한 값.
Sonar.shape
Sonar.columns

# x와 y값 설정 
xx = Sonar.drop(columns="Class")
yy= Sonar["Class"]

# 그룹별로 골고루 집어넣음 
tr_x, te_x, tr_y, te_y = train_test_split(xx, yy, test_size=0.3, random_state=200)

# KNN 패키지
from sklearn.neighbors import KNeighborsClassifier

# 이웃의 개수는 기본적으로 5개 되어있는데 분류기준이 명확하지 않으면 낮추는 것이 좋다.
knn1 = KNeighborsClassifier(n_neighbors=3)

# x는 이거고 y는 이거라고 지정하는 의미밖에 없음
knn1.fit(tr_x, tr_y)

# 그래서 test값으로 predict를 이용하면
pred5 = knn1.predict_proba(te_x)
#[[0.66666667, 0.33333333], -> 0.33이 두번 들어가서 0.66
#       [1.        , 0.        ],
#       [0.66666667, 0.33333333],
#       [0.        , 1.        ],
#       [0.        , 1.        ],

knn1.score(te_x, te_y)  # 0.82(82%)


# =============================================================================
# 의사 결정 나무(Decision Tree): 중심값을 기준으로 작은 값/큰 값 가지를 친다.
# 순수도: 이 조건을 따라 나누었을 때 기준이 될 수 있는 값을 찾는 것.
# 수치형이면 평균을 기준으로 크다 작다를 나눈다.
# 범주형이면 abc/bac/cab 등으로 그룹을 나누고 이 값을 기준으로 M과 R이 몇 개가 들어가있는지 개수를 센다. 내가 원하는 label, group으로 찾아주는 개념.
# =============================================================================

## Random Forest
# 이미지 처리에 성능이 월등하다. decision tree의 확장형.

from sklearn.tree import DecisionTreeClassifier

dt1 = DecisionTreeClassifier()
dt1.fit(tr_x, tr_y)

pred6=dt1.predict(te_x)
dt1.score(te_x, te_y)   # 0.841

# 그래프로 보기
from sklearn.tree import export_graphviz

export_graphviz(dt1, "dataset/tree1.dot")

# RandomForest
from sklearn.ensemble import RandomForestClassifier

# n_estimators를 올리면 score도 올라간다. 즉, 행의 수가 많을 수록 효과를 본다.
rf = RandomForestClassifier(n_estimators=50)
rf.fit(tr_x, tr_y)

rf.score(te_x, te_y)  # 0.857, n_estimators=50일땐 0.873

# 해당 인덱스의 decisionTree 값 접근
rf.estimators_[4]


# =============================================================================
# 계층적 군집 분석
# =============================================================================

# 1. 거리(최단: single, 최장, 중앙값, 평균...etc)
# 2. 연결 방법(거리를 나타내는 대표값에 따라 달라진다.)

# 데이터 수가 많으면 한 눈에 볼 수가 없다. 마지막에 나오는 숫자만 보고 판단해야 한다.


# =============================================================================
# 비계층적 군집 분석(k-means)
# =============================================================================

# 1. 군집수를 먼저 지정한다.
# 2. 난수에 의해 군집의 중심점이 구해진다.
# 3. 내가 가진 데이터와 중심의 거리를 구한다.
# 4. 가장 가까운 포인트들만 묶어 군집의 번호를 할당한다.
# 5. 모인 데이터로 평균을 구한다.
# 6. 처음의 난수보다 위치가 이동된다.
# 7. 다시 재할당 한다.
# 8. 위치 변화가 일어나지 않을 때까지 반복한다.
# 9. 최적화 되면(최적의 거리를 구할때까지) 동일한 결과가 나온다.

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3)

# x값만 넣어주면 유사한 성향끼리 묶어준다. 비지도학습이라 y가 없다.
km.fit_predict(tr_x)