# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:39:59 2019

@author: user
"""

# =============================================================================
# 산술평균(mean)은 이상치에 민감하지 않아서 데이터가 흩어진 정도에 집중한다.
# 중앙값(median)은 이상치에 크게 영향 받지 않는다.
# 따라서 산술평균 중앙값 중 선택할 때는 데이터의 특징을 고려해야 한다.
# 하지만 이상치를 이상치라고 생각하지 못하기 때문에 산술평균을 많이 사용한다.
# 최근에는 이상치를 분별하는 방법론이 나타나고 있다.
# 최빈수(mode)는 다양한 의사 결정이 나왔을 때 많이 나온 응답을 고려할때 사용한다.

# 앙상블 기법: 여러 개를 섞어서 해를 찾는 기법. 각각의 강점을 알고 있어야 내 데이터에 맞게 적절한 알고리즘을 적용할 수 있다.

# 데이터의 흩어진 정도, 중심값으로부터 데이터가 얼마나 떨어져 있는지 나타냄.
# 바이어스
# 빈도수는 μ(중심값)에 f를, 중심값은 m을 붙여 사용한다.

# 편차(deviation): 중심으로부터 흩어진 정도. (Xi - μ)
# 1, 3, 5의 값이 있다고 하면 이것들이 Xi다. 각각의 값을 더해 갯수로 나누는 것은 1/3*1 + 1/3*3 + 1/3*5로 나타낼 수 있다. 즉, Pi라는 확률에 Xi만큼 곱해서 합한 것이다. (∑PiXi)
# 각 Xi에 평균 3을 빼면 -2, 0, 2이고 합하면 0이 된다. 모두 동일한 비중 1/3로 구한 값이기 때문에 흩어진 값을 더하면 항상 0이 되는 것이다.
# 하지만 -2, 0, 2에 절대치를 적용하거나 제곱을 하면 이 문제를 해결할 수 있다.

# 분산(mse): 평균에서 흩어진 정도. 흩어진 정도에 제곱해서 n으로 나눈 값.  (∑(Xi-μ)^2 / N) Xi가 많을 수록 커지며 절대 음수가 나올 수 없다. 흩어진 정도를 표준화 시킨 단위. 제곱했기 때문에 정확하지 않다.
# sum of square(SS): 제곱하고 합한 값 -> 이 값을 발견하면 샘플의 값을 따져야 한다. 샘플의 값을 고려하지 않고 값이 크다 작다 판단하면 안된다.

# 표준편차(rmse): 데이터가 흩어진 정도. 분산에 루트를 씌워 정확도를 높인 값. (루트 ∑(X-μ)^2 / N)

# 가장 좋은건 실제값 y와 예측한 ^y의 차이가 0인 것이다. 하지만 ^y는 실제 다양하게 나올 수 밖에 없다. y와의 차이, 즉 에러값(SSE)이 작을수록 좋다. ∑(y-^y)^2

# 편향(bias): 흩어진 데이터를 봤을 때 데이터가 한 쪽으로 치우치는 현상
# 편차와 편향을 합해 분산(variance)라고 한다.

# 예시
# 1시간 짜리 수업 99점, 2시간 95점, 3시간 97점을 받았을 때
# 산술평균은 (99+95+97)/3 이다.
# 가중평균(시간당 평균)을 구하면 (1*99 + 2*95 + 3*97)/6
# =============================================================================


# =============================================================================
# 표본공간: 실험의 결과 하나하나를 모두 모은 것.
# 사건: 표본공간의 부분집합 즉, 일부 값의 모음집. 관심 주제가 무엇인지에 따라 모여있는 집합값이다.
# 원소: 사건의 집합값 중에 개별 값.
# 사건 A가 발생할 확률 P(A) = A의 개수 n(A) / 전체 표본공간의 개수 n(S)

# 이산형: 각 포인트 당 처리 기법
# 동전을 던졌을 때 앞(H) 뒤(T)의 확률
# HH, HT, TH, TT 4가지 경우의 수를 2, 1, 1, 0이라고 생각한다. 그럼 Xi는 {0, 1, 2} 중에 나올 것이다. 실제 0가 몇 번이나 나올지 알고 싶다면?
# 0이 몇 번, 1이 몇 번, 2가 몇 번 나왔는지 각각의 빈도수를 구한다. 이 흩어진 모양을 분포라고 한다.
# 상대도수 = (0의 빈도) / (0, 1, 2의 빈도를 다 더한 값)
# 확률은 0보다 작을 수 없고 1보다 클 수 없다.

# 연속형: 연속된 구간의 처리 기법
# 0~10, 10~20...의 구간을 나눠서 계산한 면적이 확률이 된다.

# 분포함수: 분포를 알려주는 함수. x값을 주면 빈도 그래프의 높이값을 돌려준다. 이산형/연속형에 맞게 함수들이 만들어져있다.
# 균등분포: 빈도의 높이가 똑같은 것. 주사위같이 1/6의 같은 확률을 가진 것.

# 분포 그래프는 n의 크기가 작으면 언덕이 왼쪽에 치우쳐있다가 n이 많이질수록 대칭에 가까워진다.

# 확률 = nCx * P^x * (1-P)^(n-x)
# nCx: n개 중 x가 발생할 모든 경우. nCx = n! / (n-x)! * x!
# P^x: 해당 사건에 대한 확률
# =============================================================================


# =============================================================================
# quantile 기법: 강제적으로 위 아래 1%를 잘라내서 이상치를 찾아냄. 따라서 실제 데이터는 98%이다.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# 난수 만드는 법 randint(시작, 끝, 뽑을 개수)
x1=np.random.randint(0,100,10000)
plt.hist(x1)
m2=[]
s3=[]

# 카이제곱: 샘플 숫자를 계속 늘리면 정규분포에 가까워지지만 맨 끝은 항상 남아있다.
# 10만번 반복
for i in np.arange(100000):
  # 표본 한 그룹 당 10개씩
  x1=np.random.randint(1, 101, 10)
  # 평균들의 모음집
  m1=np.mean(x1)
  s2=np.var(x1)
  m2.append(m1)
  s3.append(s2)
  
np.mean(m2)
np.mean(np.arange(101))
plt.hist(s3)

# 모집단
ss=np.arange(1, 101)
# 각각에 대한 제곱의 합을 전체 개수로 나눔. 즉, mean() 값을 구함.
np.mean((ss-np.mean(ss))**2)
np.var(m2)


# 유의수준: 신뢰할 수 없는 수준. 알파라고 하며, 양쪽을 합쳐서 1이다. 내 표본이 잘못됐을 확률도 있다.