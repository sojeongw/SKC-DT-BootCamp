# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:33:26 2019

@author: user
"""

# 워킹 디렉토리는 여기지만 라이브러리는 실제 다른 폴더에 설치되어 가져온다.
# 그 우선권은 내 워킹 디렉토리에 있기 때문에 라이브러리 폴더에
# 같은 이름이 있어도 내 워킹 디렉토리에 만들어놨던 것이 로딩된다.
# 따라서 내 워킹 디렉토리에 같은 이름의 폴더를 만들지 않도록 한다.
import pandas as pd
import os
import os.path as op

# 아래처럼 하면 인코딩 문제로 에러가 발생한다.
# data1=pd.read_csv("./dataset/서울시 공공자전거 대여소 정보_20181129.csv")

# current directory
op.curdir

# change directory
os.chdir('./dataset')

# 어떤 파일이나 폴더를 불러오고 있는지 경로 표시
# os.listdir()

# 리턴값 자체가 list 이므로 불러올 수 있다.
# os.listdir()[1]

# 따라서 리스트 전체를 불러와 적용하면 된다.
list1=os.listdir()

# 파일 내용에 있는 한글을 해결해준다.
# 이것도 안 되면 메모장에서 파일을 열고 utf-8로 다시 지정해서 저장한다.
base=pd.read_csv(list1[1], encoding="utf-8")

# Return the dictionary containing the current scope's global variables.
# 각각의 항목을 키값으로 등록한다.
# globals()

# 문자열이 변수로 바뀐다.
# globals()["y"]=100    # 100이라는 값을 가진 y

# 현재 디렉토리에 있는 파일명을 아래의 리스트에 할당한다.
f_list=["cancer","monthly2017","monthly2018","spot2018","user2017","user2018"]

# 반복문으로 파일 리스트와 파일명을 매칭한다.
# quotechar로 구분자를 없애준다.
for i in range(len(list1)):
  globals()[f_list[i]]=pd.read_csv(list1[i], encoding="utf-8")

# 합친다.
monthly=pd.concat([monthly2017,monthly2018])
monthly.shape
monthly.columns

# 서로 칼럼의 순서와 이름이 같은지 확인한다.
user2017.columns.values == user2018.columns.values
# 합친다.
user=pd.concat([user2017, user2018])
# 하지만 위의 코드는 user2017, 2018 자체도 변경시켜버린다.
# copy를 사용하면 해결된다.
user=pd.concat([user2017.copy(), user2018.copy()])
user.shape
user.columns

user.columns.values[0]="대여일자"
monthly.columns.values[0]="대여일자"

monthly.columns
user.columns

# 1. 2017년 4월 vs 2018년 4월 비교했을 때 활성화 되었는지 확인
monthly.columns     # 어떤 칼럼을 불러올지 확인한다.
monthly.head()    # 날짜가 어떻게 생겼는지 확인하고 아래에서 그 형식대로 4월을 불러온다.
# 대여일자가 4월일 때의 대여건수를 가져온다.
y2017=monthly[monthly["대여일자"]==201704][["대여소번호","대여건수"]]
y2018=monthly[monthly["대여일자"]==201804][["대여소번호","대여건수"]]

# 대여일자 칼럼 별 개수 
monthly["대여일자"].value_counts()

# 데이터 타입 확인. 문자형의 경우 object로 출력
# 둘의 데이터 타입이 같아야만 합칠 수 있다.
y2017.dtypes
y2018.dtypes

# 최종 결과
# on에 동일한 대여소번호 기준으로 합치도록 설정한다.
# 2017년과 2018년의 대여소가 다르기 때문에 비교하고자 하는 대상이
# 동일한 데이터가 되도록 설정하는 것이다.
Q1=pd.merge(y2017, y2018, on="대여소번호")
Q1.shape      # 485개
y2018.shape     # 1268개

# 두 개의 데이터에서 서로 연관있는 것만 묶어서 비교하는 함수 
from scipy.stats import ttest_rel

# 모든 행 중에 1번 열, 2번 열
ttest_rel(Q1.iloc[:,1], Q1.iloc[:,2])
# =============================================================================
#  결과 (statistic=-8.716286015636987, pvalue=4.608389309543589e-17 을 해석해야한다.
#  pvalue는 모집단의 규격 안에 있는지 밖에 있는지 내 위치를 보여준다.
#  statistic이 그 기준보다 작은지 큰지 보여주는 것이다.
#  statistic은 Q1.iloc[:,1]에서 Q1.iloc[:,2] 를 뺀 단위다. 
#  결과값이 음수이므로 후자가 더 크다는 것을 알 수 있다.
#  즉 정규분포 상에서 기준보다 밖에 있다.
#  = 대응인 집단간의 차이 검증
# =============================================================================

# 데이터 변환 이슈=============================================================
# a2=monthly2018["대여소번호"]
# monthly2018.dtypes=="object"
# type(a2)
# # dataframe 타입의 경우 applymap으로 한 번에 replace 적용해서 str 타입으로 변환한다.
# # series는 apply로 해줘야 한다. dataframe이 apply를 쓰면 열 단위로 넘어가버린다.
# monthly2018.applymap(lambda x: str(x).replace("'",""))
# =============================================================================

# 2. 2017년 4월 vs 2018년 4월 이동거리 변화 확인
# 일단 컬럼을 확인해서 이동거리가 어떤 데이터에 있는지 확인한다.
monthly.columns.values
user.columns.values     # 여기에 있다.

# 피봇테이블: pandas에서 각각의 집계를 내주는 함수.
# 201704와 201804에 있는 것만 isin()으로 가져온다.
user.columns
user["대여일자"].value_counts()
date_list=[201704, 201804]
con1=user["대여일자"].isin(date_list)
con1

# 대여일자와 대여소번호를 기준으로 이동거리를 가져온다.
gp_list=["대여일자","대여소번호"]
Q2=user[con1].groupby(gp_list)["이동거리(M)"].mean()

Q2.shape
Q2_1=Q2.index.to_frame()
Q2_1.head()
Q2_2=pd.concat([Q2_1, Q2], axis=1)
Q2_2.head()
Q2_2.columns

Q2_3=Q2_2[Q2_2["대여일자"] == 201704]
Q2_4=Q2_2[Q2_2["대여일자"] == 201804]
# 지금 사용하는 index가 동일하지 않아서 연결되지 않는다.
# 기존 index를 제거하고 다시 정의해줘야 한다.
Q2_f=pd.merge(Q2_3, Q2_4, on="대여소번호", left_index=False, right_index=False) 

# 이렇게 고려해야 하는 점이 많기 때문에 피봇을 사용한다.  
Q2_f=pd.pivot_table(user[user["대여일자"].isin(date_list)],
                    index="대여소번호",
                    columns="대여일자",
                    values="이동거리(M)")
# NaN 정리
Q2_f=Q2_f.dropna()

# 3. 2018년 4월 데이터와 유사한 행정구역 확인

=======
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:33:26 2019

@author: user
"""

# 워킹 디렉토리는 여기지만 라이브러리는 실제 다른 폴더에 설치되어 가져온다.
# 그 우선권은 내 워킹 디렉토리에 있기 때문에 라이브러리 폴더에
# 같은 이름이 있어도 내 워킹 디렉토리에 만들어놨던 것이 로딩된다.
# 따라서 내 워킹 디렉토리에 같은 이름의 폴더를 만들지 않도록 한다.
import pandas as pd
import os
import os.path as op

# 아래처럼 하면 인코딩 문제로 에러가 발생한다.
# data1=pd.read_csv("./dataset/서울시 공공자전거 대여소 정보_20181129.csv")

# current directory
op.curdir

# change directory
os.chdir('./dataset')

# 어떤 파일이나 폴더를 불러오고 있는지 경로 표시
# os.listdir()

# 리턴값 자체가 list 이므로 불러올 수 있다.
# os.listdir()[1]

# 따라서 리스트 전체를 불러와 적용하면 된다.
list1=os.listdir()

# 파일 내용에 있는 한글을 해결해준다.
# 이것도 안 되면 메모장에서 파일을 열고 utf-8로 다시 지정해서 저장한다.
base=pd.read_csv(list1[1], encoding="utf-8")

# Return the dictionary containing the current scope's global variables.
# 각각의 항목을 키값으로 등록한다.
# globals()

# 문자열이 변수로 바뀐다.
# globals()["y"]=100    # 100이라는 값을 가진 y

# 현재 디렉토리에 있는 파일명을 아래의 리스트에 할당한다.
f_list=["cancer","monthly2017","monthly2018","spot2018","user2017","user2018"]

# 반복문으로 파일 리스트와 파일명을 매칭한다.
# quotechar로 구분자를 없애준다.
for i in range(len(list1)):
  globals()[f_list[i]]=pd.read_csv(list1[i], encoding="utf-8")

# 합친다.
monthly=pd.concat([monthly2017,monthly2018])
monthly.shape
monthly.columns

# 서로 칼럼의 순서와 이름이 같은지 확인한다.
user2017.columns.values == user2018.columns.values
# 합친다.
user=pd.concat([user2017, user2018])
# 하지만 위의 코드는 user2017, 2018 자체도 변경시켜버린다.
# copy를 사용하면 해결된다.
user=pd.concat([user2017.copy(), user2018.copy()])
user.shape
user.columns

user.columns.values[0]="대여일자"
monthly.columns.values[0]="대여일자"

monthly.columns
user.columns

# 1. 2017년 4월 vs 2018년 4월 비교했을 때 활성화 되었는지 확인
monthly.columns     # 어떤 칼럼을 불러올지 확인한다.
monthly.head()    # 날짜가 어떻게 생겼는지 확인하고 아래에서 그 형식대로 4월을 불러온다.
# 대여일자가 4월일 때의 대여건수를 가져온다.
y2017=monthly[monthly["대여일자"]==201704][["대여소번호","대여건수"]]
y2018=monthly[monthly["대여일자"]==201804][["대여소번호","대여건수"]]

# 대여일자 칼럼 별 개수 
monthly["대여일자"].value_counts()

# 데이터 타입 확인. 문자형의 경우 object로 출력
# 둘의 데이터 타입이 같아야만 합칠 수 있다.
y2017.dtypes
y2018.dtypes

# 최종 결과
# on에 동일한 대여소번호 기준으로 합치도록 설정한다.
# 2017년과 2018년의 대여소가 다르기 때문에 비교하고자 하는 대상이
# 동일한 데이터가 되도록 설정하는 것이다.
Q1=pd.merge(y2017, y2018, on="대여소번호")
Q1.shape      # 485개
y2018.shape     # 1268개

# 두 개의 데이터에서 서로 연관있는 것만 묶어서 비교하는 함수 
from scipy.stats import ttest_rel

# 모든 행 중에 1번 열, 2번 열
ttest_rel(Q1.iloc[:,1], Q1.iloc[:,2])
# =============================================================================
#  결과 (statistic=-8.716286015636987, pvalue=4.608389309543589e-17 을 해석해야한다.
#  pvalue는 모집단의 규격 안에 있는지 밖에 있는지 내 위치를 보여준다.
#  statistic이 그 기준보다 작은지 큰지 보여주는 것이다.
#  statistic은 Q1.iloc[:,1]에서 Q1.iloc[:,2] 를 뺀 단위다. 
#  결과값이 음수이므로 후자가 더 크다는 것을 알 수 있다.
#  즉 정규분포 상에서 기준보다 밖에 있다.
#  = 대응인 집단간의 차이 검증
# =============================================================================

# 데이터 변환 이슈=============================================================
# a2=monthly2018["대여소번호"]
# monthly2018.dtypes=="object"
# type(a2)
# # dataframe 타입의 경우 applymap으로 한 번에 replace 적용해서 str 타입으로 변환한다.
# # series는 apply로 해줘야 한다. dataframe이 apply를 쓰면 열 단위로 넘어가버린다.
# monthly2018.applymap(lambda x: str(x).replace("'",""))
# =============================================================================

# 2. 2017년 4월 vs 2018년 4월 이동거리 변화 확인
# 일단 컬럼을 확인해서 이동거리가 어떤 데이터에 있는지 확인한다.
monthly.columns.values
user.columns.values     # 여기에 있다.

# 피봇테이블: pandas에서 각각의 집계를 내주는 함수.
# 201704와 201804에 있는 것만 isin()으로 가져온다.
user.columns
user["대여일자"].value_counts()
date_list=[201704, 201804]
con1=user["대여일자"].isin(date_list)
con1

# 대여일자와 대여소번호를 기준으로 이동거리를 가져온다.
gp_list=["대여일자","대여소번호"]
Q2=user[con1].groupby(gp_list)["이동거리(M)"].mean()

Q2.shape
Q2_1=Q2.index.to_frame()
Q2_1.head()
Q2_2=pd.concat([Q2_1, Q2], axis=1)
Q2_2.head()
Q2_2.columns

Q2_3=Q2_2[Q2_2["대여일자"] == 201704]
Q2_4=Q2_2[Q2_2["대여일자"] == 201804]
# 지금 사용하는 index가 동일하지 않아서 연결되지 않는다.
# 기존 index를 제거하고 다시 정의해줘야 한다.
Q2_f=pd.merge(Q2_3, Q2_4, on="대여소번호", left_index=False, right_index=False) 

# 이렇게 고려해야 하는 점이 많기 때문에 피봇을 사용한다.  
Q2_f=pd.pivot_table(user[user["대여일자"].isin(date_list)],
                    index="대여소번호",
                    columns="대여일자",
                    values="이동거리(M)")
# NaN 정리
Q2_f=Q2_f.dropna()

# 3. 2018년 4월 데이터와 유사한 행정구역 확인
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:33:26 2019

@author: user
"""

# 워킹 디렉토리는 여기지만 라이브러리는 실제 다른 폴더에 설치되어 가져온다.
# 그 우선권은 내 워킹 디렉토리에 있기 때문에 라이브러리 폴더에
# 같은 이름이 있어도 내 워킹 디렉토리에 만들어놨던 것이 로딩된다.
# 따라서 내 워킹 디렉토리에 같은 이름의 폴더를 만들지 않도록 한다.
import pandas as pd
import os
import os.path as op

# 아래처럼 하면 인코딩 문제로 에러가 발생한다.
# data1=pd.read_csv("./dataset/서울시 공공자전거 대여소 정보_20181129.csv")

# current directory
op.curdir

# change directory
os.chdir('./dataset')

# 어떤 파일이나 폴더를 불러오고 있는지 경로 표시
# os.listdir()

# 리턴값 자체가 list 이므로 불러올 수 있다.
# os.listdir()[1]

# 따라서 리스트 전체를 불러와 적용하면 된다.
list1=os.listdir()

# 파일 내용에 있는 한글을 해결해준다.
# 이것도 안 되면 메모장에서 파일을 열고 utf-8로 다시 지정해서 저장한다.
base=pd.read_csv(list1[1], encoding="utf-8")

# Return the dictionary containing the current scope's global variables.
# 각각의 항목을 키값으로 등록한다.
# globals()

# 문자열이 변수로 바뀐다.
# globals()["y"]=100    # 100이라는 값을 가진 y

# 현재 디렉토리에 있는 파일명을 아래의 리스트에 할당한다.
f_list=["cancer","monthly2017","monthly2018","spot2018","user2017","user2018"]

# 반복문으로 파일 리스트와 파일명을 매칭한다.
# quotechar로 구분자를 없애준다.
for i in range(len(list1)):
  globals()[f_list[i]]=pd.read_csv(list1[i], encoding="utf-8")

# 합친다.
monthly=pd.concat([monthly2017,monthly2018])
monthly.shape
monthly.columns

# 서로 칼럼의 순서와 이름이 같은지 확인한다.
user2017.columns.values == user2018.columns.values
# 합친다.
user=pd.concat([user2017, user2018])
# 하지만 위의 코드는 user2017, 2018 자체도 변경시켜버린다.
# copy를 사용하면 해결된다.
user=pd.concat([user2017.copy(), user2018.copy()])
user.shape
user.columns

user.columns.values[0]="대여일자"
monthly.columns.values[0]="대여일자"

monthly.columns
user.columns

# 1. 2017년 4월 vs 2018년 4월 비교했을 때 활성화 되었는지 확인
monthly.columns     # 어떤 칼럼을 불러올지 확인한다.
monthly.head()    # 날짜가 어떻게 생겼는지 확인하고 아래에서 그 형식대로 4월을 불러온다.
# 대여일자가 4월일 때의 대여건수를 가져온다.
y2017=monthly[monthly["대여일자"]==201704][["대여소번호","대여건수"]]
y2018=monthly[monthly["대여일자"]==201804][["대여소번호","대여건수"]]

# 대여일자 칼럼 별 개수 
monthly["대여일자"].value_counts()

# 데이터 타입 확인. 문자형의 경우 object로 출력
# 둘의 데이터 타입이 같아야만 합칠 수 있다.
y2017.dtypes
y2018.dtypes

# 최종 결과
# on에 동일한 대여소번호 기준으로 합치도록 설정한다.
# 2017년과 2018년의 대여소가 다르기 때문에 비교하고자 하는 대상이
# 동일한 데이터가 되도록 설정하는 것이다.
Q1=pd.merge(y2017, y2018, on="대여소번호")
Q1.shape      # 485개
y2018.shape     # 1268개

# 두 개의 데이터에서 서로 연관있는 것만 묶어서 비교하는 함수 
from scipy.stats import ttest_rel

# 모든 행 중에 1번 열, 2번 열
ttest_rel(Q1.iloc[:,1], Q1.iloc[:,2])
# =============================================================================
#  결과 (statistic=-8.716286015636987, pvalue=4.608389309543589e-17 을 해석해야한다.
#  pvalue는 모집단의 규격 안에 있는지 밖에 있는지 내 위치를 보여준다.
#  statistic이 그 기준보다 작은지 큰지 보여주는 것이다.
#  statistic은 Q1.iloc[:,1]에서 Q1.iloc[:,2] 를 뺀 단위다. 
#  결과값이 음수이므로 후자가 더 크다는 것을 알 수 있다.
#  즉 정규분포 상에서 기준보다 밖에 있다.
#  = 대응인 집단간의 차이 검증
# =============================================================================

# 데이터 변환 이슈=============================================================
# a2=monthly2018["대여소번호"]
# monthly2018.dtypes=="object"
# type(a2)
# # dataframe 타입의 경우 applymap으로 한 번에 replace 적용해서 str 타입으로 변환한다.
# # series는 apply로 해줘야 한다. dataframe이 apply를 쓰면 열 단위로 넘어가버린다.
# monthly2018.applymap(lambda x: str(x).replace("'",""))
# =============================================================================

# 2. 2017년 4월 vs 2018년 4월 이동거리 변화 확인
# 일단 컬럼을 확인해서 이동거리가 어떤 데이터에 있는지 확인한다.
monthly.columns.values
user.columns.values     # 여기에 있다.

# 피봇테이블: pandas에서 각각의 집계를 내주는 함수.
# 201704와 201804에 있는 것만 isin()으로 가져온다.
user.columns
user["대여일자"].value_counts()
date_list=[201704, 201804]
con1=user["대여일자"].isin(date_list)
con1

# 대여일자와 대여소번호를 기준으로 이동거리를 가져온다.
gp_list=["대여일자","대여소번호"]
Q2=user[con1].groupby(gp_list)["이동거리(M)"].mean()

Q2.shape
Q2_1=Q2.index.to_frame()
Q2_1.head()
Q2_2=pd.concat([Q2_1, Q2], axis=1)
Q2_2.head()
Q2_2.columns

Q2_3=Q2_2[Q2_2["대여일자"] == 201704]
Q2_4=Q2_2[Q2_2["대여일자"] == 201804]
# 지금 사용하는 index가 동일하지 않아서 연결되지 않는다.
# 기존 index를 제거하고 다시 정의해줘야 한다.
Q2_f=pd.merge(Q2_3, Q2_4, on="대여소번호", left_index=False, right_index=False) 

# 이렇게 고려해야 하는 점이 많기 때문에 피봇을 사용한다.  
Q2_f=pd.pivot_table(user[user["대여일자"].isin(date_list)],
                    index="대여소번호",
                    columns="대여일자",
                    values="이동거리(M)")
# NaN 정리
Q2_f=Q2_f.dropna()

# 3. 2018년 4월 데이터와 유사한 행정구역 확인
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:33:26 2019

@author: user
"""

# 워킹 디렉토리는 여기지만 라이브러리는 실제 다른 폴더에 설치되어 가져온다.
# 그 우선권은 내 워킹 디렉토리에 있기 때문에 라이브러리 폴더에
# 같은 이름이 있어도 내 워킹 디렉토리에 만들어놨던 것이 로딩된다.
# 따라서 내 워킹 디렉토리에 같은 이름의 폴더를 만들지 않도록 한다.
import pandas as pd
import os
import os.path as op

# 아래처럼 하면 인코딩 문제로 에러가 발생한다.
# data1=pd.read_csv("./dataset/서울시 공공자전거 대여소 정보_20181129.csv")

# current directory
op.curdir

# change directory
os.chdir('./dataset')

# 어떤 파일이나 폴더를 불러오고 있는지 경로 표시
# os.listdir()

# 리턴값 자체가 list 이므로 불러올 수 있다.
# os.listdir()[1]

# 따라서 리스트 전체를 불러와 적용하면 된다.
list1=os.listdir()

# 파일 내용에 있는 한글을 해결해준다.
# 이것도 안 되면 메모장에서 파일을 열고 utf-8로 다시 지정해서 저장한다.
base=pd.read_csv(list1[1], encoding="utf-8")

# Return the dictionary containing the current scope's global variables.
# 각각의 항목을 키값으로 등록한다.
# globals()

# 문자열이 변수로 바뀐다.
# globals()["y"]=100    # 100이라는 값을 가진 y

# 현재 디렉토리에 있는 파일명을 아래의 리스트에 할당한다.
f_list=["cancer","monthly2017","monthly2018","spot2018","user2017","user2018"]

# 반복문으로 파일 리스트와 파일명을 매칭한다.
# quotechar로 구분자를 없애준다.
for i in range(len(list1)):
  globals()[f_list[i]]=pd.read_csv(list1[i], encoding="utf-8")

# 합친다.
monthly=pd.concat([monthly2017,monthly2018])
monthly.shape
monthly.columns

# 서로 칼럼의 순서와 이름이 같은지 확인한다.
user2017.columns.values == user2018.columns.values
# 합친다.
user=pd.concat([user2017, user2018])
# 하지만 위의 코드는 user2017, 2018 자체도 변경시켜버린다.
# copy를 사용하면 해결된다.
user=pd.concat([user2017.copy(), user2018.copy()])
user.shape
user.columns

user.columns.values[0]="대여일자"
monthly.columns.values[0]="대여일자"

monthly.columns
user.columns

# 1. 2017년 4월 vs 2018년 4월 비교했을 때 활성화 되었는지 확인
monthly.columns     # 어떤 칼럼을 불러올지 확인한다.
monthly.head()    # 날짜가 어떻게 생겼는지 확인하고 아래에서 그 형식대로 4월을 불러온다.
# 대여일자가 4월일 때의 대여건수를 가져온다.
y2017=monthly[monthly["대여일자"]==201704][["대여소번호","대여건수"]]
y2018=monthly[monthly["대여일자"]==201804][["대여소번호","대여건수"]]

# 대여일자 칼럼 별 개수 
monthly["대여일자"].value_counts()

# 데이터 타입 확인. 문자형의 경우 object로 출력
# 둘의 데이터 타입이 같아야만 합칠 수 있다.
y2017.dtypes
y2018.dtypes

# 최종 결과
# on에 동일한 대여소번호 기준으로 합치도록 설정한다.
# 2017년과 2018년의 대여소가 다르기 때문에 비교하고자 하는 대상이
# 동일한 데이터가 되도록 설정하는 것이다.
Q1=pd.merge(y2017, y2018, on="대여소번호")
Q1.shape      # 485개
y2018.shape     # 1268개

# 두 개의 데이터에서 서로 연관있는 것만 묶어서 비교하는 함수 
from scipy.stats import ttest_rel

# 모든 행 중에 1번 열, 2번 열
ttest_rel(Q1.iloc[:,1], Q1.iloc[:,2])
# =============================================================================
#  결과 (statistic=-8.716286015636987, pvalue=4.608389309543589e-17 을 해석해야한다.
#  pvalue는 모집단의 규격 안에 있는지 밖에 있는지 내 위치를 보여준다.
#  statistic이 그 기준보다 작은지 큰지 보여주는 것이다.
#  statistic은 Q1.iloc[:,1]에서 Q1.iloc[:,2] 를 뺀 단위다. 
#  결과값이 음수이므로 후자가 더 크다는 것을 알 수 있다.
#  즉 정규분포 상에서 기준보다 밖에 있다.
#  = 대응인 집단간의 차이 검증
# =============================================================================

# 데이터 변환 이슈=============================================================
# a2=monthly2018["대여소번호"]
# monthly2018.dtypes=="object"
# type(a2)
# # dataframe 타입의 경우 applymap으로 한 번에 replace 적용해서 str 타입으로 변환한다.
# # series는 apply로 해줘야 한다. dataframe이 apply를 쓰면 열 단위로 넘어가버린다.
# monthly2018.applymap(lambda x: str(x).replace("'",""))
# =============================================================================

# 2. 2017년 4월 vs 2018년 4월 이동거리 변화 확인
# 일단 컬럼을 확인해서 이동거리가 어떤 데이터에 있는지 확인한다.
monthly.columns.values
user.columns.values     # 여기에 있다.

# 피봇테이블: pandas에서 각각의 집계를 내주는 함수.
# 201704와 201804에 있는 것만 isin()으로 가져온다.
user.columns
user["대여일자"].value_counts()
date_list=[201704, 201804]
con1=user["대여일자"].isin(date_list)
con1

# 대여일자와 대여소번호를 기준으로 이동거리를 가져온다.
gp_list=["대여일자","대여소번호"]
Q2=user[con1].groupby(gp_list)["이동거리(M)"].mean()

Q2.shape
Q2_1=Q2.index.to_frame()
Q2_1.head()
Q2_2=pd.concat([Q2_1, Q2], axis=1)
Q2_2.head()
Q2_2.columns

Q2_3=Q2_2[Q2_2["대여일자"] == 201704]
Q2_4=Q2_2[Q2_2["대여일자"] == 201804]
# 지금 사용하는 index가 동일하지 않아서 연결되지 않는다.
# 기존 index를 제거하고 다시 정의해줘야 한다.
Q2_f=pd.merge(Q2_3, Q2_4, on="대여소번호", left_index=False, right_index=False) 

# 이렇게 고려해야 하는 점이 많기 때문에 피봇을 사용한다.  
Q2_f=pd.pivot_table(user[user["대여일자"].isin(date_list)],
                    index="대여소번호",
                    columns="대여일자",
                    values="이동거리(M)")
# NaN 정리
Q2_f=Q2_f.dropna()

# 3. 2018년 4월 데이터와 유사한 행정구역 확인