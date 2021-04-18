'''
sklearn.preprocessing.StandardScaler 실습 내용을
기반으로 빈칸(_____)에 코드를 채워보세요.
주석에 추가 가이드 정보를 기재하였습니다.

최종 코드의 결과는 다음의 값만 출력하시고,
제출버튼을 눌러 제출하시면 됩니다.

-------- [최종 출력 결과] --------
Min Value
temp         ***
atemp        ***
humidity     ***
windspeed    ***
dtype: float64
Max Value
temp          ***
atemp         ***
humidity     ***
windspeed     ***
dtype: float64
--------- MinMaxScaler ---------
Min Value
temp         ***
atemp       ***
humidity    ***
windspeed   ***
dtype: float64
Max Value
temp         ***
atemp        ***
humidity     ***
windspeed    ***
dtype: float64
----------------------------------
'''
# 필요한 라이브러리 로딩
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# MinMaxScaler 로딩
from sklearn.preprocessing import MinMaxScaler

# URL 통해서 캐글의 자전거 대여 수요 데이터셋 다운로드
url = 'https://codepresso-online-platform-public.s3.ap-northeast-2.amazonaws.com/learning-resourse/python-machine-learning-20210326/bike-demand.csv'
df_bike = pd.read_csv(url)

# 확인용으로만 출력하시고, 최종 코드 제출시에는 주석 처리
# print(df_bike.head(5))

# temp, atemp, humidity, windspeed	컬럼 데이터만 저장
df_bike_num = df_bike.iloc[:,5:9]

# 확인용으로만 출력하시고, 최종 코드 제출시에는 주석 처리
# print(df_bike_num.head(5))

# 각 컬럼별 최대/최소값 출력
print('Min Value')
print(np.round_(df_bike_num.min(),3))
print('Max Value')
print(np.round_(df_bike_num.max(),3))

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()

# fit 함수 이용하여 데이터 분포 분석 및 스케일링 정보 저장
scaler.fit(df_bike_num)

# 실제 데이터 스케일링 작업 후 결과 데이터셋 반환
result = scaler.transform(df_bike_num)

# 스케일된 결과 데이터를 DataFrame 으로 저장
scaled_bike = pd.DataFrame(data=result, columns=df_bike_num.columns)

# 각 컬럼별 최대/최소값 출력
print('--------- MinMaxScaler ---------')
print('Min Value')
print(np.round_(scaled_bike.min(),3))
print('Max Value')
print(np.round_(scaled_bike.max(),3))


# 박스플롯(boxplot) 으로 시각화
plt.figure(figsize=(10,6))
scaled_bike.boxplot(column=['temp','atemp','humidity','windspeed'])
plt.show()