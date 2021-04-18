'''
강의를 수강하시면서 하단 빈칸(_____)에 코드를 채워보세요.
주석에 추가 가이드 정보를 기재하였습니다.
최종 코드의 결과는 다음의 값만 출력하시고,
제출버튼을 눌러 제출하시면 됩니다.

-------- [최종 출력 결과] --------
Data shape: ***
X shape: ***
y shape: ***
X reshape result: ***
y reshape result: ***
Testing data shape: ***
----------------------------------
'''
# 필요한 라이브러리 로딩
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# Boston housing price 데이터셋 로딩
boston = load_boston()
print('Data shape:', boston.data.shape)

# 독립변수, 종속변수 데이터 정의
X_rooms = boston.data[:,5]
y = boston.target
print('X shape:', X_rooms.shape)
print('y shape:', y.shape)

# 데이터의 분포 확인을 위해 산점도로 시각화
# plt.figure()
# plt.scatter(X_rooms,y)
# plt.xlabel('Number of rooms')
# plt.ylabel('Values of house')
# plt.show()

# 2차원 데이터로 shape 변환
X_rooms = X_rooms.reshape(-1,1)
y = y.reshape(-1,1)
print('X reshape result:', X_rooms.shape)
print('y reshape result:' , y.shape)

# LinearRegression 객체 생성
regression = LinearRegression()

# 학습데이터 연결 및 학습 수행
regression.fit(X_rooms,y)

# 테스팅에 사용할 데이터 생성
testing = np.linspace(min(X_rooms),max(X_rooms)).reshape(-1,1)
print('Testing data shape:', testing.shape)

# 모델 예측 수행
y_pred = regression.predict(testing)

# 최적의 회귀선 시각화
plt.figure(figsize=(10, 5))
# plt.scatter 이용해서 산점도 시각화
plt.scatter(X_rooms,y)
# plt.plot 이용해서 라인 그래프 시각화
plt.plot(testing, y_pred, color='red',Linewidth=3)
plt.xlabel('Number of rooms')
plt.ylabel('Values of house')
plt.show()