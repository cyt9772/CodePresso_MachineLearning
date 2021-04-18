'''
강의를 수강하시면서 하단 빈칸(_____)에 코드를 채워보세요.
주석에 추가 가이드 정보를 기재하였습니다.
최종 코드의 결과는 다음의 값만 출력하시고,
제출버튼을 눌러 제출하시면 됩니다.

-------- [최종 출력 결과] --------
Weight : ***
Bias : ***

MSE  : ***
MAE  : ***
RMSE : ***
MAPE : ***

R-squared(r2_score) : ***
R-squared(r2_metric) : ***
----------------------------------
'''
# 필요한 라이브러리 로딩
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 모델 성능 평가를 위한 metrics 모듈 로딩
from sklearn import metrics


# 데이터셋 로딩
boston = load_boston()

# 데이터셋 분할
# random_state 값은 강의와 동일하게 지정하세요.
x_train, x_test, y_train, y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    test_size=0.3,
                                                    random_state=12)

# LinearRegression 객체 생성
regression = LinearRegression()

# 학습데이터 연결 및 학습 수행
regression.fit(x_train, y_train)

# 모델 예측
y_pred = regression.predict(x_test)

# 회귀 계수 출력
weight = np.round(regression.coef_,1)
bias = np.round(regression.intercept_,2)
print('Weight:', weight)
print('Bias:', bias)

# 컬럼별 회귀계수 출력
coef_table = pd.Series(data=weight,
                       index=boston.feature_names)

# 아래는 출력 결과만 확인하시고,
# 최종 제출시에는 주석으로 처리해주세요
#print('Regression Coefficients :')
#print(coef_table.sort_values(ascending=False))

# 회귀 분석 모델을 위한 평가 지표 계산
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
mape = metrics.mean_absolute_percentage_error(y_test,y_pred)

print('MSE  : {0:.3f}'.format(mse))
print('MAE  : {0:.3f}'.format(mae))
print('RMSE : {0:.3f}'.format(rmse))
print('MAPE : {0:.3f}'.format(mape))

# R-squared 를 통한 모델의 설명력 평가
r2_score = regression.score(x_test, y_test)
r2_metric = metrics.r2_score(y_test, y_pred)

print('\nR-squared(r2_score) : {0:.3f}'.format(r2_score))
print('R-squared(r2_metric) : {0:.3f}'.format(r2_metric))