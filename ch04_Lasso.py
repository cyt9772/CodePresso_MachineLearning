'''
sklearn.linear_model.Ridge 실습 내용을 기반으로
빈칸(_____)에 코드를 채워보세요.
주석에 추가 가이드 정보를 기재하였습니다.

최종 코드의 결과는 다음의 값만 출력하시고,
제출버튼을 눌러 제출하시면 됩니다.

※최종 코드는 반드시 alpha=0.1 로 지정한 실행 결과를 제출 부탁드립니다.
-------- [최종 출력 결과] --------
Training-datasset R2 :
Test-datasset R2 :
Lasso Regression Coefficients :
----------------------------------
'''
# 필요한 라이브러리 로딩
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# sklearn.linear_model 모듈의 Lasso 클래스 로딩
from sklearn.linear_model import Lasso

# 데이터셋 로딩
boston = load_boston()

# 데이터셋 분할
# test_size=0.3, random_state=12 로 지정하세요.
x_train, x_test, y_train, y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    test_size=0.3,
                                                    random_state=12)

# 규제를 위한 alpha 값 초기화
# 학습시에는 alpha 값을 바꾸가면서 테스트해보시고,
# 최종 코드 제출시에는 0.1 로 지정후 제출하세요.
alpha = 0.1

# Lasso 클래스 객체 생성
lasso = Lasso(alpha=alpha)

# fit() 을 통한 규제 학습 수행
lasso.fit(x_train, y_train)

# predict() 를 통한 학습된 모델 기반 예측
lasso_pred = lasso.predict(x_test)

# score() 를 통해 회귀 모델의 R^2 출력
# 학습된 모델에 대한 R^2 계산
r2_train = lasso.score(x_train, y_train)
r2_test = lasso.score(x_test, y_test)
print('Training-datasset R2 : {0:.3f}'.format(r2_train))
print('Test-datasset R2 : {0:.3f}'.format(r2_test))

# 회귀 계수 저장을 위한 Seriess 객체 생성 및 출력
lasso_coef_table = pd.Series(data=np.round(lasso.coef_,1),
                        index=boston.feature_names)
print('Lasso Regression Coefficients :')
print(lasso_coef_table.sort_values(ascending=False))

# 막대그래프 시각화
plt.figure(figsize=(10,5))
lasso_coef_table.plot(kind='bar')
plt.ylim(-10, 4)
plt.show()