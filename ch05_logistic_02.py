'''
로지스틱 회귀 분석과 분류를 위한 성능 지표 이론 내용을 기반으로
빈칸(_____)에 코드를 채워보세요.
주석에 추가 가이드 정보를 기재하였습니다.

최종 코드의 결과는 다음의 값만 출력하시고,
제출버튼을 눌러 제출하시면 됩니다.

-------- [최종 출력 결과] --------
Confusion Matrixs :
 [[***    ***]
 [ ***    ***]]
Accuracy: ***, Precision: ***, Recall: 0.***
----------------------------------
'''
# 필요한 데이터셋 로딩
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# 데이터셋 로딩
cancer = load_breast_cancer()

# StandardScaler() 활용한 데이터 스케일링
scaler = StandardScaler()
scaler.fit(cancer.data)
data_scaled = scaler.transform(cancer.data)

# 학습데이터와 테스트 데이터로 분할​
x_train, x_test, y_train, y_test = train_test_split(data_scaled,
                                                    cancer.target,
                                                    test_size=0.3,
                                                    random_state=12)

# 로지스틱 회귀 분석 모델 생성 및 학습
clf = LogisticRegression()
clf.fit(x_train, y_train)

# 학습된 모델에 테스트 데이터(x_test) 입력하여 예측값 생성
y_pred = clf.predict(x_test)

# 오차행렬 생성 및 출력
confusion = confusion_matrix(y_test,y_pred)

print('Confusion Matrixs')
print(confusion)

# Accuracy, Precision, Recall 확인
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy: {0:.4f}, Presion: {1:.4f}, Recall: {2:.4f}'
      .format(accuracy , precision ,recall))