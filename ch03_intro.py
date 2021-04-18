# Kaggle : http://www.kaggle.com/datasets
# UCI 머신러닝 저장소 : http://archive.ics.uci.edu/ml
# 공공 데이터 포탈 : https://www.data.go.kr/

# 라이브러리 빌트인 데이터셋
    # scikit-learn : https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
    # seaborn : https://github.com/mwaskom/seaborn-data

import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#dataset loading
iris=load_iris()

# print(sklearn.__version__)
#print(iris.DESCR)
# print('iris feature name:', iris.feature_names)
# print('iris data shape: ', iris.data.shape)
# print('iris data: ', iris.data)
# print('iris data type: ', type(iris.data))
# #==========================================
# print('==========================================')
# print('iris target name: ', iris.target_names)
# print('iris target value: ', iris.target)

#data division for training(70%) and testing(30%)
#using train_test_split 이용
x_train, x_test, y_train, y_test=train_test_split(iris.data,
                                                      iris.target,
                                                      test_size=0.3,
                                                      random_state=11)
# print('x_train.shape= ', x_train.shape)
# print('x_test.shape= ', x_target.shape)
# print('y_train.shape= ', y_train.shape)
# print('y_target.shape= ', y_target.shape)

k_list=range(1,100,2)
acc_train=[]
acc_test=[]

for k in k_list:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    acc_train.append(knn.score(x_train, y_train))
    acc_test.append(knn.score(x_test, y_test))


plt.figure(figsize=(10,4))
plt.plot(k_list, acc_train, 'b--')
plt.plot(k_list, acc_test, 'g')
plt.title('Training and Testing Accuraty')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
plt.show()