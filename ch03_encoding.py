"""
-------- [최종 출력 결과] --------
LabelEncoding classes: ***
LabelDecoding result: ***
OneHotEncoding classes: ***
----------------------------------
"""

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

tips = sns.load_dataset('tips')
# print(tips.head(5))
#
# print('Tips Categories: ', tips['day'].unique())

items=tips['day']

encoder=LabelEncoder()
encoder.fit(items)

labels=encoder.transform(items)
#print('Label Encoding Result:', labels)
print('LabelEncoding classes: ', encoder.classes_)
print('LabelDecoding Result:', encoder.inverse_transform([2]))#2는 index

#print('======= OneHotEncoder  =============')
labels=labels.reshape(-1,1)
one_hot_encoder=OneHotEncoder()
one_hot_encoder.fit(labels)
one_hot_labels=one_hot_encoder.transform(labels)

#print('OnehotEncoding Result:', one_hot_labels.toarray())
onehot_classes=one_hot_encoder.categories_
print('OneHotEncoding classes:', onehot_classes)
