# -*- coding: utf-8 -*-
# 使用CART和其他黑盒模型进行MNIST手写数字分类
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.metrics import classification_report
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字含义
print(digits.target[0])
# 将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()

# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

model ={
    'MLP': MLPClassifier(),
    'KNC': KNeighborsClassifier(),
    'SVC': SVC(),
    'GPC': GaussianProcessClassifier(),
    'DTC': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'ExtraTree': ExtraTreesClassifier(),
    'GraBoost': GradientBoostingClassifier(),
    'logreg': LogisticRegression()

}

prediction_accuracy = []
for value in model.values():
    value.fit(train_ss_x, train_y)
    predict_y_new = value.predict(test_ss_x)
    prediction_accuracy.append(accuracy_score(predict_y_new, test_y))
print(prediction_accuracy)
model_prediction = np.hstack(prediction_accuracy)

#加入list将dict_keys转换为列表，否则会在后续画图中报错
model_name = np.hstack((list(model.keys())))

fig = plt.figure(figsize=(8,4))

sns.barplot(model_prediction, model_name, palette='Blues_d')

plt.xticks(rotation=0, size = 10)
plt.xlabel("prediction_accuracy", fontsize = 12)
plt.ylabel("Model", fontsize = 12)
plt.title("prediction accuracy for different models")

plt.tight_layout()