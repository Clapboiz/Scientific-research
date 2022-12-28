import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data_classification.csv", header = None)
# print(data)
print(data.values)
# khai bao mang
true_x=[]
true_y=[]
false_x = []
false_y = []
for item in data.values:
    #phan tu thu 2 bang 1 thi gan gia tri nhu duoi
    if item[2] ==1.:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])

#bieu do phan tan , truyen 4 gia tri vao , marker la hinh dang, c la color, b la blue, r la red
plt.scatter(true_x, true_y, marker = 'o', c='b')
plt.scatter(false_x, false_y, marker = 'x', c='r')
plt.show()

#cong thuc tinh sai so
def sigmoid(z):
    return 1.0 / (1+np.exp(-z))

#ham lam tron
def phan_chia(p):
    if p>=0.5:
        return 1
    else:
        return 0
# ham du doan dua vao ct sai so o tren
def predict(feature, weight):
    z=np.dot(feature, weights)
    return sigmoid(z)

#ham tinh gia
def cost_func (features, labels, weights):
    """
    :param features: (100X3)
    :param labels: (100X1) Co|khong 1|0
    :param weights: (3x1)
    :return: chi phi cost
    """
    n= len(labels)
    predictions = predict(features, weights)
#gia class1 dung labels nhan , phan tach duoc labels = 1, boi vi 0*....=0
    cost_class1 = -labels*np.log(predictions)
#phan tach duoc labels 0
    cost_class2 = -(1- labels)*np.log(1-predictions)
#tong 2 mang lai
    cost = cost_class1+cost_class2
#sum 2 mang
    return cost.sum()/n

def update_weight(features, labels, weights, learning_rate):
    """

    :param features: 100x3
    :param labels: 100x1
    :param weights: 3x1
    :param learning_rate: float
    :return: new weight: float
    """

    n=len(labels)
    predictions = predict(features, weights)
#features.T la chuyen vi ma tran
    gd = np.dot(features.T, predictions-labels)
    gd=gd/n
    gd = gd*learning_rate
    weights = weights-gd
    return weights

# #phuongsai
# np.array([10,8,10,8,8,4])
# #phuong sai voi n
# np.var(a)
# #phuong sai voi n - 1
# np.var(a, ddof=1)
#ham training
def train(features, labels, weights, learning_rate, iter):
    cost_hs = []
    for i in range(iter):
        weights = update_weight(features, labels, weights, learning_rate)
        cost=cost_func(features, labels, weights)
        cost_hs.append(cost)
    return weights, cos_hs
