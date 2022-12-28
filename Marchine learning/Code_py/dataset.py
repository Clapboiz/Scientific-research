#                       # in ra ma tran du lieu
from sklearn.datasets import load_iris
iris_dataset = load_iris()
#in ra do dai nhan
print(len(iris_dataset.target))
#in ra target
print(iris_dataset.target)
#in ra du lieu ma tran
print(iris_dataset.data)

#do chinh xac voi bo du lieu
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)
model = DecisionTreeClassifier()
mymodel = model.fit(x_train, y_train)
x_new = np.array([[6.0,3.23,4.5,2.0]])
print("do chinh xac")
print(mymodel.score(x_test, y_test))

# import matplotlib.pyplot as plt
# import numpy as np
# x = [3,5]
# y = [7,9]
# plt.plot(x,y)
# plt.show()

# import pandas as pd
#
# # tạo dict từ các series
# s = {'một': pd.Series([2., 0., 0., 3.]),
#      'hai': pd.Series([0., 8., 0., 3.])}
#
# # tại DataFrame từ dict
# df = pd.DataFrame(s)
#
# print(df)



