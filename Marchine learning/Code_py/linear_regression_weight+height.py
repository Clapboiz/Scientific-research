# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

# .T la ma tran chuyen vi, chuyen ma tran hang sang ma tran cot
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data
#Pyplot là một module của Matplotlib cung cấp các hàm đơn giản để thêm các thành phần plot như lines, images, text, v.v. vào các axes trong figure.
plt.plot(X, y, 'ro')
#Axis: Chúng là dòng số giống như các đối tượng và đảm nhiệm việc tạo các giới hạn biểu đồ.
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

#(cân nặng) = w_1*(chiều cao) + w_0
# Building Xbar
#Hàm np.ones() cho phép chúng ta khởi tạo một mảng có kích thước tùy chỉnh và các phần tử trong mảng sẽ chỉ mang giá trị là số 1.
one = np.ones((X.shape[0], 1))
#Hàm concatenate () là một hàm từ gói NumPy. Về cơ bản, hàm này kết hợp các mảng NumPy với nhau.
#Hàm này về cơ bản được sử dụng để nối hai hoặc nhiều mảng có cùng hình dạng dọc theo một trục được chỉ định.
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line
#dot tich vo huong
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
#Hàm np. linspace() cũng là một hàm được sử dụng để tạo ra một mảng từ các dãy số được chỉ định trước. Hàm này sẽ tạo ra một mảng Numpy thông qua một dãy số
# và các phần tử trong mảng sẽ được cách đều sao cho phù hợp với ví trị bắt đầu và vị trí kết thúc khoảng.
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line
plt.plot(X.T, y.T, 'ro')     # data
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

## run code thay du lieu train nam kha sat duong thang du doan
y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )
from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)
 ##run code thay k mac loi nao trong cach tim nghiem