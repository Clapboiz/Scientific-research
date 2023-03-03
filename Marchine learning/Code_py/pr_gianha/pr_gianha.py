import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

price = pd.read_csv("train_gianha.csv")
print(price.head())
print(price._data)
print(price.info())

sns.set_style("whitegrid")
sns.displot(price['SalePrice'], kde=True, rug=True, bins=50)
plt.show()

x = price [["Id", "MSSubClass", "BsmtUnfSF", "TotalBsmtSF","2ndFlrSF","LowQualFinSF",
            "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd","GrLivArea","BsmtFullBath",
            "1stFlrSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtHalfBath", "FullBath", "HalfBath",
            "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
            "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
            "PoolArea", "MiscVal", "MoSold", "YrSold"]]
y = price[["SalePrice"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
print(x_train)
print(y_train)
print(x_test)

#import model vao de chay
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
print(lr.fit(x_train, y_train))

pr = lr.predict(x_test)
print(pr)

#danh gia mo hinh
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pr))
print('MSE:', metrics.mean_squared_error(y_test, pr))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pr)))

#score
print("score: ", metrics.explained_variance_score(y_test, pr))