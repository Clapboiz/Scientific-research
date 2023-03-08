import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


#read datasetsZS
data = pd.read_csv("Mall_Customers.csv")
print(data.head())
print(data.info)
print(data._data)

#bieu do theo thu nhap
sns.set(style = 'whitegrid', rc={"figure.figsize":(10, 6)})
# plt.figure(figsize=(10,6))
sns.displot(data['Annual Income (k$)'])
#ve theo doanh thu

plt.title('bieu do sx thu nhap "$"')
plt.xlabel('sap xep thu nhap')
plt.ylabel('count')
#bieu do theo do tuoi
sns.displot(data['Age'])
plt.title('bieu do theo do tuoi "Age"')
plt.xlabel('sap xep do tuoi')
plt.ylabel('count')

genders = data.Gender.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values)
plt.title('Genders')
plt.xlabel('sap xep gioitinh')
plt.ylabel('count')
# plt.show()

df1 = data[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
df2 = data[['CustomerID']]
# Chuẩn hóa dữ liệu
X = data.iloc[:, 3:].values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(X)
y = scaler.fit_transform(df2)
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)
#The elbow curve
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
# plt.show()
############# Nhin vao do thi tren thi ta thay den k=5 thi do thi bat dau giam nhe, vi vay chung ta se lay k = 5
#Hien thi duoi dang phan cum

n_clusters = 5 #so luong cum
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'cum 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'cum 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'cum 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'cum 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'cum 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Tâm cụm')
plt.title('Phân cụm khách hàng trung tâm')
plt.xlabel('Độ tuổi')
plt.ylabel('Điểm thu nhập')
plt.legend()
# plt.show()

# # Lưu nhãn của từng điểm sau khi phân cụm vào cột 'cluster'
df1['cluster'] = kmeans.labels_
# Nhóm các điểm theo nhãn của chúng
grouped = df1.groupby('cluster')
# Lấy ra ID của các điểm trong mỗi nhóm
for cluster, group in grouped:
    ids = list(group.index)
    print(f'Các ID trong nhóm {cluster}: {ids}')
# tạo DataFrame mới với 2 giá trị Annual Income (k$) = 50 và Spending Score (1-100) = 80
new_data = pd.DataFrame({
    'Annual Income (k$)': [50],
    'Spending Score (1-100)': [80]
})

# khởi tạo scaler
sc = StandardScaler()
# chuẩn hóa giá trị trong DataFrame mới
new_data = sc.fit_transform(new_data)
# tạo mô hình KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
# fit mô hình với dữ liệu huấn luyện
kmeans.fit(X)
# dự đoán nhóm của new_data
prediction = kmeans.predict(new_data)
# in ra nhóm của new_data
print('Nhóm của new_data:', prediction)

#score
# # Silhouette Score
# silhouette = silhouette_score(X, y_kmeans)
# print("Silhouette Score df1, df2:", silhouette)

# # Calinski-Harabasz Index
# ch_score = calinski_harabasz_score(df1, df2)
# print("Calinski-Harabasz Index:", ch_score)
#
# # Davies-Bouldin Index
# db_score = davies_bouldin_score(X, y_kmeans)
# print("Davies-Bouldin Index:", db_score)
