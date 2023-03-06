# Import thư viện
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix # for sparse matrix
from sklearn.metrics import accuracy_score # for evaluating results
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix

#import model classification
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from scipy.sparse import csr_matrix
# Đọc dữ liệu email từ tập tin
emails = pd.read_csv("spam_emails_dataset.csv")
print(emails.head())
print(emails._data)
print(emails.info)

np.random.seed(0)

# Tạo một danh sách chứa các nội dung email và các nhãn tương ứng (spam hoặc ham)
contents = emails['text']
labels = emails["label_num"]

# Chuyển đổi nội dung email thành các vector đặc trưng
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(contents)

# Phân chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Huấn luyện mô hình Naive Bayes trên tập huấn luyện
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Đánh giá độ chính xác của mô hình trên tập kiểm tra
accuracy = clf.score(X_test, y_test)
print("Độ chính xác của mô hình là:", accuracy)

# Dự đoán các nhãn của các email trong tập kiểm tra
y_pred = clf.predict(X_test)
print(y_pred)

#train hang loat model theo kieu cross , nghia la theo tung fold
seed = 2023
models = [
    LinearSVC(max_iter=12000,random_state=seed),
    SVC(random_state=seed),
    KNeighborsClassifier(metric='minkowski', p=2),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    MultinomialNB(),
    BernoulliNB(),
    # bug GaussianNB() em fix chua duoc :"))
    # GaussianNB() #typeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
]

from sklearn.model_selection import StratifiedKFold
def generate_baseline_results(models, X, labels, metrics, cv=5, plot_results=False):

    # define k-fold:
    kfold = StratifiedKFold(cv, shuffle=True, random_state=seed)
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        print(model_name)
        scores = cross_val_score(model, X, labels, scoring=metrics, cv=kfold)
        for fold_idx, score in enumerate(scores):
            entries.append((model_name, fold_idx, score))

    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_id', 'accuracy_score'])
    print(cv_df)
    if plot_results:
        sns.boxplot(x='model_name', labels='accuracy_score', data=cv_df, color='lightblue', showmeans=True)
        plt.title("Boxplot of Base-Line Model Accuracy using 5-fold cross-validation")
        plt.xticks(rotation=45)
        plt.show()

    # Summary result
    mean = cv_df.groupby('model_name')['accuracy_score'].mean()
    std = cv_df.groupby('model_name')['accuracy_score'].std()

    baseline_results = pd.concat([mean, std], axis=1, ignore_index=True)
    baseline_results.columns = ['Mean', 'Standard Deviation']

    # sort by accuracy
    baseline_results.sort_values(by=['Mean'], ascending=False, inplace=True)
    print(baseline_results)

generate_baseline_results(models, X, labels, metrics='accuracy', cv=5, plot_results=False)