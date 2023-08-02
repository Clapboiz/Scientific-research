#Import Lib
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score

from tensorflow.keras.layers import Dense, LSTM, RNN, Embedding, Dropout
from tensorflow.keras.models import Sequential

from keras.backend import clear_session

import os

df = pd.read_csv(r'D:/Users/Desktop/Scientific-research/Deep learning/Data/Android-malware/drebin-215-dataset-5560malware-9476-benign.csv')
#Pretrain dataset
def preprocess(dataframe):
    dataframe.loc[dataframe['class'] == "B", "class"] = 0
    dataframe.loc[dataframe['class'] == "S", "class"] = 1

    return dataframe

scaled_train = preprocess(df)

X = scaled_train.drop(['class'], axis=1).values
y = scaled_train['class'].values

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert columns to numeric data type
df = df.apply(pd.to_numeric)

# Replace NaN values with mean of column
df.fillna(df.mean(), inplace=True)

# Convert labels to integers
df['class'] = df['class'].astype('category').cat.codes

# Convert labels to integers
y = y.astype('int')

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

model = Sequential()
model.add(LSTM(units=32, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

# Compile models
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit model
history = model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, epochs=10, batch_size=32,
                    validation_data=(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test))