import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import keras
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.backend import clear_session

import matplotlib.pyplot as plt    

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from keras.layers import BatchNormalization
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Split data into train and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      test_size=0.1,
                                                      random_state=40)

# Print shape of each set
print("Shape of training set:", x_train.shape)
print("Shape of validation set:", x_valid.shape)
print("Shape of test set:", x_test.shape)
import matplotlib.pyplot as plt

for i in range(20):
    #define subplot
    plt.subplot(5, 5, i+1)
    #plot pixel data
    plt.imshow(x_train[i], cmap=plt.cm.binary)
#diplay the images
plt.show()
import keras
from keras.utils import to_categorical

#set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train_ = keras.utils.to_categorical(y_train, num_category)
y_test_ = keras.utils.to_categorical(y_test, num_category)
y_valid_ = keras.utils.to_categorical(y_valid, num_category)
#dividing image pixel by 255 so that pixel comes in range 0 to 1...
x_test=x_test/255.0
x_train=x_train/255.0
x_valid=x_valid/255.0
# create model
model = Sequential()

model.add(BatchNormalization())

#16 is filters
model.add(Conv2D(32, input_shape = (32, 32, 3),
                 kernel_size = (3, 3),
                 kernel_initializer = "normal", 
                 padding = 'same',
                 activation = 'relu'))

model.add(MaxPool2D(pool_size = (2, 2), strides = 2))

model.add(Dropout(0.25))

model.add(BatchNormalization())

model.add(Conv2D(64, 
                 kernel_size = (3, 3),
                 kernel_initializer = "normal",
                 padding = 'same',
                 activation = 'relu'))       

model.add(MaxPool2D(pool_size = (2, 2), strides = 2))

model.add(Dropout(0.25))

model.add(BatchNormalization())

model.add(Conv2D(128, 
                kernel_size = (3, 3),
                padding = "same",
                kernel_initializer = "normal",
                activation = "relu"))    

model.add(MaxPool2D(pool_size = (2, 2), strides = 2))

model.add(Dropout(0.25))

model.add(Conv2D(256, 
                kernel_size = (3, 3),
                padding = "same",
                kernel_initializer = "normal",
                activation = "relu"))    

model.add(MaxPool2D(pool_size = (2, 2), strides = 2))

model.add(Dropout(0.25))

model.add(Conv2D(512, 
                kernel_size = (3, 3),
                padding = "same",
                kernel_initializer = "normal",
                activation = "relu"))    


model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax'))

#compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
num_fold = 1

qtd_fods = 2

caminho_salvar_modelo = "models/"
monitor = EarlyStopping(monitor = "val_loss", min_delta = 1e-1, patience = 5, verbose = 1, mode  ="auto")

folds  = KFold(n_splits = qtd_fods, shuffle = True, random_state = 1).split(x_train, y_train)

argumentation = ImageDataGenerator(rotation_range=20,
                                   zoom_range=0.15,
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2,
                                   shear_range=0.15,
                                   horizontal_flip=True, 
                                   fill_mode="nearest")
# train model
Score = model.fit(x_train, y_train_, epochs=64, batch_size=256, verbose=1,
                  validation_data=(x_test, y_test_))
# Metrics
y_test_predictions = np.argmax(model.predict(x_test), axis=-1)
sc_rc = recall_score(y_test, y_test_predictions, average='weighted')
sc_f1 = f1_score(y_test, y_test_predictions, average='weighted')

print("=========== METRICS =============")
print("Recall score: %.2f" % (sc_rc * 100))
print("F1 score: %.2f" % (sc_f1 * 100))