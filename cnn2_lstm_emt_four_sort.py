
import numpy as np
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from keras.layers import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix

raw_num = 240
col_num = 2000

data_ini = pd.read_csv('dl_data_train.csv')
data = np.array(data_ini)

(h, w) = data.shape
# learns to repeat simple sequence from random inputs
np.random.seed(0)
# parameters for input data dimension and lstm cell count
mem_cell_ct = 200
x_dim = w - 1
label = data[:, 0:1]

data_samp = data[:, 1:w]
lb = LabelBinarizer()
y = lb.fit_transform(label)

X_train, X_test, y_train, y_test = train_test_split(data_samp, y, test_size=0.3)

def built_model():
    input_seq = Input(shape=(1000,))
    X = Reshape((1, 1000))(input_seq)


    ec2_layer1 = Conv1D(filters=50, kernel_size=6, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(X)
    ec2_layer2 = Conv1D(filters=40, kernel_size=8, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer1)
    ec2_layer3 = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                              data_format='channels_first')(ec2_layer2)
    ec2_layer4 = Conv1D(filters=30, kernel_size=6, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer3)
    ec2_layer5 = Conv1D(filters=30, kernel_size=8, strides=2,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer4)
    ec2_outputs = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                               data_format='channels_first')(ec2_layer5)



    dc_layer1 = LSTM(60, return_sequences=True)(ec2_outputs)
    dc_layer2 = LSTM(60)(dc_layer1)
    dc_layer3 = Dropout(0.5)(dc_layer2)
    dc_layer4 = Dense(10)(dc_layer3)
    dc_layer5= Dense(4,activation='softmax')(dc_layer4)

    model = Model(input_seq, dc_layer5)

    return model

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, normalize=False):
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_mark = np.arange(len(classes))
    plt.xticks(tick_mark, classes, rotation=40)
    plt.yticks(tick_mark, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = '%.2f' % cm
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.show()

import time

begain_time = time.time()

# In[8]:


model = built_model()
opt = Adam(lr=0.00020)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model.summary()

# In[9]:


history = model.fit(x=X_train, y=y_train, batch_size=100, epochs=60,
                    verbose=2, validation_data=(X_test, y_test),
                    shuffle=True, initial_epoch=0)
model.save("h5forder/cnn_lstm_high_80epoch.h5")

y_pre = model.predict(X_test)
label_pre = np.argmax(y_pre, axis=1)
label_true = np.argmax(y_test, axis=1)
confusion_mat = confusion_matrix(label_true, label_pre)
plot_confusion_matrix(confusion_mat, classes=range(4))





