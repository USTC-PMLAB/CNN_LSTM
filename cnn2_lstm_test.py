import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Model
import matplotlib.pyplot as plt

data_ini = pd.read_csv('dl_data_test.csv')
data = np.array(data_ini)
(h, w) = data.shape
x_dim = w - 1
y_test = data[0:1199, 1:w]
new_model=tf.keras.models.load_model("h5forder/cnn_lstm_high_80epoch.h5.h5")
new_model.summary()
net_var='dense_1'
middle=Model(inputs=new_model.input,outputs=new_model.get_layer(net_var).output)
result=middle.predict(y_test)
print(result)
file_var='cnn_lstm_high_80epoch.h5.csv'
pd.DataFrame(result).to_csv(file_var)
plt.imshow(result)
plt.show()


