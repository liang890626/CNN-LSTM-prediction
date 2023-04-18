import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility 
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Input, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

#讀取數據
dataset = pd.read_csv('raw_1.csv')
dataset.index = dataset.Time
dataset.drop(columns = ['Time'], axis = 0, inplace = True)

#數值歸一化
scaler = MinMaxScaler()
dataset['Value'] = scaler.fit_transform(dataset['Value'].values.reshape(-1,1))
print(dataset.head)
dataset['Value'].plot(figsize=(16,8))


#功能函數
def create_new_dataset(dataset, seq_len = 12):
    x = []
    y = []
    start = 0 #初始位置
    end = dataset.shape[0] - seq_len #截止位置
    for i in range(start, end):
        sample = dataset[i : i+seq_len] #基於時間跨度seq_len建立樣本
        label = dataset[i+seq_len] #創建sample對應的標籤
        x.append(sample)
        y.append(label)
    return np.array(x), np.array(y)

#功能函數:基於新的特徵的數據集和標籤集，切分訓練及測試集
def split_dataset(x, y ,train_ratio = 0.67):
    x_len = len(x) #特徵值採集樣本x的樣本數量
    train_data_len = int(x_len * train_ratio) #訓練集樣本數量

    x_train = x[:train_data_len] #訓練集
    y_train = y[:train_data_len] #訓練標籤集

    x_test = x[train_data_len:]
    y_test = y[train_data_len:]
    return x_train, x_test, y_train, y_test

#原始數據集
dataset_original = dataset
print(" 原始數據集:", dataset_original.shape)
#構造特徵數據集和標籤集
SEQ_LEN = 28 #序列長度
x, y = create_new_dataset(dataset_original.values, seq_len = SEQ_LEN)
print(x.shape)
print(y.shape)
#樣本1 - 特徵數據
print(x[0])
print(y[0])
#數據切分
X_train , X_test, y_train, y_test = split_dataset(x, y, train_ratio=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import math

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional


# 定义模型
model = Sequential()

# 添加卷积层
model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# 添加LSTM层
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=128))
model.add(Dropout(0.3))

# 添加全连接层
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

#計算R2
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("r2:", score)
RMSE = math.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE: " , RMSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE: " , MAE)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE: " , MAPE)



y_true = y_test[:40000]
y_pred = y_pred[:40000]
plt.figure(figsize=(20,8))
plt.plot(y_true, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()
