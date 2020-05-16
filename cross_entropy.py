import pandas as pd
# import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 库引用时如果没有被调用，是不能成功运行程序的，可以暂时注释掉
data = pd.read_csv('./credit-a.csv', header=None)  # 第一行没写参数，这样它就把第一行作为数据
data.head()
# print(data)
data.iloc[:, -1].value_counts()  # 取最后一列
# print(c)
x = data.iloc[:, :-1]  # iloc[]第一个参数表示取所有行，第二个参数表取最后一列之前的，这些都作为我们的数据
y = data.iloc[:, -1].replace(1, 0)  # 把-1替换成0,使其只有1和0，来辨别为欺诈数据和非欺诈数据
model = tf.keras.Sequential()  # 顺序模型
model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))  # 添加第一次隐藏层,设置4个单元数，数据的形状为前15行，然后relu激活
model.add(tf.keras.layers.Dense(4, activation='relu'))  # 添加第二层隐藏层，其他的就不用填了，它会自动判断
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 这个sigmoid他有两个隐藏层+一个输出层
model.summary()
# 编译配置
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 二元概率，计算交叉熵
              metrics=['acc']  # 在运行过程中，计算它的正确率情况，它是个list
              )
# 训练过程
history = model.fit(x, y, epochs=100)
history.history.keys()  # 字典
plt.plot(history.epoch, history.history.get('loss'))
plt.plot(history.epoch, history.history.get('acc'))

plt.show()