import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv('./Advertising.csv')
data.head()
#print(data)
plt.scatter(data.TV,data.sales)
#plt.scatter(data.radio, data.sales)
plt.show()
# 取所有行，除去第一列和最后一列
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]  # 取最后一列
# 完成ax1+bx2+...+...这样一个动作
# Dense()输出多少个单元 这里是个隐藏层由自己选择，参数越大越强
# TV  radio  newspaper 长度为3
# activation='relu'添加激活函数
model = tf.keras.Sequential([tf.keras.layers.Dense(10,
                                                   input_shape=(3,),
                                                   activation='relu'),
                             tf.keras.layers.Dense(1)])
model.summary()
# test = data.iloc[:10, 1:-1]
# test_predict=model.predict(test)
# print(test_predict)