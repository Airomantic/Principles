import tensorflow as tf
#print('Tensorflow Version:{}'.format(tf.__version__))
#Tensorflow Version:1.3.0-rc0
import pandas as pd
data = pd.read_csv('./Income1.csv')
#print(data)
import matplotlib.pyplot as plt
#%matplotlib inline 只在jupyter notebook中用到
plt.scatter(data.Education,data.Income)
plt.show() #做出图像
x=data.Education
y=data.Income
model=tf.keras.Sequential() #初始化模型
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))   #添加层，Dense第一个参数是输出，第二个参数是输入数据的形状，纬度是1
model.summary() #ax+b
#训练 均方差
model.compile(optimizer='adam',
              loss='mse')

history=model.fit(x,y,epochs=5000)  #epochs把所以数据进行训练
model.predict(pd.Series([20])) #Series这样的格式
#梯度下降法 ：是一种致力于找到函数极值点的算法
#1.前面的“学习（机器学习）”是改进模型参数，以便通过大量的训练步骤将损失最小化
#2.将梯度下降法应用于寻找损失函数的极值点便构成了依据输入数据的模型学习