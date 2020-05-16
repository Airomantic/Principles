import tensorflow as tf
#print('Tensorflow Version:{}'.format(tf.__version__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(train_image,train_lable),(test_image,test_label)=tf.keras.datasets.fashion_mnist.load_data()   #将fashion_mnist直接加进来
train_image.shape
train_lable.shape
train_image.shape,train_lable.shape
plt.imshow(train_image[0])
#plt.show()
#print(train_image[0])
#print(np.max(train_image[0]))
#print(train_lable)
#数据归一化
train_image=train_image/255
test_image =test_image/255
#print(train_image.shape)
#(60000, 28, 28)
model=tf.keras.Sequential()
#Dense是将一个一维的数据映射另一个一维数据上（张量为一维）
#而这个是二维的，需要flatten转化为一维的
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  #扁平成28*28的向量
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# #输出
# model.add(tf.keras.layers.Dense(10,activation='softmax')) #变成长度为10个概率值,softmax把它激活成概率分布
# #编译模型
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy', #顺序数字编码时使用
#               metrics=['acc'])     #度量正确率
# #训练
# model.fit(train_image,train_lable,epochs=5)
#评价
#model.evaluate(test_image,test_label)
#print(train_lable)
#独热编码
# beijing [1,0,0]
# shanghai [0,1,0]
# shenzhen [0,0,1]
train_lable_onehot=tf.keras.utils.to_categorical(train_lable)
# print(train_lable_onehot)
# # print(train_lable_onehot[-1])
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  #扁平成28*28的向量
model.add(tf.keras.layers.Dense(128,activation='relu'))
#0.5就是随机丢弃50%
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
#输出
model.add(tf.keras.layers.Dense(10,activation='softmax')) #变成长度为10个概率值,softmax把它激活成概率分布
model.summary()

#编译模型,注意学习率为0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', #顺序数字编码时使用
              metrics=['acc'])     #度量正确率
#训练，当可训练参数多提高epochs
#model.fit(train_image,train_lable_onehot,epochs=10)
model.fit(train_image,train_lable_onehot,epochs=10)
predict = model.predict(test_image)
print(test_image.shape)
predict.shape
print(predict[0])
print(np.argmax(predict[0]))    #预测最大值
#看看是不是最大值
print(test_label[0])