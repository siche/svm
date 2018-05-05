# -*- coding: utf-8 -*-

# Hog + Linear Model
from mysvm import *

train_datas=load_image('Data/train_im.idx3-ubyte')
train_labels=load_label('Data/train_label.idx1-ubyte')
      
test_datas=load_image('Data/test_im.idx3-ubyte')
test_labels=load_label('Data/test_label.idx1-ubyte')

# 对所有数据进行Hog特征提取
train_num=np.size(train_datas,0)
test_num=np.size(test_datas,0)

bin_n=32
train_hist=np.zeros((train_num,4*bin_n),dtype=np.float32)
test_hist=np.zeros((test_num,4*bin_n),dtype=np.float32)

for i in range(train_num):
       c=hog(deskw(train_datas[i][:]))
       train_hist[i,:]=c

for k in range(test_num):
       test_hist[k][:]=hog(deskw(test_datas[k][:]))
       

# 测试集和训练集对调
# Linear       
model=svm()
t_t1=model.train(test_datas,test_labels)
t_res1, t_acc1=model.predict(train_datas,train_labels)

# Linear + Hog
model2=svm(c=2.67,gamma=5.382,kernel=1)
t_t2=model2.train(test_hist,test_labels)
t_res2, t_acc2 =model2.predict(train_hist,train_labels)