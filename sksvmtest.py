import numpy as np  
import matplotlib.pyplot as plt 
from sklearn import svm, datasets
from load_data import *
from sksvm import *
# load data
iris = datasets.load_iris()
data1 = iris.data 
label1 = iris.target
n1 = len(label1)
train_num = round(50*2/3)

choice = np.random.permutation(np.arange(0,50))
train1 = choice[:train_num]
test1 = choice[train_num:] 

'''
# train_data1 = np.vstack((data1[train1],data1[train1+50],\
                        data1[train1+100]))

# train_label1 = np.hstack((label1[train1],label1[train1+50],\
                        label1[train1+100]))

# test_data1 = np.vstack((data1[test1],data1[test1+50],\
                        data1[test1+100]))
# test_label1 = np.hstack((label1[test1],label1[test1+50],\
                        label1[test1+100]))
'''


data2 = load_image('Data/train_im.idx3-ubyte')
label2 = load_label('Data/train_label.idx1-ubyte')
label2 = label2.reshape((-1,))
n2 = max(label2.shape)

train_num2 = round(n2*2/3)
arr = np.arange(0,n2)
arr = np.random.permutation(arr)
train_data2 = data2[arr[:train_num2]]
train_label = label2[arr[:train_num2]]

test_data2 = data2[arr[train_num2:]]
test_label2 = label2[arr[train_num2:]]

carr = np.linspace(0.001,2,10)
garr = np.linspace(0.001,2,10)

best_c, best_g = search_paramter(test_data2, test_label2, carr, garr, kernel='rbf')
model = sksvm(best_c, best_g, kernel='rbf')
label_prd = model.predict(test_data2)
result = (label_prd == test_label2)
acc = 100*sum(result)/np.size(test_label2,0)
print ('acc is %s %%' %acc)

