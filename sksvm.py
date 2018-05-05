# !bin/python
# encoding = utf-8
import numpy as np 
import time
from sklearn import svm

class sksvm():

    # kernel can be the following string
    # linear
    # rbf
    # poly
    # sigmoid

    def __init__(self,c=1.0, gamma=0.01, kernel='rbf'):
        self.model = svm.SVC(kernel=kernel, gamma=1, C=c)
        self.acc = 0

    def train(self, data, label):
        t1 = time.time()
        self.model.fit(data,label)
        predict_label = self.model.predict(data)
        accuracy = (predict_label == label)
        accuracy = float(sum(accuracy)/(np.size(label,0)))
        self.acc = accuracy
        t2 = time.time()
        train_time = t2-t1
        print('train time is %s s\n' %train_time)
        
    
    def predict(self,data):
        label = self.model.predict(data)
        return label
    
def search_paramter(data, label, carr, garr, kernel):
    carr =  np.array(carr)
    m = np.size(carr,0)
    n = np.size(garr,0)
    acc = 0
    best_c = 0.
    best_g = 0.
    for i in range(m):
        for j in range(n):
            ptg = 100*((i-1)*n+j)/(m*n)
            print('searching... %f %%' % ptg)

            model = sksvm(carr[i],garr[i],kernel)
            model.train(data,label)
            acc1 = model.acc
            if acc1 >= acc:
                acc = acc1
                print('acc is %f %%' % acc)
                best_c = carr[i]
                best_g = garr[j]  
    return best_c, best_g

