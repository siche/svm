# -*- coding:utf-8 -*-
from cv2 import moments
import cv2
import struct
import numpy as np
import time


def load_image(filename):
    print ("load image set")
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print ("head,", head)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum*width*height
    bitsString = '>'+str(bits)+'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width*height])
    print ("load imgs finished")
    return np.float32(imgs/255.0)


def load_label(filename):
    print ("load label set")
    binfile = None
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print ("head,", head)
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>'+str(imgNum)+"B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    # print labels
    print ('load label finished')
    return labels


class svm():
    def __init__(self, c=10.0, gamma=0.01, kernel=1):

        # kernel
        # kernel=1   LINEAR
        # nernel=2   RBF
        # kernal=3

        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)  # SVM类型
        if kernel == 1:
            self.model.setKernel(cv2.ml.SVM_LINEAR)
            print ('Linear model is built')
        else:
            self.model.setKernel(cv2.ml.SVM_RBF)
            print ('Rbf model is built')
        self.model.setC(c)
        self.model.setGamma(gamma)

        print ('New svm model has been built')

    def train(self, train_data, train_label):
        # train the model and return train time
        t1 = time.time()
        print ('The Svm is training ... \n')
        self.model.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
        t2 = time.time()
        print ('Cost time is %s s \n' % (t2-t1))
        return (t2-t1)

    def save(self, name):
        #path='D:/OneDrive - sjtu.edu.cn/Desktop/number_svm/'
        full_name = name
        self.model.save(full_name)

    def load(self, name):
        #path='D:/OneDrive - sjtu.edu.cn/Desktop/number_svm/'
        full_name = name
        self.model = cv2.ml.SVM_load(full_name)

    def predict(self, test_data, test_label):
        [res, result] = self.model.predict(test_data)
        accuracy = result == test_label
        acc = sum(accuracy)
        acc = float(acc)/(np.size(test_label, 0))
        print ('accuracy is %s' % acc)
        return result, acc

    def test(self, result, test_label):
        accuracy = result == test_label
        acc = sum(accuracy)
        acc = float(acc)/(np.size(test_label, 0))
        print ('accuracy is %s' % acc)
        return acc, accuracy


def grid_search(c, gamma, train_data, train_label, test_data, test_label, kernel=1):

    s1 = np.size(c, 0)
    s2 = np.size(gamma, 0)
    acc = np.zeros((s1, s2))

    for i in range(s1):
        for k in range(s2):
            print ('c=%s, gamma=%s' % (c[i], gamma[k]))

            model = svm(c[i], gamma[k], kernel)
            model.train(train_data, train_label)
            result, acc[i][k] = model.predict(test_data, test_label)
            print ('%s%% finshed' % ((i*s2+k+1)/(s1*s2/100.0)))

    # name='gamma=%s_c=%s.xml' % (c(k),gamma(i))
    best_para = np.amax(acc)
    for p in range(s1):
        for q in range(s2):
            if acc[p][q] == best_para:
                break
    best_c = c[p]
    best_g = gamma[q]
    np.save('acc_search.npy', acc)
    return best_c, best_g


SZ = 28
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


def deskw(img):
    img = np.reshape(img, [28, 28])
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


def hog(img):
    # Define Sobel derivatives

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bin_n = 31
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:14, :14], bins[14:, :14], bins[:14, 14:], bins[14:, 14:]
    mag_cells = mag[:14, :14], mag[14:, :14], mag[:14, 14:], mag[14:, 14:]
    # bincount(x,weight)
    # return_val[n]=number(x=n) ,如果有wieght的话对应的转化为+weight

    hists = [np.bincount(b.ravel(), m.ravel(), (bin_n+1))
             for b, m in zip(bin_cells, mag_cells)]
    # reval(x,order='C'(F,K,A) 按照column, row等reshape成向量
    # bincout 中权重是 mag_cells 最小长度是 bin_cell
    hist = np.hstack(hists)
    hist = np.array(hist, dtype=np.float32)
    # 化成行向量
    return hist
