
import numpy as np 
import struct 
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
    
    # normalize to 1
    return imgs/255.0


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

