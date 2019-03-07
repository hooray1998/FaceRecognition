#python -m pip install h5py
from __future__ import print_function
import os

#===============================================================
#    重要参数:
allImgPath = "./NLGAll" # 人脸集路径
testWeight = './model_weights.h5' # 测试权重 (重新训练和单张测试的时候有效)
trainWeight = './model_weights.h5' # 训练权重  (重新训练的时候有效)
Weidu = 3
nb_classes = 100  # 40个类别
epochs = 4  # 进行40轮次训
batch_size = 1  # 每次迭代训练使用40个样本
#===============================================================
choice_kernel_size = (3,3)
choice_Activation_1 = 'tanh'

# if choice_Pooling_size==(1,1) then do nothing.
choice_Pooling_size_1 = (6,6)
choice_Pooling_size_2 = (6,6)

number_Dropout_1 = 0.5
number_Dense     = 300
choice_Activation_2 = 'tanh'
number_Dropout_2 = 0.5
#===============================================================

# input image dimensions尺寸
img_rows, img_cols = 413, 579
# number of convolutional卷积 filters to use
nb_filters1, nb_filters2 = 5, 10  # 卷积核的数目（即输出的维度）
# size of pooling area for max pooling
nb_pool = 2
# convolution规模 kernel卷积核 size
# 代表 (3*3)
nb_conv = 3  # 单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。


import numpy as np
np.random.seed(
    1337
)  # for reproducibility  用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随即数都相同

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K


def getFileList(Dir):
    curList = os.listdir(Dir)
    allFileList = []
    for f in curList:
        if f == '.DS_Store':
            continue
        filePath = os.path.join(Dir,f)
        if os.path.isdir(filePath):
            allFileList.extend(getFileList(filePath))
        else:
            allFileList.append(filePath)
        
    return allFileList


def load_data(dataset_path):
    #NOTE : load all photo
    imgList = getFileList(dataset_path)
    imgList.sort()

    faces = np.empty((nb_classes*10,   Weidu*img_rows*img_cols))  # 400 * (57*47)

    for iterNum in range(len(imgList)):
        img = Image.open(imgList[iterNum])

        (x,y) = img.size #read image size

        # 调整大小
        if x!=img_rows or y!=img_cols:
            print(imgList[iterNum])
            img = img.resize((img_rows,img_cols),Image.ANTIALIAS) #resize image with high-quality
            img.save(imgList[iterNum])

        # 转化为灰度图像
        if Weidu == 1:
            img = img.convert('L')

        img_ndarray = np.asarray(
            img, dtype='float64') / 255  # asarray，将数据转化为np.ndarray，但使用原内存

        # print('shape:',img_ndarray.shape)

        if Weidu == 1:
            faces[iterNum] = np.ndarray.flatten(img_ndarray[0:img_cols,0:img_rows])
        else:
            faces[iterNum] = np.ndarray.flatten(img_ndarray[0:img_cols,0:img_rows,0:Weidu])
        # flatten将多维数组降为一维

    label = np.empty(nb_classes*10)
    for i in range(nb_classes):
        label[i * 10:i * 10 + 10] = i
    label = label.astype(np.int)

    #train:320,valid:40,test:40
    train_data = np.empty((nb_classes*8, Weidu*img_rows*img_cols))
    train_label = np.empty(nb_classes*8)
    valid_data = np.empty((nb_classes,    Weidu*img_rows*img_cols))
    valid_label = np.empty(nb_classes)
    test_data = np.empty((nb_classes,    Weidu*img_rows*img_cols))
    test_label = np.empty(nb_classes)

    for i in range(nb_classes):
    #WARN:
        train_data[i * 8:i * 8 + 8] = faces[i * 10:i * 10 + 8]  # 训练集中的数据
        train_label[i * 8:i * 8 + 8] = label[i * 10:i * 10 + 8]  # 训练集对应的标签
        valid_data[i] = faces[i * 10 + 8]  # 验证集中的数据
        valid_label[i] = label[i * 10 + 8]  # 验证集对应的标签
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')

    rval = [(train_data, train_label), (valid_data, valid_label),
            (test_data, test_label)]
    return rval

def testOne(oneImgPath):

    # load all className
    classFile = open('./face.classes','r')
    classList = []
    for name in classFile:
        classList.append(name)
    # print(classList)

    oneFace = np.empty((1,   Weidu*img_rows*img_cols))  # 400 * (57*47)
    img = Image.open(oneImgPath)
    (x,y) = img.size #read image size
    # 调整大小
    if x!=img_rows or y!=img_cols:
        print(imgList[iterNum])
        # WARN: 没有重新写入
        img = img.resize((img_rows,img_cols),Image.ANTIALIAS) #resize image with high-quality
        img.save(imgList[iterNum])

    # 转化为灰度图像
    if Weidu == 1:
        img = img.convert('L')
    img_ndarray = np.asarray(
        img, dtype='float64') / 255  # asarray，将数据转化为np.ndarray，但使用原内存

    # print('shape:',img_ndarray.shape)

    # flatten将多维数组降为一维
    if Weidu == 1:
        oneFace[0] = np.ndarray.flatten(img_ndarray[0:img_cols,0:img_rows])
    else:
        oneFace[0] = np.ndarray.flatten(img_ndarray[0:img_cols,0:img_rows,0:Weidu])

    oneFace = oneFace.astype('float32')
    oneLabel = np.empty(1)
    oneLabel[0] = 0
    oneLabel = oneLabel.astype(np.int)

    X_test = np.empty((1,    Weidu*img_rows*img_cols))
    y_test = np.empty(1)
    X_test[0] = oneFace[0]
    y_test[0] = oneLabel[0]

    print(nb_classes, end=' ')
    print(epochs, end=' ')
    print(batch_size, end=' ')
    print(choice_kernel_size, end=' ')
    print(choice_Activation_1, end=' ')
    print(choice_Pooling_size_1, end=' ')
    print(choice_Pooling_size_2, end=' ')

    print(number_Dropout_1, end=' ')
    print(number_Dense    , end=' ')
    print(choice_Activation_2, end=' ')
    print(number_Dropout_2)
    # the data, shuffled and split between tran and test sets

    if K.image_data_format() == 'channels_first':
        X_test = X_test.reshape(X_test.shape[0], Weidu, img_rows, img_cols)
        input_shape = (Weidu, img_rows, img_cols)
    else:
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, Weidu)
        input_shape = (img_rows, img_cols, Weidu)  # 1 为图像像素深度

    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = set_model()
    model.load_weights(testWeight)
    score = model.evaluate(X_test, Y_test, verbose=0)
    model.load_weights(testWeight)

    classes = model.predict_classes(X_test, verbose=0)
    test_accuracy = np.mean(np.equal(y_test, classes))
    print("accuarcy:", test_accuracy)
    print(y_test)
    print(classes)
    for i in range(0, 1):
        if y_test[i] != classes[i]:
            print(y_test[i], '被错误分成', classList[classes[i]])

def set_model(lr=0.005, decay=1e-6, momentum=0.9):
    model = Sequential()  #(连续的)
    if K.image_data_format() == 'channels_first':  # 通道1
        model.add(
            Conv2D(5, kernel_size=choice_kernel_size, input_shape=(Weidu, img_rows, img_cols)))
    else:
        model.add(
            Conv2D(5, kernel_size=choice_kernel_size, input_shape=(img_rows, img_cols, Weidu)))
    model.add(Activation(choice_Activation_1)) # sigmod ,relu,tanh
    #WARN:
    if choice_Pooling_size_1 != (1,1):
        model.add(MaxPooling2D(pool_size=choice_Pooling_size_1))
    model.add(Conv2D(10, kernel_size=choice_kernel_size))
    model.add(Activation(choice_Activation_2))
    #WARN:
    if choice_Pooling_size_2 != (1,1):
        model.add(MaxPooling2D(pool_size=choice_Pooling_size_2))
    model.add(Dropout(number_Dropout_1))
    model.add(Flatten())
    model.add(Dense(number_Dense))  #Full connection
    model.add(Activation(choice_Activation_2))
    #WARN:
    model.add(Dropout(number_Dropout_2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def train_model(model, X_train, Y_train, X_val, Y_val):
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, Y_val))
    model.save_weights(trainWeight, overwrite=True)
    return model


def test_model(model, X, Y):
    model.load_weights(trainWeight)
    score = model.evaluate(X, Y, verbose=0)
    return score


# if __name__ == '__main__':
def main_Function():
    print(nb_classes, end=' ')
    print(epochs, end=' ')
    print(batch_size, end=' ')
    print(choice_kernel_size, end=' ')
    print(choice_Activation_1, end=' ')
    print(choice_Pooling_size_1, end=' ')
    print(choice_Pooling_size_2, end=' ')

    print(number_Dropout_1, end=' ')
    print(number_Dense    , end=' ')
    print(choice_Activation_2, end=' ')
    print(number_Dropout_2)
    # the data, shuffled and split between tran and test sets

    global X_train,y_train,X_val,y_val,X_test,y_test

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], Weidu, img_rows, img_cols)
        X_val = X_val.reshape(X_val.shape[0], Weidu, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], Weidu, img_rows, img_cols)
        input_shape = (Weidu, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, Weidu)
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, Weidu)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, Weidu)
        input_shape = (img_rows, img_cols, Weidu)  # 1 为图像像素深度

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = set_model()
    train_model(model, X_train, Y_train, X_val, Y_val)
    score = test_model(model, X_test, Y_test)

    model.load_weights(trainWeight)
    classes = model.predict_classes(X_test, verbose=0)
    test_accuracy = np.mean(np.equal(y_test, classes))
    print("accuarcy:", test_accuracy)
    print(y_test)
    print(classes)
    for i in range(0, nb_classes):
        if y_test[i] != classes[i]:
            print(y_test[i], '被错误分成', classes[i])


if __name__ == '__main__':

#===============================================================
#    describe:  训练
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(allImgPath)
    main_Function()
#===============================================================
#    describe:  单张测试
#    Usage:输入图片的路径
    # path = input("Please input ImgPath: ")
    # while path != '-1':
        # testOne(path)
        # path = input("Please input ImgPath: ")

