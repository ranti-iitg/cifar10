from model import model

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
X_train = X_train.astype('float32')
X_train /= 255.0


print "hi"
h = model.fit(X_train,Y_train, verbose=1, batch_size=32,validation_split=0.1, nb_epoch=100,shuffle=True)
print model.get_weights()