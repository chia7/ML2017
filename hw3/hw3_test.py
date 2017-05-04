import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import sys
from keras.utils import plot_model
import scipy

def histeq(data):

	for i in range( int(len(data)) ):
		imhist, bins = np.histogram(data[i], 255, normed = True)
		cdf = imhist.cumsum()
		cdf = 255 * cdf / cdf[-1]

		data[i] = np.interp(data[i], bins[:-1], cdf)

	return data


def load_data():

	f = open( sys.argv[1], 'r' )
	data = []
	for line in f.readlines():
		data.append(line.replace(',', ' ').split())
	data = np.array(data[1:]).astype('float32')

	x_test = data[:, 1:]
	x_test = histeq(x_test)

	return x_test


x_test = load_data()


x_test = x_test.reshape( (7178, 48, 48, 1) )



model = load_model('model_1.h5')
p1 = model.predict(x_test)

model = load_model('model_2.h5')
p2 = model.predict(x_test)

model = load_model('model_3.h5')
p3 = model.predict(x_test)

model = load_model('model_4.h5')
p4 = model.predict(x_test)


y_prob = 25*p1 + 109*p2 + 50*p3 + 2*p4
y_pred = np.argmax(y_prob, axis=1)


f = open( sys.argv[2], "w")
print("id,label", file = f)
for i in range(y_pred.size):
	print(str(i) + "," + str(y_pred[i]), file = f)
f.close()







