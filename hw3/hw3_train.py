import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import sys



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

	data = np.delete(data, 59, 0)		# dirty data

	x_train = data[:, 1:]
	x_train = histeq(x_train)

	y_train = data[:, 0]
	y_train = np_utils.to_categorical(y_train, 7)


	return (x_train, y_train)


(x_train, y_train) = load_data()



num = x_train.shape[0]
x_train = x_train.reshape( (num, 48, 48, 1) )		# 28709 = 19×1511

x_train = np.append(x_train, x_train[:, :, ::-1], axis = 0)
y_train = np.append(y_train, y_train, axis = 0)




model = Sequential()



model.add( Convolution2D( 256, (5, 5), activation = 'elu', input_shape = (48, 48, 1) ) )
model.add( MaxPooling2D( (2, 2) ) )
model.add(BatchNormalization())
model.add( Dropout(0.5) )

model.add( Convolution2D( 512, (3, 3), activation = 'elu' ) )
model.add( MaxPooling2D( (2, 2) ) )
model.add(BatchNormalization())
model.add( Dropout(0.55) )

model.add( Convolution2D( 1024, (3, 3), activation = 'elu' ) )
model.add( MaxPooling2D( (2, 2) ) )
model.add(BatchNormalization())
model.add( Dropout(0.6) )

model.add( Convolution2D( 2048, (3, 3), activation = 'elu' ) )
model.add( MaxPooling2D( (2, 2) ) )
model.add(BatchNormalization())
model.add( Dropout(0.65) )


model.add( Flatten() )

model.add( Dense(output_dim = 2048) )
model.add( Activation('elu') )
model.add( Dropout(0.7) )

model.add( Dense(output_dim = 7) )
model.add( Activation('softmax') )

model.compile( loss = 'categorical_crossentropy', optimizer = 'Adam', metrics=['accuracy'] )


datagen = ImageDataGenerator(
	width_shift_range = 0.1,
	height_shift_range = 0.1,
	rotation_range = 10
)


datagen.fit(x_train)

model.summary()




model.fit_generator( datagen.flow(x_train, y_train, batch_size=256), steps_per_epoch=len(x_train) / 256, epochs = 500 )

model.save( "new_model.h5" )







