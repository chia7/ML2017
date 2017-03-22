import sys
import numpy as np
import random

AMB_TEMP = 0;	CH4 = 1;	CO = 2;			NMHC = 3;			NO = 4;				NO2 = 5
NOx = 6;		O3 = 7;		PM10 = 8;		PM25 = 9;			RAINFALL = 10;		RH = 11;
SO2 = 12;		THC = 13;	WD_HR = 14;		WIND_DIREC = 15;	WIND_SPEED = 16;	WS_HR = 17;

train_feature = np.array( [ [PM25, 2], [O3, 2], [PM10, 2], [RAINFALL, 1], [WD_HR, 1], [WIND_DIREC, 1], [WIND_SPEED, 1], [WS_HR, 1], [CO, 2] ] )
num_feature = int( np.sum( train_feature[:, 1] ) )

# fold = 9
# it = 200000
# lr = 1.25e-1
# lamda = 0

def read_train_data(train_feature, num_feature):
	raw_data = np.genfromtxt( sys.argv[1], delimiter = ',' )
	raw_data = np.delete( raw_data, [0], 0 )
	raw_data = np.delete( raw_data, [0, 1, 2], 1 )
	raw_data = np.nan_to_num( raw_data )

	######################################################################
	### get pm2.5 (actual value)

	pm25 = np.array([])			# shape = (1, 5652)
	for month in range(12):
		for day in range(20):
			if day == 0:
				pm25 = np.append( pm25, raw_data[ month*360 + 9, 9: ] )
			else:
				pm25 = np.append( pm25, raw_data[ month*360 + day*18 + 9] )

	######################################################################
	### get selected feature (from the train_feature list)

	feature = np.array([])
	normalize = np.array([])
	for i in range( int( len(train_feature) ) ):
		tmp = np.array([])
		for day in range(240):
			tmp = np.append( tmp, raw_data[ day*18 + train_feature[i][0] ] )

		##################################################################
		### feature scaling (normalization)	x' = (x - mean) / standard deviation			<<---- normalization
		mean = np.mean(tmp)
		sd = np.std(tmp)
		# tmp = (tmp - mean) / sd
		normalize = np.append( normalize, [mean, sd] )

		for order in range( int(train_feature[i][1]) ):		#								<<---- order
			feature = np.append( feature, tmp**(order+1) )

	feature = feature.reshape( ( num_feature, 5760) )
	normalize = normalize.reshape( ( int( len(train_feature) ), 2 ) )

	######################################################################
	### get all training data ready

	out = np.array([])
	for month in range(12):
		for hour in range(471):
			out = np.append( out, pm25[ month*471 + hour ] )
			for num in range(num_feature):
				out = np.append( out, feature[ num, month*480 + hour : month*480 + hour + 9] )

	out = out.reshape( ( 5652, num_feature*9 + 1 ) )


	return ( out, normalize )

def predit(train_feature, num_feature, normalize, b, w):
	test = np.genfromtxt( sys.argv[2], delimiter = ',' )
	test = np.nan_to_num( test )
	test = np.delete( test, [0, 1], 1 )

	feature = np.array([])
	for num in range(240):
		tmp_1 = np.array([])
		for i in range( int( len(train_feature) ) ):
			tmp_2 = np.array([])
			tmp_2 = np.append( tmp_2, test[ num*18 + train_feature[i][0] ] )

			# tmp_2 = ( tmp_2 - normalize[i][0] ) / normalize[i][1]

			for order in range( int(train_feature[i][1]) ):
				tmp_1 = np.append( tmp_1, tmp_2**(order+1) )

		feature = np.append( feature, tmp_1 )

	feature = feature.reshape( ( 240, num_feature*9 ) )

	ans = np.dot(feature, w) + b

	f = open( sys.argv[3], 'w' )
	print( 'id,value', file = f )
	for i in range(240):
		print('id_', i, ',', round(ans[i]), sep = '', file = f)
	f.close()


( out, normalize ) = read_train_data( train_feature, num_feature )

model = np.genfromtxt( sys.argv[4])
b = model[0]
w = model[1:]


predit(train_feature, num_feature, normalize, b, w)