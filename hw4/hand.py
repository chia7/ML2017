from PIL import Image
import numpy as np
from keras.models import load_model
import sys

data = np.array([])
path = 'hand/hand.seq'
for i in range(1, 482):
	im = Image.open( path + str(i) + ".png" )
	im = np.array(im.convert('L'))[list(range(0, 480, 8)), :]
	im = im[:, list(range(0, 512, 8))]
	data = np.append(data, im.flatten().astype("float32") )

data = data.reshape( (481 ,3840) )

val, vec = np.linalg.eigh( np.cov(data.T) )
x_test = val[ val.shape[0]-60: ].reshape(1, 60)



model = load_model( sys.argv[1] )
ans = model.predict(x_test)

print(ans)

