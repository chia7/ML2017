import numpy as np
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras.backend as K
import sys
import json


def f1score(y_true, y_pred):
	print(y_pred)
	THRES = 0.35
	y_pred = K.round(y_pred - THRES + 0.5)
	tp = K.sum(y_true*y_pred)
	fn = K.sum(y_true*(1.0-y_pred))
	fp = K.sum((1.0-y_true)*y_pred)
	tn = K.sum((1.0-y_true)*(1.0-y_pred))
	return 2.0*tp/(2.0*tp+fn+fp)




MAXLEN = 320


all_label = np.genfromtxt("rnn_label.txt", dtype="str")


x_test = [[]for i in range(1234)]
f = open( sys.argv[1], "r", encoding="utf8")
f.readline()
for i, line in enumerate( f.readlines() ):
	for k in range(10):
		if line[k] == ',':
			x_test[i] = text_to_word_sequence(line[k+1:], lower=True, split=" ")
			break

f.close()


tokenizer_word = Tokenizer()
f = open('dictionary.json', 'r', encoding="utf8")
tokenizer_word.word_index = json.load(f)
f.close()

for i in range(1234):
	x_test[i] = tokenizer_word.texts_to_sequences(x_test[i])

x_test = pad_sequences(x_test, maxlen = MAXLEN)

x_test = x_test.reshape( (1234, MAXLEN) )


model = load_model( "model_0.5051.h5", custom_objects={"f1score": f1score} )


ans = model.predict(x_test)
out = np.zeros(ans.shape)
out[ans>0.35] = 1


f = open( sys.argv[2], "w")
print('"id","tags"', file = f)
for i in range(1234):
	print('"', i, '"', sep='', end = ',', file = f)
	if out[i].sum() == 0:
		print('"', all_label[np.argmax(ans[i])], sep='', end = '"\n', file = f)
	else:
		tmp = '"'
		for w in range(38):
			if out[i, w] == 1:
				tmp += all_label[w]
				tmp += ' '
		print(tmp[:len(tmp)-1], end = '"\n', file = f)

f.close()






