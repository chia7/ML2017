import numpy as np
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras.backend as K

import sys
import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")


stopword = stopwords.words("english")
lmtzr = WordNetLemmatizer()




x_test = [[]for i in range(1234)]
f = open( sys.argv[1], "r", encoding="utf8")
f.readline()
for i, line in enumerate( f.readlines() ):
	for k in range(10):
		if line[k] == ',':
			x_test[i] = line[k+1:]
			x_test[i] = re.sub("[^a-zA-Z]", " ", x_test[i])
			x_test[i] = text_to_word_sequence(x_test[i], lower=True, split=" ")
			x_test[i] = [ w for w in x_test[i] if w not in stopword ]
			x_test[i] = [ lmtzr.lemmatize(w) for w in x_test[i] ]
			x_test[i] = " ".join(x_test[i])
			break

f.close()


tokenizer_word = pickle.load(open("token", "rb"))


x_test = tokenizer_word.texts_to_matrix(x_test, mode="tfidf")


all_label = np.genfromtxt("label.txt", dtype="str")




def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))



model = load_model( "best.h5", custom_objects={"f1_score": f1_score} )
ans = model.predict(x_test)
out = np.zeros(ans.shape)
out[ans > 0.5] = 1


f = open( sys.argv[2], "w" )
print('"id","tags"', file = f)
for i in range(1234):
	print('"', i, '"', sep='', end = ',', file = f)
	if out[i].sum() == 0:
		print('""', file = f)
	else:
		tmp = '"'
		flag = 0
		for w in range(38):
			if out[i, w] == 1:
				tmp += all_label[w]
				tmp += ' '
				flag = 1
		if flag == 1:
			print(tmp[:len(tmp)-1], end = '"\n', file = f)
		else:
			print(tmp, end = '"\n', file = f)
f.close()
