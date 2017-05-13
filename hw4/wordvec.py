import word2vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text
import nltk

corpus_path = 'Book5TheOrderOfThePhoenix/all.txt'
model_path = 'model.bin'

WORDVEC_DIM = 300
WINDOW = 7
NEGATIVE_SAMPLES = 5
ITERATIONS = 5
MIN_COUNT = 10
LEARNING_RATE = 0.025
MODEL = 1

word2vec.word2vec(corpus_path, model_path, size=WORDVEC_DIM, window=WINDOW, negative=NEGATIVE_SAMPLES, iter_=ITERATIONS, min_count=MIN_COUNT, alpha=LEARNING_RATE, cbow=MODEL, verbose=True)

model = word2vec.load(model_path)

plot_num = 777

vocabs = []                 
vecs = []                   
for vocab in model.vocab:
	vocabs.append(vocab)
	vecs.append(model[vocab])
vecs = np.array(vecs)[:plot_num]
vocabs = vocabs[:plot_num]



tsne = TSNE(n_components=2, method='exact')
reduced = tsne.fit_transform(vecs)



use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = [ '“', ',', '.', ':', ';', '’', "!", "?", '”', '"', "'" ]


plt.figure(figsize=(18,12))
texts = []
for i, label in enumerate(vocabs):
	pos = nltk.pos_tag([label])
	if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags and all(c not in label for c in puncts)):
		x, y = reduced[i, :]
		texts.append(plt.text(x, y, label))
		plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

plt.savefig('visualization.png', dpi=600)
# plt.show()








