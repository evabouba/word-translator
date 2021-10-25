import numpy as np
from scipy.spatial.distance import cdist

def get_vectors(lines, word_min_len):
	vectors = []
	words = []
	for line in lines:
		v = line.rstrip().split(' ')[1:]
		w = line.rstrip().split(' ')[0]

		if(len(w) >= word_min_len):
			vectors.append(v)
			words.append(w)
   
	vectors = np.array(vectors).astype("float32")
	words = np.array(words)
 
	return vectors, words


def metric(Wx, y, k):
	cos_sim = 1 - cdist(Wx, y, metric="cosine")
	mean_sim1 = np.reshape(np.mean(np.sort(cos_sim, axis=1)[:,-k:], axis=1), (-1,1))
	mean_sim2 = np.mean(np.sort(cos_sim, axis=0)[-k:,:], axis=0)
	
	sim = 2 * cos_sim - mean_sim1 - mean_sim2

	i = np.argmax(sim, axis=1)

	return np.mean(cos_sim[range(len(i)), i])
