from scipy.spatial.distance import cdist
from utils import get_vectors
import numpy as np

class Language:
	def __init__(self, name, file_path):
		self.name = name
		
		f = open(file_path, "r")
		lines = f.readlines(int(1e8))[1:]
		f.close()
 
		self.vec, self.voc = get_vectors(lines, 2)
		self.embd_size = self.vec.shape[1]
		self.voc_size = self.vec.shape[0]
		
 
	def most_similar(self, positive, negative, n=1):
		v = np.zeros((1,self.embd_size))
 
		for neg in negative:
			i = np.where(self.voc == neg)[0]
			if len(i)==0:
				print("Unkown word: ", neg)
				return
				
			v -= self.vec[i]
 
		for pos in positive:
			i = np.where(self.voc == pos)[0]
			if len(i) == 0:
				print("Unkown word: ", pos)
				return

			v += self.vec[i]
 
		distances = cdist(v, self.vec, metric="cosine")[0]
		indices = np.argsort(distances)[:n]
		d = [(self.voc[i], distances[i]) for i in indices]
 
		return d, v
