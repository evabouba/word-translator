from scipy.linalg import svd
from scipy.spatial.distance import cdist
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LeakyReLU, Input, GaussianNoise
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy

from language import Language
from utils import metric

class Translator:
	def __init__(self, source_language, target_language):
		self.source = source_language
		self.target = target_language
		self.W = np.zeros((self.target.embd_size, self.source.embd_size))



	def process(self, dictionary):
	# dictionary : data frame with a column for each language and a line for each words pair
 
		X = []
		y = []
		dic = {}
 
		if dictionary is None: #Construction of dictionary with only identical words pair
			inverse_index = { element: index for index, element in enumerate(self.target.voc) }
			for index, element in enumerate(self.source.voc):
				if element in inverse_index:
					X.append(self.source.vec[index])
					y.append(self.target.vec[inverse_index[element]])
					dic[element] = [self.target.voc[inverse_index[element]]]
		else:
			for i in range(dictionary.shape[0]):
				source_word = dictionary.iloc[i,:][self.source.name]
				target_word = dictionary.iloc[i,:][self.target.name]
				i1 = np.where(self.source.voc == source_word)[0]
				i2 = np.where(self.target.voc == target_word)[0]
 
				if len(i1)*len(i2) > 0: # both words are in their respective language vocab
					v1 = self.source.vec[i1[0]]
					v2 = self.target.vec[i2[0]]
 
					X.append(v1)
					y.append(v2)
 
					if self.source.voc[i1[0]] not in dic.keys():
						dic[source_word] = []
					dic[source_word].append(target_word)


		return np.array(X), np.array(y), dic




	def translate(self, words, p=1):
		# words : list of strings to be translated
		# p : precision
		vectors = np.zeros((len(words), self.source.embd_size))

		for i, w in enumerate(words):
			j = np.where(self.source.voc == w)[0]
			if len(j) > 0: 
				vectors[i] = self.source.vec[j]
			else: # w is not in the vocab
				vectors[i] = np.repeat(np.NaN, self.source.embd_size)
 
		trslt_vec = np.dot(vectors, self.W)

		distances = cdist(trslt_vec, self.target.vec, metric="cosine") 
		indices = np.argsort(distances, axis=1)[:,:p]
		translations = self.target.voc[indices]

		return [(words[i], list(translations[i])) for i in range(len(words))]


	def evaluate(self, dictionary, precisions=[1]): 

		# dictionary : dataframe with words to be used to evaluate translations
		# precisions : list of evaluation precisions

		acc = {p:0 for p in precisions}

		X,y,d = self.process(dictionary)
		translations = self.translate(list(d.keys()), max(precisions))
		n = 0
		for t in translations:
			if len(t) > 0: # the word is in both language vocab
				possibilities = set(d[t[0]]) # possible translations according to the dictionary
				n +=1

				for p in precisions:
					predictions = set(t[1][:p]) # obtaines translations with a certain precision

					if len(predictions.intersection(possibilities)) > 0 :
						acc[p] += 1

		return {p:acc[p]/n for p in acc.keys()}


	def evaluate_2(self, W, X, Y, split=1):
		# W : translation matrix
		# X, Y embeddings such that y_i is the traduction of x_i
		# Return the partial (split) mean cosinus distance between Wx and y
 
		n_vectors = int(split * X.shape[0])
		indices = np.random.choice(range(X.shape[0]), n_vectors, False)
		trslt_vec = np.dot(X[indices], W)

		mean_cos_dis= np.mean(np.diag(cdist(trslt_vec, Y[indices], metric="cosine")))
 
		return mean_cos_dis


    def save_mapping(self,path):
		with open(path, "wb") as f:   
			pickle.dump(self.W, f)
            
            
class SupervisedTranslator(Translator):
	def __init__(self, source_language, target_language, dictionary=None):
		super(SupervisedTranslator, self).__init__(source_language, target_language)
		self.X, self.y, self.dictionary = self.process(dictionary)



	
	def fit(self, method='MGD', step_size = 0.1, batch_size = 4, epochs = 60, verbose=1, n_refinement=5):

		if method=='MGD':
			nn = Sequential()
			nn.add(Dense(self.target.embd_size, activation="linear", input_shape=(self.source.embd_size,), use_bias=False))
			
			nn.compile(loss="MSE", optimizer=SGD(step_size), metrics=["cosine_similarity"])

			h = nn.fit(self.X, self.y, batch_size=batch_size, epochs=epochs, verbose=verbose)
			self.W = nn.get_weights()[0]

			return h

		elif method=='procrustes':
			best_eval_score= np.inf
			for i in range(n_refinement):
				M = np.dot(self.X.T, self.y)
				U, S , Vt = svd(M)
				W = U.dot(Vt)
				eval_score=self.evaluate_2(W, self.X, self.y, 0.4)
				if eval_score < best_eval_score:
					best_eval_score = eval_score
					self.W = W

			return best_eval_score

		else:
			print("Unknown method")



class UnsupervisedTranslator(Translator):
	def __init__(self, source_language, target_language):
		super(UnsupervisedTranslator, self).__init__(source_language, target_language)

	def get_discriminator(self, learning_rate):
		inputs = Input(shape=(self.target.embd_size,))
		discriminator = Dropout(0.1)(inputs)
		discriminator = GaussianNoise(0.1)(discriminator)
		discriminator = Dense(2048)(discriminator)
		discriminator = LeakyReLU(0.2)(discriminator)
		discriminator = Dense(2048)(discriminator)
		discriminator = LeakyReLU(0.2)(discriminator)
		outputs = Dense(1, activation="sigmoid")(discriminator)
		model = Model(inputs, outputs)
		model.compile(loss=BinaryCrossentropy(label_smoothing=0.2), optimizer=SGD(learning_rate=learning_rate), metrics=["accuracy"])

		return model

	def get_generator(self):
		inputs = Input(shape=(self.source.embd_size,))
		outputs = Dense(self.target.embd_size, use_bias=False)(inputs)
		model = Model(inputs, outputs)
  
		return model

	def get_gan(self,generator, discriminator, learning_rate):
		discriminator.trainable = False
		inputs = Input(shape=(generator.input_shape[1],))
		outputs = discriminator(generator(inputs)) 
		model = Model(inputs, outputs)
		model.compile(loss=BinaryCrossentropy(), optimizer=SGD(learning_rate=learning_rate), metrics=["accuracy"])  

		return model
  
  
	def fit(self, epochs=100, batch_size=50, learning_rate=0.1, learning_decay=0.0008, discriminator_steps=1, verbose=1):
		# learning_decay : each epoch lr <- lr / (1 + learning_decay)
		# discriminator_steps : number of discriminator trainings for one generator training
		generator = self.get_generator()
		discriminator = self.get_discriminator(learning_rate)
		gan = self.get_gan(generator, discriminator, learning_rate)

		self.generator = generator
		self.discriminator = discriminator
		self.gan = gan

		half_batch = int(batch_size / 2)
		n_steps = int(int(self.source.voc_size/batch_size)/2)

		d_history={'loss':[], 'acc':[]}
		g_history={'loss':[], 'acc':[]}
		m_history=[]
		m = 0
		m_decay = 0

		for e in range(epochs):
			for i in range(n_steps):
				# Discriminator training(s)
				for j in range(discriminator_steps):
					x = self.source.vec[np.random.choice(range(self.source.voc_size), half_batch, False),:]
					generated_vectors = generator.predict(x)
					real_vectors = self.target.vec[np.random.choice(range(self.target.voc_size), half_batch, False),:]

					X_d= np.concatenate([generated_vectors, real_vectors])
					y_d=np.ones(2*half_batch)
					y_d[:half_batch]=0 
					d_loss, d_acc = discriminator.train_on_batch(X_d, y_d)

				# Generator training
				x = self.source.vec[np.random.choice(range(self.source.voc_size), batch_size, False),:]
				g_loss, g_acc = gan.train_on_batch(x,np.repeat(1, batch_size)) 

				# Orthogonalization 
				w = generator.get_weights()[0]
				beta = 0.01
				generator.set_weights([(1+beta) * w - beta * w.dot(w.T).dot(w)])

			self.W = generator.get_weights()[0]

			m2 = metric(np.dot(self.source.vec[:1000,:], self.W), self.target.vec[:1000,:], 10)

			if(m2 < m):
				if m_decay < 2:
					m_decay += 1
				else: # Dividing the learning_rate after two worsenings of the metric
					m_decay = 0
					old_lr = keras.backend.get_value(gan.optimizer.learning_rate) 
					new_lr = max(old_lr / 1.15, 0.00001)
					keras.backend.set_value(gan.optimizer.learning_rate, new_lr)  
					keras.backend.set_value(discriminator.optimizer.learning_rate, new_lr)


 			# Learning_rate decay
			old_lr = keras.backend.get_value(gan.optimizer.learning_rate) 
			new_lr = max(old_lr / (1 + learning_decay), 0.00001) # 0.00001 is the minimum learning_rate we won't get below
			keras.backend.set_value(gan.optimizer.learning_rate, new_lr)
			keras.backend.set_value(discriminator.optimizer.learning_rate, new_lr)

			m = m2

			d_history['loss'].append(d_loss)
			d_history['acc'].append(d_acc)
			g_history['loss'].append(g_loss)
			g_history['acc'].append(g_acc)
			m_history.append(m)

			if verbose:
				print("Epoch: ", e+1)

				print("Metric: ", round(m,3),
					"D_loss: ", round(d_loss,3), 
					", D_acc: ", round(d_acc,3),  
					", G_loss: ", round(g_loss,3),
					", G_acc: ", round(g_acc,3), "\n\n")

		return d_history, g_history, m_history, acc_history
