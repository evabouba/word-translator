from translator import Translator, SupervisedTranslator, UnsupervisedTranslator
import matplotlib.pyplot as plt

def supervised(source, target, dictionary=None, method="MGD"):
	trslt = SupervisedTranslator(source, target, dictionary)

	
	s = "\nSupervised learning by "+str(method)+ " using "+str(len(trslt.dictionary))
	if dictionary is None:
		print(s, " identical words found")
	else:
		print(s, "translations given")


	h = trslt.fit(method=method)

	if(method=="MGD"):
		plt.plot(h.history["cosine_similarity"], label="cosine_similarity")
		plt.legend()
		plt.savefig("MGD.jpg")
	else:
		print("Mean cosine distance: ", h)

	return trslt


def unsupervised(source, target):
	trslt = UnsupervisedTranslator(source, target)

	print("Unsupervised learning by GAN")
	h = trslt.fit()

	plt.plot(h[2], label="Metric")
	plt.legend()
	plt.savefig("GAN.jpg")


	return trslt
