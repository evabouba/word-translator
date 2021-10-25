from src.language import Language
import pandas as pd
import matplotlib.pyplot as plt
from src.train import supervised, unsupervised
import pandas as pd
import subprocess

subprocess.call("./data/Embeddings/fetch_data.sh", shell=True)

print("Uploading Embeddings ")
fr = Language("french", "./data/Embeddings/wiki.fr.vec")
en = Language("english", "./data/Embeddings/wiki.en.vec")
#es = Language("spanish", "./data/Embeddings/wiki.es.vec")
#de = Language("german", "./data/Embeddings/wiki.de.vec")


print("\nUploading Dictionnaries ")
fr_en_dic = pd.read_csv('./data/Dictionaries/fr-en.0-5000.txt', sep=" ", header=None, names=["french", "english"], na_filter= False) # Space
fr_en_test_dic = pd.read_csv('./data/Dictionaries/fr-en.5000-6500.txt', sep=" ", header=None, names=["french", "english"], na_filter= False) # Space
#en_fr_dic = pd.read_csv('./data/Dictionaries/en-fr.0-5000.txt', sep=" ", header=None, names=["english", "french"], na_filter= False) # Space
#en_fr_test_dic = pd.read_csv('./data/Dictionaries/en-fr.5000-6500.txt', sep=" ", header=None, names=["english", "french"], na_filter= False) # Space
#fr_es_dic = pd.read_csv('./data/Dictionaries/fr-es.0-5000.txt', sep="	", header=None, names=["french", "spanish"], na_filter= False) # Tab
#fr_es_test_dic = pd.read_csv('./data/Dictionaries/fr-es.5000-6500.txt', sep="	", header=None, names=["french", "spanish"], na_filter= False) # Tab
#fr_de_dic = pd.read_csv('./data/Dictionaries/fr-de.0-5000.txt', sep="	", header=None, names=["french", "german"], na_filter= False) # Tab
#fr_de_test_dic = pd.read_csv('./data/Dictionaries/fr-de.5000-6500.txt', sep="	", header=None, names=["french", "german"], na_filter= False) # Tab


print("\nTranslators initialization")
trslt1 = supervised(fr, en, fr_en_dic, "MGD")
trslt2 = supervised(fr, en, fr_en_dic, "procrustes")
trslt3 = supervised(fr, en, None, "procrustes")
#trslt_un = unsupervised(fr, en)


print("\nTest accuracies:")
print("MGD on dictionary: ", trslt1.evaluate(fr_en_test_dic, [1, 5, 10]))
print("Procrustes on dictionary: ", trslt2.evaluate(fr_en_test_dic, [1, 5, 10]))
print("Procrustes on identical words: ", trslt3.evaluate(fr_en_test_dic, [1, 5, 10]))
#print("GAN: ", trslt_un.evaluate(fr_en_test_dic, [1, 5, 10]))


print("Exemples of translations: ")
print("MGD on dictionary: ", trslt1.translate(["homme", "femme", "avion", "pays"], 4))
print("Procrustes on dictionary: ", trslt2.translate(["homme", "femme", "avion", "pays"], 4))
print("Procrustes on identical words: ", trslt3.translate(["homme", "femme", "avion", "pays"], 4))
#print("GAN: ", trslt_un.translate(["homme", "femme", "avion", "pays"], 4))
