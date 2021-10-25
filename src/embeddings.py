from language import Language
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess

subprocess.call("./data/Embeddings/fetch_data.sh", shell=True)

print("Embeddings uploading")
en = Language("english", "./data/Embeddings/wiki.en.vec")
fr = Language("french", "./data/Embeddings/wiki.fr.vec")


print("\nEmbeddings additivity")
print("reine+fils-roi :",fr.most_similar(positive=["reine", "fils"], negative=["roi"], n=3)[0])
print("france+london-paris: ", en.most_similar(positive=["france", "london"], negative=["paris"], n=3)[0])
print("his-woman+man: ", en.most_similar(positive=["his", "woman"], negative=["man"] , n=3)[0])
