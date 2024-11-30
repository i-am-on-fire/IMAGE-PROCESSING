# Data Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.style.use('seaborn')
sns.set_style("whitegrid")

# Modelling
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from spherecluster import SphericalKMeans
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy import stats

# Additional
import math
import random
import itertools
import multiprocessing
from tqdm import tqdm
from time import time
import logging
import pickle



product = readTXT("train.txt", start_line = 2) + readTXT("test.txt", start_line = 2)
print(f"Playlist Count: {len(playlist)}")

customer_train, customer_test = train_test_split(product, test_size = 1000,
                                                 shuffle = True, random_state = 123)

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.training_loss = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            current_loss = loss
        else:
            current_loss = loss - self.loss_previous_step
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.training_loss.append(current_loss)
        self.epoch += 1
        self.loss_previous_step = loss

model = Word2Vec(
    size = 40,
    window = 10,
    min_count = 1,
    sg = 1,
    negative = 20,
    workers = multiprocessing.cpu_count()-1)
print(model)

logging.disable(logging.NOTSET) # enable logging
t = time()

model.build_vocab(customer_train)

logging.disable(logging.INFO) # disable logging
callback = Callback() # instead, print out loss for each epoch
t = time()

model.train(customer_train,
            total_examples = model.corpus_count,
            epochs = 100,
            compute_loss = True,
            callbacks = [callback]) 

model.save(MODEL_PATH+"cust2vec.model")

embedding_matrix = model.wv[model.wv.vocab.keys()]
embedding_matrix.shape