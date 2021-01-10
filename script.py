import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Dense, Merge
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import nltk


# Create a dictionary containing all the captions of the images
def compute_captions_dict():
    token = 'data/flickr8k/Flickr8k.token.txt'
    captions = open(token, 'r').read().strip().split('\n')
    d = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0])-2]
        if row[0] in d:
            d[row[0]].append(row[1])
        else:
            d[row[0]] = [row[1]]
    return d

# get a list of the images in l (train/test/validation)
def split_data(l,image_files):
    temp = []
    img = glob.glob(image_files+'*.jpg')
    for i in img:
        if i[len(image_files):] in l:
            temp.append(i)
    return temp

# get all test images
def compute_testimg(image_files):
    img = glob.glob(image_files+'*.jpg')
    test_images_file = 'data/flickr8k/Flickr_8k.testImages.txt'
    test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
    test_img = split_data(test_images,image_files)
    return test_img

# get all test images' ground truth captions
def compute_testcaptions(test_img,image_files):
    test_d = {}
    d = compute_captions_dict()
    for i in test_img:
        if i[len(image_files):] in d:
            test_d[i] = d[i[len(image_files):]]
    return test_d

# define vocabulary
def compute_dictionnaries():
    unique = pickle.load(open('utils/unique.p', 'rb'))
    vocab_size = len(unique)
    word2idx = {val:index for index, val in enumerate(unique)}
    idx2word = {index:val for index, val in enumerate(unique)}
    return vocab_size, word2idx, idx2word

# Aggregate all the necessary parameters for the execution of the notebook
def compute_all(image_files):
    max_len = 40
    embedding_size = 300
    vocab_size, word2idx, idx2word = compute_dictionnaries()
    encoding_test = pickle.load(open('utils/encoded_images_test_inceptionV3.p', 'rb'))
    test_img = compute_testimg(image_files)
    test_captions = compute_testcaptions(test_img,image_files)
    return {'max_len':max_len,
            'embedding_size':embedding_size,
            'vocab_size':vocab_size,
            'word2idx':word2idx,
            'idx2word':idx2word,
            'encoded_test_img':encoding_test,
            'test_img':test_img,
            'test_captions':test_captions}
