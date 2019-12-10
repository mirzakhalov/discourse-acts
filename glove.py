import json
import random
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv2D, Input, Embedding, TimeDistributed, Flatten
import numpy
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TweetTokenizer
from nltk import tokenize
import tensorflow as tf
from nltk.corpus import wordnet as wn

tokenizer = TweetTokenizer()
t = Tokenizer()

comments = []
tokens = set()
max_len = 0
labels = []
vocab_size = 0

glove = None
trained_model = None
discourse = ['other', 'agreement', 'announcement', 'appreciation', 'humor', 'answer', 'elaboration', 'negativereaction',
             'question', 'disagreement']


def get_data(filename):
    load_json_data = []
    count = 0
    with open(filename) as jsonfile:
        for line in jsonfile:
            jline = json.loads(line)
            load_json_data.append(jline)
            count += 1
    return load_json_data


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_word2vec(filename):
    word2vec = dict()
    with open(filename) as f:
        for line in f:
            key = line.split(' ',1)[0] # the first word is the key
            value = numpy.array([float(val) for val in line.split(' ')[1:]])
            word2vec[key] = value
            
    return word2vec


def process_body(load_data):
    global comments
    global tokens
    global max_len
    global labels
    count = 0
    for jline in load_data:
        for post in jline['posts']:
            try:
                features = []
                b = post['body']
                #p = f.getParentBody(jline, post['id'])
                label = discourse.index(post['majority_type'])
                t = tokenizer.tokenize(b)
                max_len = max(max_len, len(t))
                t = [get_lemma(token) for token in t]
                for token in t:
                    tokens.add(token)
                comments.append(b)
                labels.append(label)
            except Exception as e:
                count += 1
    print(count)
    print(len(comments))
    print(len(labels))
    print(len(tokens))
    return "Done"

def text2seq():
    global vocab_size
    global t
    global comments
    global max_len
    # prepare tokenizer
    t.fit_on_texts(tokens)
    vocab_size = len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(comments)
    # overriding max_len with 500
    max_len = 500
    comments = pad_sequences(encoded_docs, maxlen=max_len, padding='post')

def init_embeddings():
    global embedding_matrix
    embedding_matrix = numpy.zeros((vocab_size, 100))
    for word, i in t.word_index.items():
     if word in glove:
        embedding_matrix[i] = glove[word]

json_data = get_data("coarse_discourse_dump_reddit.jsonlist")
print(process_body(json_data))
glove = get_word2vec("glove.6B.100d.txt")
text2seq()
init_embeddings()

input = Input(shape=(max_len,))
model = Embedding(vocab_size,100,weights=[embedding_matrix],input_length=max_len)(input)
model =  Bidirectional (LSTM (32,return_sequences=True,dropout=0.50),merge_mode='concat')(model)
model = TimeDistributed(Dense(32,activation='relu'))(model)
model = Flatten()(model)
model = Dense(32,activation='relu')(model)
output = Dense(10,activation='softmax')(model)
model = tf.keras.Model(input,output)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.fit(numpy.array(comments), numpy.array(labels), validation_split=0.25, epochs=10, batch_size=16)

