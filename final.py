"""This is a sample file for final project. 
It contains the functions that should be submitted,
except all it does is output a random value.
- Dr. Licato"""
import json
import random
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
import numpy
from gensim.models.fasttext import load_facebook_model
from sklearn.model_selection import train_test_split
from features import Features

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import TweetTokenizer
from nltk import tokenize


lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()


trained_model = None
f = Features()
f.test()

global fasttext_model

discourse = ['other', 'agreement', 'announcement', 'appreciation', 'humor', 'answer', 'elaboration', 'negativereaction',
             'question', 'disagreement']


def load_fasttext():
    global fasttext_model
    fasttext_model = load_facebook_model('wiki-news-300d-1M-subword.bin')

def get_data(filename):
    load_json_data = []
    count = 0
    with open(filename) as jsonfile:
        for line in jsonfile:
            jline = json.loads(line)
            load_json_data.append(jline)
            count += 1
    return load_json_data


def process_data(load_data):
    global errors
    global f
    count = 0
    count_no_author = 0
    count_no_title = 0
    process_data_list = []
    process_label_list = []
    for jline in load_data:
        author = None
        if 'author' in jline['posts'][0]:
            author = jline['posts'][0]['author']
        for post in jline['posts']:
            try:
                # Structure
                features = f.getStructureFeatures(jline, post['id'])
                # Content
                if 'body' in post:
                    features.append(fasttext_Vec(post['body']))
                else:
                    features.append(numpy.zeros(300))
                    count_no_title += 1
                    
                # ge66t the vector for the parent body
                features.append(fasttext_Vec(f.getParentBody(jline, post['id'])))
                # Author
                features.append(f.isSameAuthor(jline, post))
                
                if 'author' in post:
                    features.append(fasttext_Vec(post['author']))
                    
                    if author == post['author']:
                        features.append(numpy.full(300, 1.0))
                    else:
                        features.append(numpy.full(300, 0.0))
                else:
                    features.append(numpy.zeros(300))
                    features.append(numpy.zeros(300))
                    count_no_author += 1
                
                
                
                if 'title' in jline:
                    features.append(fasttext_Vec(jline['title']))
                else:
                    features.append(numpy.zeros(300))
                    
                
                    
                    
                    
                # Community
                if 'subreddit' in jline:
                    features.append(fasttext_Vec(jline['subreddit']))
                else:
                    features.append(numpy.zeros(300))
                    
                # Thread
                features += f.thread_info(jline)
                feature_nparr = numpy.array(features)
                label = discourse.index(post['majority_type'])
                process_label_list.append([label])
                process_data_list.append(feature_nparr)
            except Exception as e:
                count += 1
    print("Total exception count: " + str(count))
    print("No authors: " + str(count_no_author))
    print("No titles: " + str(count_no_title))
    process_data_list = numpy.array(process_data_list)
    process_label_list = numpy.array(process_label_list)
    print(process_data_list.shape)
    print(process_label_list.shape)
    print("done processing")
    return process_data_list, process_label_list




def cross_validation(load_data, num_validation=10):
    sum_acc = 0
    data, labels = process_data(load_data)

    for i in range(num_validation):
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.9)
        train(train_data, train_labels)
        curr_acc = test_model(test_data, test_labels)
        print(f'\n\n{i}: Accuracy - {curr_acc}')
        sum_acc += curr_acc
    average_acc = sum_acc / num_validation
    print(f'\n\n\n\nAverage Accuracy: {average_acc}')
    return average_acc

def process(filename):
    load_fasttext()
    json_data = get_data(filename)
    data, labels = process_data(json_data)
    train(data, labels)
    test_model()

def set_model(data):
    global trained_model
    input_shape = data[0].shape
    # After trying a bunch of different methods, this one worked the best
    t_model = tf.keras.Sequential([
        
        Bidirectional(LSTM(100, input_shape=input_shape, return_sequences=True)),
        Bidirectional(LSTM(100)),
        #Bidirectional(LSTM(64)),
        Dropout(0.5),
        #tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    t_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])
    trained_model = t_model

def train(data, labels):
    global trained_model
    checkpoint_path = 'training_1/cp.ckpt'
    set_model(data)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    trained_model.fit(data, to_categorical(labels), validation_split=0.1, epochs=10, batch_size=64, callbacks=[cp_callback])
    
def load_model(checkpoint_dir, data):
    global trained_model
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    set_model(data)
    trained_model.load_weights(latest)


def fasttext_Vec(body):
    global fasttext_model
    global lemmatizer
    global tokenizer
    tokens = tokenizer.tokenize(body)
    output = numpy.zeros(300)
    for token in tokens:
        try:
            output = numpy.add(output, fasttext_model[lemmatizer.lemmatize(token)])
        except KeyError:
            output = numpy.add(output, numpy.zeros(300))
    return output


def test_model(test_data=None, test_labels=None):
    global trained_model
    if test_data is None:
        test_json_data = get_data("testData.jsonlist")
        test_data, test_labels = process_data(test_json_data)
    results = trained_model.evaluate(test_data, to_categorical(test_labels), batch_size=128)
    print('test loss, test acc:', results)
    predictions = trained_model.predict(test_data)
    prediction_ind_list = [numpy.argmax(pred) for pred in predictions]
    test_labels = [label[0] for label in test_labels]
    print('Confusion Matrix: ')
    with tf.Session():
        confusion = tf.confusion_matrix(labels=test_labels, predictions=prediction_ind_list, num_classes=len(discourse))
        confusion = tf.Tensor.eval(confusion, feed_dict=None, session=None)
        print(confusion)
    return results[1]
    
def loadModel():
    process("coarse_discourse_dump_reddit.jsonlist")

def classify(thread):
    num_posts = len(thread['posts'])
    return [random.choice(['question', 'answer', 'appreciation', 'elaboration', 'agreement']) for _ in range(num_posts)]


loadModel()