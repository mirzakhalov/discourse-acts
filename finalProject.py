"""This is a sample file for final project. 
It contains the functions that should be submitted,
except all it does is output a random value.
- Dr. Licato"""
import json
import random
from nltk.tokenize import TweetTokenizer
from nltk import tokenize
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
import numpy
import gensim.models.keyedvectors as word2vec
from gensim.models.fasttext import load_facebook_model
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()

global w2vModel
global fasttext_model

discourse = ['other', 'agreement', 'announcement', 'appreciation', 'humor', 'answer', 'elaboration', 'negativereaction',
             'question', 'disagreement']
trained_model = None

def load_fasttext():
    global fasttext_model
    fasttext_model = load_facebook_model('wiki-news-300d-1M-subword.bin')

def load_word2vec():
    global w2vModel
    w2vModel = word2vec.KeyedVectors.load_word2vec_format \
        ("GoogleNews-vectors-negative300.bin", binary=True)
# json_data = get_data("coarse_discourse_dump_reddit.jsonlist")
def get_data(filename):
    load_json_data = []
    count = 0
    with open(filename) as jsonfile:
        for line in jsonfile:
            jline = json.loads(line)
            load_json_data.append(jline)
            count += 1
    return load_json_data
# data, labels = process_data(json_data)
def process_data(load_data):
    count = 0
    process_data_list = []
    process_label_list = []
    for jline in load_data:
        for post in jline['posts']:
            try:
                # Structure
                features = getStructureFeatures(jline, post['id'])
                # Content
                features.append(fasttext_Vec(getBodyFromID(jline, post['id'])))
                # Author
                features.append(isSameAuthor(jline, post))
                # Community
                features.append(fasttext_Vec(jline['subreddit']))
                # Thread
                features += thread_info(jline)
                feature_nparr = numpy.array(features)
                label = discourse.index(post['majority_type'])
                process_label_list.append([label])
                process_data_list.append(feature_nparr)
            except:
                count += 1
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
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8)
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
        Bidirectional(LSTM(64, input_shape=input_shape, return_sequences=True)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    t_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])
    trained_model = t_model
# train(data, labels)
def train(data, labels):
    global trained_model
    checkpoint_path = 'training_1/cp.ckpt'
    set_model(data)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    trained_model.fit(data, to_categorical(labels), epochs=50, batch_size=32, callbacks=[cp_callback])
def load_model(checkpoint_dir, data):
    global trained_model
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    set_model(data)
    trained_model.load_weights(latest)
def thread_info(thread):
    output = []
    unique_reply_dict = {}
    for thread_post in thread['posts']:
        if 'in_reply_to' in thread_post:
            if thread_post['in_reply_to'] in unique_reply_dict:
                unique_reply_dict[thread_post['in_reply_to']] += 1
            else:
                unique_reply_dict[thread_post['in_reply_to']] = 1
    branch_num = len(unique_reply_dict)
    is_self_post = 1 if 'is_self_post' in thread and thread['is_self_post'] else 0
    # Total Number of posts
    output.append(numpy.full(300, 1.0 * len(thread['posts'])))
    # Number of unique branches
    output.append(numpy.full(300, 1.0 * branch_num))
    # Average Length of branches
    output.append(numpy.full(300, 1.0 * average_branches(unique_reply_dict, branch_num)))
    # Whether it is a self post
    output.append(numpy.full(300, 1.0 * is_self_post))
    return output
def average_branches(branch_obj, branch_num):
    sum_of_branch = sum([branch_obj[key] for key in branch_obj])
    return sum_of_branch / branch_num
def isSameAuthor(thread, post):
    try:
        reply_id = post['in_reply_to']
        if getAuthorFromID(thread, reply_id) == post['author']:
            return numpy.full(300, 1.0)
        else:
            return numpy.full(300, 0.0)
    except:
        return numpy.full(300, 0.0)
def getAuthorFromID(thread, target_id):
    for post in thread['posts']:
        if post['id'] == target_id:
            try:
                return post['author']
            except:
                return None
# returns body given the post id
def getBodyFromID(thread, target_id):
    for post in thread['posts']:
        if post['id'] == target_id:
            return post['body']
# use doc2vec on the result of the tokenizer
# def doc2Vec(body):
#     return model.infer_vector(tokenizer.tokenize(body))
# use doc2vec on the result of the tokenizer
def word2Vec(body):
    global w2vModel
    global lemmatizer
    tokens = tokenizer.tokenize(body)
    output = numpy.zeros(300)
    for token in tokens:
        try:
            output = numpy.add(output, w2vModel[lemmatizer.lemmatize(token)])
        except KeyError:
            output = numpy.add(output, numpy.zeros(300))
    return output
def fasttext_Vec(body):
    global fasttext_model
    tokens = tokenizer.tokenize(body)
    output = numpy.zeros(300)
    for token in tokens:
        try:
            output = numpy.add(output, fasttext_model[lemmatizer.lemmatize(token)])
        except KeyError:
            output = numpy.add(output, numpy.zeros(300))
    return output
# get the depth of the comment
def getStructureFeatures(thread, target_id, parent=False):
    output = []
    for post in thread['posts']:
        if post['id'] == target_id:
            # get the depth
            output.append(getDepth(post, 300))
            # get the character count
            output.append(numpy.full(300, 1.0 * len(post['body'])))
            # get the word count
            output.append(numpy.full(300, 1.0 * len(tokenizer.tokenize(post['body']))))
            # get the sentence count
            output.append(numpy.full(300, 1.0 * len(tokenize.sent_tokenize(post['body']))))
            if not parent:
                try:
                    parent = getStructureFeatures(thread, post['in_reply_to'], True)
                    output = output + parent
                except:
                    # this is the case when the comment is the first one
                    for i in range(0, 4):
                        output += [numpy.zeros(300)]
            return output
def getDepth(post, length):
    try:
        return numpy.full(length, 1.0 * post['post_depth'])
    except:
        return numpy.zeros(length)
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
    import time
    #global trained_model
    #latest = tf.train.latest_checkpoint("training_1/cp.ckpt")
    #set_model()
    #trained_model.load_weights(latest)
    print("Loading fasttext")
    start = time.time()
    load_fasttext()
    print(time.time() - start)
	

def classify(thread):

	numPosts = len(thread['posts'])
	return [random.choice(['question', 'answer', 'appreciation', 'elaboration', 'agreement']) for i in range(numPosts)]


loadModel()
