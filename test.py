import json
import random
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv2D
import numpy
from gensim.models.fasttext import load_facebook_model
from sklearn.model_selection import train_test_split
from features import Features

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import TweetTokenizer
from nltk import tokenize
import tensorflow_hub as hub
import tensorflow as tf


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
    fasttext_model = load_facebook_model('crawl-300d-2M-subword.bin')

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


def set_model(data):
    global trained_model
    input_shape = data[0].shape
    # After trying a bunch of different methods, this one worked the best
    t_model = tf.keras.Sequential([

        Bidirectional(LSTM(100, input_shape=input_shape)),
        # Bidirectional(LSTM(100)),
        # Bidirectional(LSTM(64)),
        # hub_layer,
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(32, activation='relu'),

        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    t_model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])
    trained_model = t_model


def train(data, labels):
    global trained_model
    checkpoint_path = 'training_1/cp.ckpt'
    set_model(data)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    trained_model.fit(data, to_categorical(labels), validation_split=0.1, epochs=10, batch_size=128,
                      callbacks=[cp_callback])


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

global data
global labels




from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.externals import joblib
#
from sklearn.datasets import load_iris
#
from sklearn.model_selection import train_test_split


def run_KNN(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("KNN Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
    print("KNN Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test))

    # Example of making prediction for out of sample data
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = knn.predict(sample)
    # print("Predictions:", preds)

    # saving the model
    joblib.dump(knn, 'knn_model.pkl')

    # To load model use: knn = joblib.load('knn_model.pkl')


def run_SVM(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set
    svm = SVC(gamma='auto', verbose=True, cache_size=101639)
    svm.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("SVM Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
    print("SVM Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test))

    # Example of making prediction for out of sample data
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = svm.predict(sample)
    # print("Predictions:", preds)

    # saving the model
    joblib.dump(svm, 'svm_model.pkl')

    # To load model use: knn = joblib.load('svm_model.pkl')


def run_RandomForest(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set
    rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rf.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("RandomForest Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
    print("RandomForest Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test))

    # Example of making prediction for out of sample data
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = rf.predict(sample)
    # print("Predictions:", preds)

    # saving the model
    joblib.dump(rf, 'rf_model.pkl')

    # To load model use: knn = joblib.load('rf_model.pkl')


def run_MLP(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set
    mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(128,))
    mlp.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("MLP Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
    print("MLP Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test))

    # Example of making prediction for out of sample data
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = mlp.predict(sample)
    # print("Predictions:", preds)

    # saving the model
    joblib.dump(mlp, 'mlp_model.pkl')

    # To load model use: knn = joblib.load('mlp_model.pkl')


def run_LRG(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set
    mlp = LogisticRegression()
    mlp.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("MLP Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
    print("MLP Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test))

    # Example of making prediction for out of sample data
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = mlp.predict(sample)
    # print("Predictions:", preds)

    # saving the model
    joblib.dump(mlp, 'mlp_model.pkl')

    # To load model use: knn = joblib.load('mlp_model.pkl')


def main():
    import numpy
    load_fasttext()
    json_data = get_data("coarse_discourse_dump_reddit.jsonlist")
    data, labels = process_data(json_data)


    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))

    new_data = [numpy.concatenate((numpy.array([datum[0][0]]), numpy.array([datum[1][0]]), numpy.array([datum[2][0]]), numpy.array([datum[3][0]]), numpy.array([datum[4][0]]), numpy.array([datum[5][0]]), numpy.array([datum[6][0]]), numpy.array(datum[7]), numpy.array(datum[8]), numpy.array([datum[9][0]]), numpy.array(datum[10]), numpy.array([datum[11][0]]), numpy.array(datum[12]), numpy.array(datum[13]), numpy.array([datum[14][0]])), axis=None) for datum in data]

    print(numpy.array(new_data).shape)

    X_train, X_test, y_train, y_test = train_test_split(new_data, labels, test_size=0.2, random_state=1)

    data_dict = {"x_tr": X_train, "x_te": X_test, "y_tr": y_train, "y_te": y_test}


    # Iris testing data
    print("#"*90)
    print("#"*90)
    print("\nIRIS DATA INITIAL TESTS *IGNORE*\n")
    print("#"*90)
    print("#"*90)
    print()


    data = data_dict

    print("\nK NEAREST NEIGHBOR:\n")

    run_KNN(data)

    print()
    print("#"*30)
    print("\nRANDOM FOREST:\n")

    run_RandomForest(data)

    print()
    print("#"*30)
    print("\nMULTILAYER PERCEPTRON:\n")

    run_MLP(data)

    print()

    print()
    print("#"*30)
    print("\nLinear Regression:\n")

    run_LRG(data)

    print()
    print("#"*30)
    print("\nSUPPORT VECTOR MACHINE:\n")

    run_SVM(data)

    # ***Important Note*** the saved model data will overwrite everytime you run the same model functions
    # TODO Potential More Complex Model RNN/LSTM


if __name__ == '__main__':
    main()