# from numpy import array
import numpy as np
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import random


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions


# load photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    word_count_threshold = 10
    word_counts = {}

    for key in descriptions.keys():
        sentences = descriptions[key]
        for sent in sentences:
            for w in sent.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc, vocab


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    _,lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines,_ = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


# define the captioning model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
    # loop for ever over images
    count = 0
    x_1 = None
    x_2 = None
    y = None
    keys = list(descriptions.keys())
    while 1:
        random.shuffle(keys)
        for key in keys:
            desc_list = descriptions[key]
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
            if count == 0:
                x_1 = in_img
                x_2 = in_seq
                y = out_word
            else:
                x_1 = np.append(x_1, in_img,axis =0)
                x_2 = np.append(x_2, in_seq, axis =0)
                y = np.append(y, out_word, axis =0 )
            count += 1
            if count == 16:
                yield [[x_1, x_2], y]
                #print(x_1.shape)
                x_1 = None
                x_2 = None
                y = None
                count = 0


random.seed(1)
# load training dataset (6K)
directory = 'COCO'
filename = directory +'/text/train_images.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions(directory+'/features/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features(directory+'/features/features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# load validation set
filename = directory+'/text/validation_images.txt'
validation = load_set(filename)
print('Dataset: %d' % len(validation))
# descriptions
validation_descriptions = load_clean_descriptions(directory+'/features/descriptions.txt', validation)
print('Descriptions: validation=%d' % len(validation_descriptions))
# photo features
validation_features = load_photo_features(directory+'/features/features.pkl', validation)
print('Photos: validation=%d' % len(validation_features))


# define the model
model = define_model(vocab_size, max_length)
# model = load_model(directory+'/models/prev/model_8.h5')
# train the model, run epochs manually and save after each epoch
epochs = 15
steps = int(len(train_descriptions)/16)
validation_steps = int(len(validation_descriptions)/16)

best_model = None
min_loss = None


for i in range(epochs):
    # create the data generator
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    validation_generator = data_generator(validation_descriptions, validation_features, tokenizer, max_length)
    # fit for one epoch
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    loss = model.evaluate_generator(validation_generator, steps=validation_steps, verbose=1)

    if best_model is None or loss < min_loss:
        best_model = model
        min_loss = loss
        best_model.save(directory + '/models/best_model.h5')
    # save model
    model.save(directory+'/models/model_' + str(i) + '.h5')
