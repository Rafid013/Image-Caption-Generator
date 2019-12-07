from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


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


# calculate the length of the description with the most words
def max_length(descriptions):
    lines,_ = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = VGG19()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


directory = 'COCO'
# load the tokenizer
filename = directory+'/text/train_images.txt'
train = load_set(filename)
train_descriptions = load_clean_descriptions(directory+'/features/descriptions.txt', train)

tokenizer = load(open(directory+'/features/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_cap_size = max_length(train_descriptions)
# load the model
model = load_model(directory+'/models/model_9.h5')
# load and prepare the photograph
img_filename = input("Image Filename with extension: ")
photo_name = 'Generator/'+img_filename
photo = extract_features(photo_name)
# generate description
description = generate_desc(model, tokenizer, photo, max_cap_size).split()
description = description[1:len(description)-1]
description = ' '.join(description)
print(description)

img = load_img(photo_name)
plt.title(description)
plt.imshow(img)
plt.axis('off')
plt.show()