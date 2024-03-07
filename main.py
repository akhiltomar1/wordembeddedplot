import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
import matplotlib.pyplot as plt

file = open('royal_data.txt','r')
royal_data = file.readlines()
file.close()
print(royal_data)

for i in range(len(royal_data)):
    royal_data[i] = royal_data[i].lower().replace('\n','')

print(royal_data)

stopwords = ['the', 'is', 'will', 'be', 'a', 'only', 'can', 'their', 'now', 'and', 'at', 'it']

filter_data = []

for sent in royal_data:
    temp = []
    for word in sent.split():
        if word not in stopwords:
            temp.append(word)
    filter_data.append(temp)

print(filter_data)

bigrams = []

for word_list in filter_data:
    for i in range(len(word_list) - 1):
        for j in range(i+1, len(word_list)):
            bigrams.append([word_list[i], word_list[j]])
            bigrams.append([word_list[j], word_list[i]])

print(bigrams)


#Vocab
all_words = [] #unique words

for sent in filter_data:
    all_words.extend(sent)

all_words = list(set(all_words))

all_words.sort()

print(all_words)
print(len(all_words))

#onehot encoding

words_dict= {}

counter = 0
for word in all_words:
    words_dict[word] = counter
    counter+=1

print(words_dict)

onehot_data = np.zeros((len(all_words),len(all_words)))

for i in range(len(all_words)):
    onehot_data[i][i] = 1

print(onehot_data)

onehot_dict = {}

for i in range(len(all_words)):
    onehot_dict[all_words[i]] = onehot_data[i]

for word in onehot_dict:
    print(word,":",onehot_dict[word])

X = []
Y = []

for bi in bigrams:
    X.append(onehot_dict[bi[0]])
    Y.append(onehot_dict[bi[1]])

X = np.array(X)
Y = np.array(Y)


#MODEL

model  = Sequential()

vocal_size = len(onehot_data[0])
embed_size = 2

model.add(Input(shape = (12,)))
model.add(Dense(embed_size, activation = 'linear'))
model.add(Dense(vocal_size, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

model.fit(X, Y, epochs = 1000)

weights = model.get_weights()[0]

word_embeddings = {}
for word in all_words:
    word_embeddings[word] = weights[words_dict[word]]

for word in list(words_dict.keys()):
    coord = word_embeddings.get(word)
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, (coord[0], coord[1]))

plt.savefig('img.jpg')
