# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
import string
#my_path = '20_newsgroups'
#my_path = 'general model'
my_path = 'newdata'
#dataset path
#creating a list of folder names to make valid pathnames later
folders = [f for f in listdir(my_path)]#collecting category names
#print(folders)
files = []#initialization
for folder_name in folders:
    folder_path = join(my_path, folder_name)
    files.append([f for f in listdir(folder_path)])#get all files
#print(sum(len(files[i]) for i in range(20)))
pathname_list = []#initialization
for fo in range(len(folders)):
    for fi in files[fo]:
        pathname_list.append(join(my_path, join(folders[fo], fi)))#get all files
#print(len(pathname_list))
Y = []#initialization
for folder_name in folders:
    folder_path = join(my_path, folder_name)
    num_of_files= len(listdir(folder_path))
    for i in range(num_of_files):
        Y.append(folder_name)#get all files
#print(len(Y))
from sklearn.model_selection import train_test_split
doc_train, doc_test, Y_train, Y_test = train_test_split(pathname_list, Y, random_state=0, test_size=0.25) # train:test = 0.75:0.25
stopwords = ['a',  'above', 'after', 'again', 'against',  'an', 'and',   'as', 'at',
 'be',   'being', 'below', 'between', 'both',  'by', 
  'down', 'during','abc',
 'each', 'few', 'for', 'from', 'further', 
 'hers', 'herself', 'him', 'himself', 'his', 
  'in', 'into',  'its', 'itself',
 "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
 'same', "shan't", 'she', "she'd", "she'll", "she's",  'so', 'some', 'such', 
 'than', 'that',"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", 
 "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
 '4th', '5th', '6th', '7th', '8th', '9th', '10th']
#stopwords to be removed
def preprocess(words):
    #use python's translate function,that maps one set of characters to another
    #create an empty mapping table, the third argument allows us to list all of the characters 
    #to remove during the translation process
    
    #filter out some  unnecessary data like tabs
    table = str.maketrans('', '', '\t')
    words = [word.translate(table) for word in words]
    
    punctuations = (string.punctuation).replace("'", "") 
    # the character: ' appears in a lot of stopwords and changes meaning of words if removed
    #hence it is removed from the list of symbols that are to be discarded from the documents
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in words]
    
    #some white spaces may be added to the list of words, due to the translate function & nature of our documents
    #remove them below
    words = [str for str in stripped_words if str]
    
    #some words are quoted in the documents & as we have not removed ' to maintain the integrity of some stopwords
    #unquote such words below
    p_words = []#initialization
    for word in words:
        if (word[0] and word[len(word)-1] == "'"):
            word = word[1:len(word)-1]
        elif(word[0] == "'"):
            word = word[1:len(word)]
        else:
            word = word
        p_words.append(word)
    
    words = p_words.copy()
        
    #remove just-numeric strings as they do not have any significant meaning in text classification
    words = [word for word in words if not word.isdigit()]
    
    #remove single character strings
    words = [word for word in words if not len(word) == 1]
    
    #after removal of so many characters it may happen that some strings have become blank, remove those
    words = [str for str in words if str]
    
    #normalize the cases of our words
    words = [word.lower() for word in words]
    
    #remove words with only 2 characters
    #words = [word for word in words if len(word) > 2]
    
    return words

#function to remove stopwords

def remove_stopwords(words):
    words = [word for word in words if not word in stopwords]
    return words
#function to convert a sentence into list of words

def tokenize_sentence(line):
    words = line[0:len(line)-1].strip().split(" ")
    words = preprocess(words)
    words = remove_stopwords(words)
    
    return words

#function to remove metadata

#def remove_metadata(lines):
#    for i in range(len(lines)):
#        if(lines[i] == '\n'):
#            start = i+1
#            break
#    new_lines = lines[start:]
#    return new_lines
##function to convert a document into list of words

def tokenize(path):
    #load document as a list of lines
    f = open(path, 'r')
    text_lines = f.readlines()
    
    #removing the meta-data at the top of each document
   # text_lines = remove_metadata(text_lines)
    
    #initiazing an array to hold all the words in a document
    doc_words = []
    
    #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
    for line in text_lines:
        doc_words.append(tokenize_sentence(line))

    return doc_words
#a simple helper function to convert a 2D array to 1D, without using numpy

def flatten(list):
    new_list = []#initialization
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list
print(len(folders))
list_of_words = []#initialization

for document in doc_train:
        list_of_words.append(flatten(tokenize(document)))

#print(len(list_of_words))
#print(len(flatten(list_of_words)))

import numpy as np
np_list_of_words = np.asarray(flatten(list_of_words))
#finding the number of unique words that we have extracted from the documents

words, counts = np.unique(np_list_of_words, return_counts=True)
#print(len(words))
#sorting the unique words according to their frequency
#plot a graph for frequency
freq, wrds = (list(i) for i in zip(*(sorted(zip(counts, words), reverse=True))))
f_o_w = []
n_o_w = []
for f in sorted(np.unique(freq), reverse=True):
    f_o_w.append(f)
    n_o_w.append(freq.count(f))
import matplotlib.pyplot as plt
y = f_o_w
x = n_o_w
plt.xlim(0,250)
plt.xlabel("No. of words")
plt.ylabel("Freq. of words")
plt.plot(x, y)
plt.grid()
plt.show()
#deciding the no. of words to use as feature

n = 5000
features = wrds[0:n]
#print(features)
with open('features.txt', 'w') as filehandle:
    for listitem in features:
        filehandle.write('%s\n' % listitem)
#write into a text file
#creating a dictionary that contains each document's vocabulary and ocurence of each word of the vocabulary 

dictionary = {}#initialization
doc_num = 1
for doc_words in list_of_words:
    #print(doc_words)
    np_doc_words = np.asarray(doc_words)
    w, c = np.unique(np_doc_words, return_counts=True)
    dictionary[doc_num] = {}
    for i in range(len(w)):
        dictionary[doc_num][w[i]] = c[i]
    doc_num = doc_num + 1
#print(dictionary.keys())
#make a 2D array having the frequency of each word of our feature set in each individual documents

X_train = []#initialization
for k in dictionary.keys():
    row = []#initialization
    for f in features:
        if(f in dictionary[k].keys()):
            #if word f is present in the dictionary of the document as a key, its value is copied
            #this gives us no. of occurences
            row.append(dictionary[k][f]) 
        else:
            #if not present, the no. of occurences is zero
            row.append(0)
    X_train.append(row)
#convert the X and Y into np array for concatenation and conversion into dataframe

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
#print(len(X_train))
#print(len(Y_train))
list_of_words_test = []#initialization

for document in doc_test:
        list_of_words_test.append(flatten(tokenize(document)))
dictionary_test = {}#initialization
doc_num = 1
for doc_words in list_of_words_test:
    #print(doc_words)
    np_doc_words = np.asarray(doc_words)
    w, c = np.unique(np_doc_words, return_counts=True)
    dictionary_test[doc_num] = {}
    for i in range(len(w)):
        dictionary_test[doc_num][w[i]] = c[i]
    doc_num = doc_num + 1

#make a 2D array having the frequency of each word of our feature set in each individual documents

X_test = []
for k in dictionary_test.keys():
    row = []
    for f in features:
        if(f in dictionary_test[k].keys()):
            #if word f is present in the dictionary of the document as a key, its value is copied
            #this gives us no. of occurences
            row.append(dictionary_test[k][f]) 
        else:
            #if not present, the no. of occurences is zero
            row.append(0)
    X_test.append(row)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
#print(len(X_test))
#print(len(Y_test))

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()#using naive bayes classifier
print(clf.fit(X_train, Y_train))
Y_predict = clf.predict(X_test)
print(clf.score(X_test, Y_test))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(Y_test, Y_predict))#evaluation result
Y_predict_tr = clf.predict(X_train)
print(clf.score(X_train, Y_train))#score
print(classification_report(Y_train, Y_predict_tr))#evaluation result for train
#print(X_test)
#print(len(X_test[0]))
#print(Y_test)
#print(Y_predict)
#function to create a training dictionary out of the text files for training set, consisiting the frequency of
#words in our feature set (vocabulary) in each class or label of the 20 newsgroup
# save the model to disk
import pickle
#filename = 'finalized_model.sav'
#filename = 'general_model.sav'
filename = 'new_model.sav'#model name to be saved
pickle.dump(clf, open(filename, 'wb'))
#save model to a file
# some time later...
#test
my_path_test = './test_input/data.txt'
#creating a list of folder names to make valid pathnames later
test = open(my_path_test,"r")
list_of_words_test2 = []#initialization

list_of_words_test2.append(flatten(tokenize(my_path_test)))#preprocess
#print(list_of_words_test2)
dictionary_test2 = {}#initialization
doc_num2 = 1
for doc_words in list_of_words_test2:
    #print(doc_words)
    np_doc_words = np.asarray(doc_words)
    w, c = np.unique(np_doc_words, return_counts=True)
    dictionary_test2[doc_num2] = {}
    for i in range(len(w)):
        dictionary_test2[doc_num2][w[i]] = c[i]
    doc_num = doc_num + 1

#make a 2D array having the frequency of each word of our feature set in each individual documents

new_test = []#initialization
for k in dictionary_test2.keys():
    row = []#initialization
    for f in features:
        if(f in dictionary_test2[k].keys()):
            #if word f is present in the dictionary of the document as a key, its value is copied
            #this gives us no. of occurences
            row.append(dictionary_test2[k][f]) 
        else:
            #if not present, the no. of occurences is zero
            row.append(0)
    new_test.append(row)
new_test = np.asarray(new_test)#convert to array
#print(new_test)
new_predict = clf.predict(new_test)#begin predicting
print(new_predict)#prediction result