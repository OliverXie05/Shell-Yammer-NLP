# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:42:14 2019

@author: xyf11
"""

from os import listdir
from os.path import isfile, join
import string
from sklearn.datasets import load_files
import os
import pickle
import numpy as np
#filename = 'finalized_model.sav'
#filename1 = 'general_model.sav'
filename1 = 'specific_model.sav'
#model name
#clf1 = pickle.load(open(filename1, 'rb'))
clf1 = pickle.load(open(filename1, 'rb'))
# load the model from disk
#print(test.read())
stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
 'each', 'few', 'for', 'from', 'further', 
 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",
 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
 "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
 'than', 'that',"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", 
 "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
 "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",'will', 'with', "won't", 'would', "wouldn't", 
 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
 '4th', '5th', '6th', '7th', '8th', '9th', '10th']
#stopwards that will be ignored in the input
def preprocess(words):
    #use python's translate function,that maps one set of characters to another
    #create an empty mapping table 
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
    p_words = []
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
    words = [word for word in words if len(word) > 2]
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
    #use functions defined previously
    return words

#function to remove metadata
def remove_metadata(lines):
    for i in range(len(lines)):
        if(lines[i] == '\n'):
            start = i+1
            break
    new_lines = lines[start:]
    return new_lines
#function to convert a document into list of words
def tokenize(path):
    #load document as a list of lines
    f = open(path, 'r')
    text_lines = f.readlines()    
    #removing the meta-data at the top of each document
    #text_lines = remove_metadata(text_lines)    
    #initiazing an array to hold all the words in a document
    doc_words = []    
    #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
    for line in text_lines:
        doc_words.append(tokenize_sentence(line))
    return doc_words
#a simple helper function to convert a 2D array to 1D, without using numpy
def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list
my_path_test = './test_input/hockey.txt'
#creating a list of folder names to make valid pathnames later
test = open(my_path_test,"r")
list_of_words_test2 = []
#initialization
list_of_words_test2.append(flatten(tokenize(my_path_test)))
print(list_of_words_test2)
if(len(list_of_words_test2[0]))<10:#condition in case input is too short
    print("The input is too short for Tier 2 categorization.\n")
else:#begin preprocess the input
    dictionary_test2 = {} #initialization
    doc_num2 = 1
    for doc_words in list_of_words_test2:
        #print(doc_words)
        np_doc_words = np.asarray(doc_words)
        w, c = np.unique(np_doc_words, return_counts=True)
        dictionary_test2[doc_num2] = {}
        for i in range(len(w)):
            dictionary_test2[doc_num2][w[i]] = c[i]
        doc_num2 = doc_num2 + 1
    #now we make a 2D array having the frequency of each word of our feature set in each individual documents
    # define an empty list
    features = []
    # open file and read the content in a list
    with open('features.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
    
            # add item to the list
            features.append(currentPlace)
    new_test = [] #initialization
    for k in dictionary_test2.keys():
        row = [] #initialization
        for f in features:
            if(f in dictionary_test2[k].keys()):
                #if word f is present in the dictionary of the document as a key, its value is copied
                #this gives us no. of occurences
                row.append(dictionary_test2[k][f]) 
            else:
                #if not present, the no. of occurences is zero
                row.append(0)
        new_test.append(row)
    #convert into array
    new_test = np.asarray(new_test)
   # print(new_test)
    #print(max(new_test[0]))
    #begin predicting
    new_predict1 = clf1.predict(new_test)
    #print(new_predict1)
    if new_predict1 == 'atheism':
        general_predict = 'society'
    elif new_predict1 =='autos':
        general_predict = 'transportation'
    elif new_predict1 =='baseball':
        general_predict = 'sport'
    elif new_predict1 =='crypt':
        general_predict = 'science'
    elif new_predict1 =='electronics':
        general_predict = 'science'
    elif new_predict1 =='forsale':
        general_predict = 'advertisement'
    elif new_predict1 =='graphics':
        general_predict = 'computer'
    elif new_predict1 =='guns':
        general_predict = 'politics'
    elif new_predict1 =='hockey':
        general_predict = 'sport'
    elif new_predict1 =='medical':
        general_predict = 'science'
    elif new_predict1 =='mideast':
        general_predict = 'politics'
    elif new_predict1 =='motorcycles':
        general_predict = 'transportation'
    elif new_predict1 =='os.windows':
        general_predict = 'computer'
    elif new_predict1 =='religion':
        general_predict = 'society'
    elif new_predict1 =='religion.misc':
        general_predict = 'society'
    elif new_predict1 =='space':
        general_predict = 'science'
    elif new_predict1 =='sys.hardware':
        general_predict = 'computer'
    elif new_predict1 =='system.pc.hardware':
        general_predict = 'computer'
    elif new_predict1 =='talk.misc':
        general_predict = 'politics'
    elif new_predict1 =='windows.x':
        general_predict = 'computer'
 #a simple mapping for general prediction
    
    print('The predicted first-level category of the input is :',general_predict,'\n')
    print('The predicted second-level category of the input is :',new_predict1,'\n')
    #print output
    #ubuntu query