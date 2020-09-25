# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:02:35 2020

@author: xyf11
"""

# predictor.py

import boto3
import pickle
import string
import numpy as np
labels = ["Reply", "Learning being shared", "Question","Announcement", "None","Appreciation"]


class PythonPredictor:
#
    def __init__(self):
        
        
        #s3 = boto3.client("s3")
        #s3.download_file(config["bucket"], config["key"], "new_model.sav")
        self.model = pickle.load(open("new_model.sav", "rb"))

    def preprocess(self,words):
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
        #words = [word for word in words if len(word) > 2]
        return words
    #function to remove stopwords
    def remove_stopwords(self,words):
        stopwords = ['a',  'above', 'after', 'again', 'against',  'an', 'and',   'as', 'at',
   'being', 'below', 'between', 'both',  'by', 
  'down', 'during','abc',
 'each', 'few', 'for', 'from', 'further', 
 'hers', 'herself', 'him', 'himself', 'his', 
  'in', 'into',  'itself',
 "let's",  'more', 'most', "mustn't", 'myself',
 'of', 'off', 'on', 'once', 'other', 'ought', 'ours' 'ourselves', 'out', 'over', 'own',
  "shan't",  "she'd", "she'll", "she's",  'some', 'such', 
 'than', "that's", 'theirs',  'themselves',   "there's",   "they'd", 
 "they'll", "they're", "they've", 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
 '4th', '5th', '6th', '7th', '8th', '9th', '10th']
        words = [word for word in words if not word in stopwords]
        return words
    #function to convert a sentence into list of words
    def tokenize_sentence(self,line):
        words = line[0:len(line)-1].strip().split(" ")
        words = self.preprocess(words)
        words = self.remove_stopwords(words)
        #use functions defined previously
        return words
    
    #function to remove metadata
    def remove_metadata(self,lines):
        for i in range(len(lines)):
            if(lines[i] == '\n'):
                start = i+1
                break
        new_lines = lines[start:]
        return new_lines
    #function to convert a document into list of words
    def tokenize(self,path):
        #load document as a list of lines
        f = open(path, 'r')
        text_lines = f.readlines()    
        #removing the meta-data at the top of each document
        #text_lines = remove_metadata(text_lines)    
        #initiazing an array to hold all the words in a document
        doc_words = []    
        #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
        for line in text_lines:
            doc_words.append(self.tokenize_sentence(line))
        return doc_words
    #a simple helper function to convert a 2D array to 1D, without using numpy
    def flatten(self,list):
        new_list = []
        for i in list:
            for j in i:
                new_list.append(j)
        return new_list
    def predict(self, payload):
        #my_path_test = 'data.txt'
        #test = open(my_path_test,"r")
        list_of_words_test2 = []
        #initialization
        #print(self.tokenize(payload)[0])
        list_of_words_test2.append(self.flatten(self.tokenize(payload)))
        #print(list_of_words_test2,'\n')
        
        #if(len(list_of_words_test2[0]))<10:#condition in case input is too short
         #   print("The input is too short for categorization.\n")
        #else:#begin preprocess the input
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
        new_predict1 = self.model.predict(new_test)
        print('The predicted category of the input is :',new_predict1,'\n')
        return new_predict1
        #label_id = self.model.predict(new_predict1)
        #return label_id
test = PythonPredictor()
payload = 'data.txt'
test.predict(payload)