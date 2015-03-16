#Imports
import os
import sys
import numpy
import glob
import nltk
import re
from scipy.cluster.vq import whiten
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
import authorshipIdentifier
import random

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
#Import data
#TODO: intergrate tweet fetching
trainSource_folder = "./source/train"
#Global declaration
train_tweets = []
#order is important since all resulting files needs to be compared in order
trainSource_files = sorted(glob.glob(os.path.join(trainSource_folder, "tweet_*.txt")))
for tweet_train_files in trainSource_files:
    with open(tweet_train_files) as f:
        train_tweets.append(f.read().replace('\n', ' '))

dataSource_folder = "./source"
#Global declaration
tweets = []
#order is important since all resulting files needs to be compared in order
dataSource_files = sorted(glob.glob(os.path.join(dataSource_folder, "tweet_*.txt")))
for tweet_files in dataSource_files:
    with open(tweet_files) as f:
        tweets.append(f.read().replace('\n', ' '))


testSource_folder = "./source/test"
#Global declaration
test_tweets = []
#order is important since all resulting files needs to be compared in order
testSource_files = sorted(glob.glob(os.path.join(testSource_folder, "tweet_*.txt")))
for tweet_files in testSource_files:
    with open(tweet_files) as f:
        test_tweets.append(f.read().replace('\n', ' '))



######################################################
#                  Lexical Features                  #
######################################################
def getLex(T):
    word_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    tokens = nltk.word_tokenize(T.lower())
    words = word_tokenizer.tokenize(T.lower())
    sentences = sentence_tokenizer.tokenize(T)
	#declare word count per sentence
    words_count_per_sentence = numpy.array([len(word_tokenizer.tokenize(sen)) for sen in sentences])
        #first rule: average words per sentence
    lex0 = words_count_per_sentence.mean()
	#second rule: vocab abundance
    vocab_ab = set(words)
    lex1 = len(vocab_ab) / float(len(words))
	#third rule: average length of words
    lex2 = float(len(T)) / float(len(words))
    lex = [lex0,lex1,lex2]
    return lex

def getPuct(T):
    word_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    tokens = nltk.word_tokenize(T.lower())
    words = word_tokenizer.tokenize(T.lower())
    sentences = sentence_tokenizer.tokenize(T)
	#declare word count per sentence
    words_count_per_sentence = numpy.array([len(word_tokenizer.tokenize(sen)) for sen in sentences])
	#rule 1: number of comms per sentence
    puct0 = (tokens.count(',')+0.1)
	#rule 2: number of ! per sentence
    puct1 = (tokens.count('!')+0.1)
	#rule 3: number of ? per sentence
    puct2 = (tokens.count('?')+0.1)	       
        #rule 4: number of : per sentence
    puct3 = (tokens.count(':')+0.1)

	#Whitening to decorrelate the features
    puct = [lex0,lex1,lex2]
    return puct


def find_url(text):
    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

def getTH(T):
    #first rule: number of hashtags per tweet
    TH0 = T.count('#')
    #second rule: numbre of 'at's per tweet
    TH1 = T.count('@')
    #third rule: a link or pic?
    TH2 = find_url(T)
    TH = [TH0,TH1,TH2]
    return TH


def train_classifier():
    name = "user1"
    trainSet_th = [({'tweethabit' : (v[0],v[1],v[2])}, name) for v in authorshipIdentifier.GenTweetFeatures(train_tweets)]
    trainSet_th += [({'tweethabit' : (t[0],t[1],t[2])}, 'not User1') for t in authorshipIdentifier.GenTweetFeatures(tweets)]
    trainSet_lex = [({'tweetlex' : (l[0],l[1],l[2])}, name) for l in authorshipIdentifier.GenLexicalFeatures(train_tweets)[0]]
    trainSet_lex += [({'tweelex' : (l2[0],l2[1],l2[2])}, 'not User1') for l2 in authorshipIdentifier.GenLexicalFeatures(tweets)[0]]
    trainSet_puct = [({'tweetpunct' : (p[0],p[1],p[2])}, name) for p in authorshipIdentifier.GenLexicalFeatures(train_tweets)[1]]
    trainSet_puct = [({'tweetpunct' : (p2[0],p2[1],p2[2])}, name) for p2 in authorshipIdentifier.GenLexicalFeatures(tweets)[1]]
    
    random.shuffle(trainSet_th)
    random.shuffle(trainSet_lex)
    random.shuffle(trainSet_puct)

    classifier_th = nltk.NaiveBayesClassifier.train(trainSet_th)
    classifier_lex = nltk.NaiveBayesClassifier.train(trainSet_lex)
    classifier_puct = nltk.NaiveBayesClassifier.train(trainSet_puct)
    return classifier_th,classifier_lex,classifier_puct

if __name__ == '__main__':
    cf = train_classifier()
    for test_text in test_tweets:
    	print cf[0].classify({'tweethabit' : getTH(test_text)})
    	print cf[1].classify({'tweetlex' : getLex(test_text)})
    	print cf[2].classify({'tweetpunct' : getPunct(test_text)})













