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
#Tokenizers from NLTK

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
#Import data
#TODO: intergrate tweet fetching
dataSource_folder = "./source"
#Global declaration
tweets = []
#order is important since all resulting files needs to be compared in order
dataSource_files = sorted(glob.glob(os.path.join(dataSource_folder, "tweet_*.txt")))
for tweet_files in dataSource_files:
    with open(tweet_files) as f:
        tweets.append(f.read().replace('\n', ' '))
tweet_all = ' '.join(tweets)
######################################################
#                  Lexical Features                  #
######################################################
def GenLexicalFeatures():
    word_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    num_tweets = len(tweets)
    featureVector_lexi = numpy.zeros((num_tweets,3), numpy.float64)
    featureVector_punct = numpy.zeros((num_tweets,4), numpy.float64)
#Here tokenizer is kind of tricky, nltk tokenizer is used for formal text with correct punctuations, however tweets are different.Other suggestions are: using myTonkenizer with splits by normal sentence terminator + Long space + urls etc. TODO
    for e, tweet_text in enumerate(tweets):
	tokens = nltk.word_tokenize(tweet_text.lower())
	words = word_tokenizer.tokenize(tweet_text.lower())
	sentences = sentence_tokenizer.tokenize(tweet_text)
	#declare word count per sentence
	words_count_per_sentence = numpy.array([len(word_tokenizer.tokenize(sen)) for sen in sentences])
	#print words_count_per_sentence
	#first rule: average words per sentence
	featureVector_lexi[e,0] = words_count_per_sentence.mean()
	#second rule: vocab abundance
	vocab_ab = set(words)
	featureVector_lexi[e,1] = len(vocab_ab) / float(len(words))
	#third rule: average length of words
	featureVector_lexi[e,2] = float(len(tweet_text)) / float(len(words))
	
	#for punctuations
	#rule 1: number of comms per sentence
	featureVector_punct[e,0] = (tokens.count(',')+0.1)
	#rule 2: number of ! per sentence
	featureVector_punct[e,1] = (tokens.count('!')+0.1)
	#rule 3: number of ? per sentence
	featureVector_punct[e,2] = (tokens.count('?')+0.1)	       #rule 4: number of : per sentence
	featureVector_punct[e,3] = (tokens.count(':')+0.1)

	#Whitening to decorrelate the features
	featureVector_lexi = whiten(featureVector_lexi)
	#featureVector_punct = whiten(featureVector_punct)
    #print featureVector_lexi
    #print featureVector_punct
    return featureVector_lexi, featureVector_punct


def find_url(text):
    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

######################################################
#                  Tweet Features                  #
######################################################
def GenTweetFeatures():
    num_tweets = len(tweets)
    featureVector_tw = numpy.zeros((num_tweets,3), numpy.float64)
    for enum, tweet_text in enumerate(tweets):
	#first rule: number of hashtags per tweet
	featureVector_tw[enum,0] = tweet_text.count('#')
	#second rule: numbre of 'at's per tweet
	featureVector_tw[enum,1] = tweet_text.count('@')
	#third rule: a link or pic?
	featureVector_tw[enum,2] = find_url(tweet_text)

	#featureVector_tw = whiten(featureVector_tw)
    #print featureVector_tw
    return featureVector_tw

def GenBagOfWordsFeatures():
    number_top_words = 15
    tokens_all = nltk.word_tokenize(tweet_all)
    freq_dist = nltk.FreqDist(tokens_all)
    vocab = freq_dist.keys()[:number_top_words]

    #use sklearn to create the bag for words feature vector for each tweet
    vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=nltk.word_tokenize)
    featureVector_bow = vectorizer.fit_transform(tweets).toarray().astype(numpy.float64)
    print featureVector_bow
    return featureVector_bow

def GenSyntacticFeatures():
    def token_to_pos(tw):
        tokens = nltk.word_tokenize(tw)
        return [p[1] for p in nltk.pos_tag(tokens)]

    tweet_pos = [token_to_pos(tw) for tw in tweets]
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    featureVector_syn = numpy.array([[tw.count(pos) for pos in pos_list]
                           for tw in tweet_pos]).astype(numpy.float64)

    # normalise by dividing each row by number of tokens in the chapter
    featureVector_syn /= numpy.c_[numpy.array([len(tw) for tw in tweet_pos])]

    return featureVector_syn

def TweetAuthorship(featureSet,num_clust,num_docs):
    """
    Use k-means clustering to fit a model
    """
    km = KMeans(n_clusters=num_clust, init='k-means++', n_init=num_docs, verbose=0)
    km.fit(fvs)

    return km


if __name__ == '__main__':
    featureSets = list(GenLexicalFeatures())
    featureSets.append(GenTweetFeatures())
    #featureSets.append(GenBagOfWordsFeatures())
    featureSets.append(GenSyntacticFeatures())
    classifications = [TweetAuthorship(featureSets,3,150).labels_ for fvs in featureSets]
    for results in classifications:
	#if results[0] == 0: results = 1 - results
        print(' '.join([str(a) for a in results])) 















