import numpy as np
import nltk
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


data_folder = "./"
files = sorted(glob.glob(os.path.join(data_folder, "*.txt")))
chapters = []
for fn in files:
    with open(fn) as f:
        chapters.append(f.read().replace('\n', ' '))
all_text = ' '.join(chapters)



def LexicalFeatures():
    """
    Compute feature vectors for word and punctuation features
    """
    num_chapters = len(chapters)
    fvs_lexical = np.zeros((len(chapters), 3), np.float64)
    fvs_punct = np.zeros((len(chapters), 3), np.float64)
    for e, ch_text in enumerate(chapters):
        # note: the nltk.word_tokenize includes punctuation
        tokens = nltk.word_tokenize(ch_text.lower())
        words = word_tokenizer.tokenize(ch_text.lower())
        sentences = sentence_tokenizer.tokenize(ch_text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                       for s in sentences])

        # average number of words per sentence
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[e, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[e, 2] = len(vocab) / float(len(words))

        # Commas per sentence
        fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
        # Semicolons per sentence
        fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
        # Colons per sentence
        fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))

    # apply whitening to decorrelate the features
    fvs_lexical = whiten(fvs_lexical)
    fvs_punct = whiten(fvs_punct)

    print fvs_lexical
    return fvs_lexical, fvs_punct

LexicalFeatures()
