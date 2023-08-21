import spacy
import re
from gensim.models import Word2Vec

nlp = spacy.load('en_core_web_sm')
wv = Word2Vec.load("./models/word2vec.model").wv

def tokenizer(sentence):
    doc = nlp(sentence)
    word_list = [token.lemma_ for token in doc[:-3]]
    
    return word_list

def vectorizer(token_list):
    vector=0
    for token in token_list:
        vector += wv[token]
    return vector / len(token_list)

