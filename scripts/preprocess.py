
# LIBRARIES

import pandas as pd
import numpy as np

# for text pre-processing
import re, string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist, bigrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree
from sklearn.feature_extraction.text import CountVectorizer
# import fasttext
# path = "/Users/camillecoeurjoly/Documents/vscode_workspace/comment_analysis/data_cleaning_and_text_pre-processing/fasttext/lid.176.bin"
# ft_model = fasttext.load_model(path)


# FUNCTIONS

# Text pre-processing
# function for text cleaning
def clean_text(text):
    text = str(text)
    text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)',' ',text)
    text = text.lower().strip() #removes upper case and leading and trailing whitespaces
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.compile('https?://S+|www.S+').sub('', text) #removing URLs
    text = re.sub('\s+', ' ', text) #replacing double space with single space
    text = re.sub(r'\[[0-9]*\]',' ', text)  #replacing any number by a single space
    text = re.sub(r'\d',' ', text) #get rid of decimal digits
    text = re.sub(r'\s+',' ', text) #get rid of duplicate whitespaces
    text = text.replace('rt', '')
    return text

# function for removing emojis
def termninemoji(text):
    no_emo = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F" 
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
                           "]+", flags = re.UNICODE).sub(r'', text)
    return no_emo

# function for Stop Word removal
def stop(text):
    stop = stopwords.words('english')
    no_stop_words = [word for word in text.split() if word not in stop]
    return ' '.join(no_stop_words)

# function text pre-processing
def preprocess(text):
    text = clean(text)
    text = correct_spelling(text)
    text = termninemoji(text)
    text = stop(text)
    return text

# functions for lemmatization
# getting the tags (word type: verb, noun, adverb etc.)
def get_type(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('DT'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# lemmatization
def lemmatization(text):
    tokens = word_tokenize(text) #tokenizing
    word_and_tag = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(pair[0], pos=get_type(pair[1])) for pair in word_and_tag]
    return ' '.join(lemmatized)

# function for final pre-processing
def final_prep(text):
    if text != None:
        text = clean(text)
        text = correct_spelling(text)
        text = termninemoji(text)
        text = stop(text)
        text = lemmatization(text)
    return text
    
def fast_preprocess(text):

    try:
        text = str(text)
        text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|https?://S+|www.S+',' ', str(text))
        text = text.lower().strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        no_emo = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F" 
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"
            u"\u3030"
                            "]+", flags = re.UNICODE).sub(r'', text)
        stop_words = stopwords.words('english')
        words = no_emo.split()
        no_stop_words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in no_stop_words]
        text = ' '.join(lemmatized_words)

    except Exception as e:
        return repr(e)
    
    return text


