import pandas as pd
import numpy as np
import os
import nltk
import re
import time
import emoji
from nltk.corpus import stopwords
from nltk import SnowballStemmer
import sys

stemmer = SnowballStemmer("english")

nltk.download("stopwords")
eng_stopwords = stopwords.words('english')
add_stopwords = [word.replace('\'', '') for word in eng_stopwords]
add_stopwords += ['im', 'th'] # additional stopwords
stopwords = list(set(eng_stopwords+add_stopwords))

def clean_for_content(string):

    string = string.lower()

    string = re.sub(r'\bhttps?\:\/\/[^\s]+', ' ', string) #remove websites

    # Classic replacements:
    string = re.sub(r'\&gt;', ' > ', string)
    string = re.sub(r'\&lt;', ' < ', string)
    string = re.sub(r'<\s?3', ' ❤ ', string)
    string = re.sub(r'\@\s', ' at ', string)
    string = re.sub(r'(\&(amp)?|amp;)', ' and ', string)
    string = re.sub(r'(\bw\/?\b)', ' with ', string)
    string = re.sub(r'\brn\b', ' right now ', string)
    # string = re.sub(r'\bn\b', ' and ', string) # UNSURE OF THIS ONE! How about n-word? North? etc.
    # replace some user names by real names?

    string = re.sub(r'\s+', ' ', string).strip()

    return string

def clean_for_topic(string, stemmer=stemmer, stopwords=stopwords):

    string = re.sub(r'\b\d\d?\d?\b', ' ', string) # remove 1, 2, 3 digit numbers
    string = re.sub(r'\b\d{4}\d+\b', ' ', string) # remove 5+ digit numbers

    string = re.sub(
        r'[^a-z\d\s\@\#\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]',
        ' ', string
    ) # remove all non alpha-numeric characters + # + @ + emojis

    string = re.sub(r'\b[a-z]\b', ' ', string) # remove 1-letter words

    string = re.sub(r'\b', ' ', string) # separate text and emojis
    string = re.sub(r'\@\s', '@', string) # reform usernames
    string = re.sub(r'\#\s', '#', string) # reform hashtags

    string = emoji.demojize(string)
    string = re.sub(r'\:\:', ': :', string) # separate consecutive emojis
    string = emoji.emojize(string)

    string = string.split()
    string = [elem for elem in string if elem not in stopwords]
    string = [stemmer.stem(elem) if elem[0] not in ['#', '@'] else elem for elem in string] # Do not stem usernames and hashtags
    string = ' '.join(string)

    return string

# TO DO:
# Missing emojis: 🤣, 🤷‍♂️,
# Underscores etc. in usernames

# From "Tweets Classification with BERT in the Field of Disaster Management"
# Texts are lowercased.
# Non-ascii letters, urls, @RT:[NAME], @[NAME] are removed.
# For BERT, an additional [CLS] token is inserted to the beginning of each text.
# Texts with length less than 4 are thrown away.
# No lemmatization is performed and no punctuation mark is removed since pre-trained embeddings are always used.
# No stop-word is removed for fluency purpose.
